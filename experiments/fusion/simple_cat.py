"""
Simple fusion experiment.

This experiment is a simple concatenation fusion model.

The experiment can be parameterized using the following command line arguments:

- ``image_encoder_backbone``: The backbone to use for the image encoder.
- ``text_encoder_backbone``: The backbone to use for the text encoder.

"""
import lightning.pytorch as pl
import typer

import wandb
from callbacks.classification import LogClassificationPredictionSamplesCallback
from datamodules.vqa_v2 import VqaV2SampleDataModule
from lightningmodules.classification import MultiModalClassificationModule
from loggers.wandb import get_lightning_logger
from models.backbones import registry as backbones_config_registry
from models.fusion.simple_cat import SimpleCatFusionModel
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace
from utils.registry import initialize_registries
from utils.torch import (
    backbone_name_to_kebab_case,
    ensure_reproducibility,
    freeze_model_parameters,
)


@initialize_registries()
def experiment(
    image_encoder_backbone: str = "google/vit-base-patch16-224-in21k",
    text_encoder_backbone: str = "bert-base-uncased",
):
    """
    Run the simple concatenation fusion experiment.

    :param image_encoder_backbone: The name of the backbone model
                                    to use for the image encoder.
    :param text_encoder_backbone: The name of the backbone model
                                    to use for the text encoder.
    :return: None.
    """
    # Ensure reproducibility.
    ensure_reproducibility()

    # Login to wandb.
    wandb.login()

    # Get the backbone configs.
    image_encoder_backbone_config = backbones_config_registry.get_by_name(image_encoder_backbone)()
    text_encoder_backbone_config = backbones_config_registry.get_by_name(text_encoder_backbone)()

    # Get the backbone models and pre-processors
    image_processor = image_encoder_backbone_config.get_image_processor()
    image_encoder = image_encoder_backbone_config.get_model()
    image_representation_size = image_encoder_backbone_config.get_image_representation_size()

    text_processor = text_encoder_backbone_config.get_tokenizer()
    text_encoder = text_encoder_backbone_config.get_model()
    text_representation_size = text_encoder_backbone_config.get_text_representation_size()

    # Freeze the backbone models.
    freeze_model_parameters(image_encoder)
    freeze_model_parameters(text_encoder)

    # Initialize the logger.
    logger = get_lightning_logger(
        "-".join(
            [
                "simple-cat-fusion",
                f"{backbone_name_to_kebab_case(image_encoder_backbone)}",
                f"{backbone_name_to_kebab_case(text_encoder_backbone)}",
            ]
        )
    )

    # Initialize the answer space.
    answer_space = VqaV2SampleAnswerSpace()

    # Create a trainer.
    # TODO: Try deterministic=True.
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        devices=1,
        num_nodes=1,
        strategy="ddp",
        max_epochs=10,
        log_every_n_steps=4,
        callbacks=[
            LogClassificationPredictionSamplesCallback(
                logger=logger,
                answer_space=answer_space,
            )
        ],
    )

    # Create a data module.
    data_module = VqaV2SampleDataModule(
        image_processor=image_processor,
        tokenizer=text_processor,
        answer_space=answer_space,
        batch_size=64,
    )

    # Create a model.
    model = MultiModalClassificationModule(
        classifier=SimpleCatFusionModel(
            answers_num=len(answer_space),
            image_representation_size=image_representation_size,
            text_representation_size=text_representation_size,
        ),
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_processor=image_processor,
        tokenizer=text_processor,
        image_encoder_backbone_config=image_encoder_backbone_config,
        text_encoder_backbone_config=text_encoder_backbone_config,
    )

    # Train, test, validate and predict.
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)
    trainer.validate(datamodule=data_module)
    trainer.predict(datamodule=data_module)


if __name__ == "__main__":
    typer.run(experiment)
