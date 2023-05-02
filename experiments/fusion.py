"""
Late fusion experiments.

This experiment is a simple late fusion experiment.

The experiment can be parameterized using the following command line arguments:

- ``image_encoder_backbone``: The backbone to use for the image encoder.
- ``text_encoder_backbone``: The backbone to use for the text encoder.
- ``fusion``: The fusion model to use.

"""
import lightning.pytorch as pl
import typer
import wandb

from callbacks.classification import LogClassificationPredictionSamplesCallback
from datamodules.vqa_v2 import VqaV2SampleDataModule
from lightningmodules.classification import MultiModalClassificationModule
from loggers.wandb import get_lightning_logger
from models.backbones import AvailableBackbones
from models.backbones import registry as backbones_config_registry
from models.classifiers import default_classifier_factory
from models.fusions import AvailableFusionModels, BaseFusionModel
from models.fusions import registry as fusions_registry
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace
from utils.registry import initialize_registries
from utils.torch import (
    backbone_name_to_kebab_case,
    ensure_reproducibility,
    freeze_model_parameters,
)


@initialize_registries()
def experiment(
    image_encoder_backbone: AvailableBackbones = AvailableBackbones.RESNET,
    text_encoder_backbone: AvailableBackbones = AvailableBackbones.BERT,
    fusion: AvailableFusionModels = AvailableFusionModels.CAT,
):
    """
    Run the simple concatenation fusion experiment.

    :param image_encoder_backbone: The name of the backbone model
                                    to use for the image encoder.
    :param text_encoder_backbone: The name of the backbone model
                                    to use for the text encoder.
    :param fusion: The name of the fusion model to use.
    :return: None.
    """
    # Ensure reproducibility.
    ensure_reproducibility()

    # Login to wandb.
    wandb.login()

    # Get the backbone configs.
    image_encoder_config = backbones_config_registry.get(image_encoder_backbone)
    text_encoder_config = backbones_config_registry.get(text_encoder_backbone)

    # Get the fusion model factory.
    fusion_model_factory: type[BaseFusionModel] = fusions_registry.get(fusion)

    # Get the backbone models and pre-processors
    image_processor = image_encoder_config.get_image_processor()
    image_encoder = image_encoder_config.get_model()
    image_representation_size = image_encoder_config.get_image_representation_size()

    text_processor = text_encoder_config.get_tokenizer()
    text_encoder = text_encoder_config.get_model()
    text_representation_size = text_encoder_config.get_text_representation_size()

    # Freeze the backbone models.
    freeze_model_parameters(image_encoder)
    freeze_model_parameters(text_encoder)

    # Initialize the logger.
    logger = get_lightning_logger(
        "-".join(
            [
                f"{fusion}",
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
        image_encoder_config=image_encoder_config,
        text_encoder_config=text_encoder_config,
        batch_size=64,
    )

    # Create a model.
    model = MultiModalClassificationModule(
        fusion=fusion_model_factory(
            image_representation_size=image_representation_size,
            text_representation_size=text_representation_size,
            final_representation_size=768,
        ),
        classifier=default_classifier_factory(classes_num=len(answer_space)),
        classes_num=len(answer_space),
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_processor=image_processor,
        tokenizer=text_processor,
        image_encoder_config=image_encoder_config,
        text_encoder_config=text_encoder_config,
    )

    # Train, test, validate and predict.
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)
    trainer.validate(datamodule=data_module)
    trainer.predict(datamodule=data_module)


if __name__ == "__main__":
    typer.run(experiment)
