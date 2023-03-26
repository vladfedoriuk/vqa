"""
Simple fusion experiment.

This experiment is a simple concatenation fusion model.

The experiment can be parameterized using the following command line arguments:

- ``image_encoder_backbone``: The backbone to use for the image encoder.
- ``text_encoder_backbone``: The backbone to use for the text encoder.

"""
import lightning.pytorch as pl
import typer
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

import wandb
from callbacks.classification import LogClassificationPredictionSamplesCallback
from datamodules.vqa_v2 import VqaV2SampleDataModule
from lightningmodules.classification import MultiModalClassificationModule
from loggers.wandb import get_lightning_logger
from models.fusion.simple_cat import SimpleCatFusionModel
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace
from utils.torch import ensure_reproducibility, freeze_model_parameters


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

    # Initialize the logger.
    logger = get_lightning_logger(
        f"simple-cat-fusion-{image_encoder_backbone}-{text_encoder_backbone}"
    )

    # Initialize the answer space.
    answer_space = VqaV2SampleAnswerSpace()

    # Initialize the tokenizer and model for the text encoder.
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_backbone)
    text_encoder = AutoModel.from_pretrained(text_encoder_backbone)

    # Freeze the text encoder parameters.
    freeze_model_parameters(text_encoder)

    # Initialize the image processor and model for the image encoder.
    image_processor = AutoImageProcessor.from_pretrained(image_encoder_backbone)
    image_encoder = AutoModel.from_pretrained(image_encoder_backbone)

    # Freeze the image encoder parameters.
    freeze_model_parameters(image_encoder)

    # Create a trainer.
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[
            LogClassificationPredictionSamplesCallback(
                logger=logger,
                answer_space=answer_space,
            )
        ],
    )

    # Create a data module.
    data_module = VqaV2SampleDataModule(
        tokenizer=tokenizer,
        image_processor=image_processor,
        answer_space=answer_space,
        batch_size=64,
    )

    # Create a model.
    model = MultiModalClassificationModule(
        classifier=SimpleCatFusionModel(
            answers_num=len(answer_space),
        ),
        text_encoder=text_encoder,
        image_encoder=image_encoder,
    )

    # Train, test, validate and predict.
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)
    trainer.validate(datamodule=data_module)
    trainer.predict(datamodule=data_module)


if __name__ == "__main__":
    typer.run(experiment)
