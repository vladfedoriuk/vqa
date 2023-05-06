"""
Late fusion experiments.

This experiment is a simple late fusion experiment.

The experiment can be parameterized using the following command line arguments:

- ``image_encoder_backbone``: The backbone to use for the image encoder.
- ``text_encoder_backbone``: The backbone to use for the text encoder.
- ``multimodal_backbone``: The multimodal backbone to use.
- ``fusion``: The fusion model to use.
- ``dataset``: The dataset to use.
- ``epochs``: The number of epochs to train for.

"""
from typing import cast

import lightning.pytorch as pl
import typer
import wandb

from callbacks.checkpoints import get_model_checkpoint
from callbacks.sample import PredictionSamplesCallback
from collators import ClassificationCollator
from collators import registry as collators_registry
from datamodules.classification import MultiModalClassificationDataModule
from lightningmodules.classification import MultiModalClassificationModule
from loggers.wandb import get_lightning_logger
from models.backbones import AvailableBackbones, prepare_backbones
from models.classifiers import default_classifier_factory
from models.fusions import AvailableFusionModels
from models.fusions import registry as fusions_registry
from utils.config import load_env_config
from utils.datasets import AvailableDatasets
from utils.datasets import registry as datasets_registry
from utils.datasets.answer_space import registry as answer_space_registry
from utils.registry import initialize_registries
from utils.torch import (
    backbone_name_to_kebab_case,
    ensure_reproducibility,
    freeze_model_parameters,
)


@initialize_registries()
@load_env_config()
def experiment(
    image_encoder_backbone: AvailableBackbones | None = None,
    text_encoder_backbone: AvailableBackbones | None = None,
    multimodal_backbone: AvailableBackbones | None = None,
    fusion: AvailableFusionModels = AvailableFusionModels.CAT,
    dataset: AvailableDatasets = AvailableDatasets.VQA_V2_SAMPLE,
    epochs: int = 10,
):
    """
    Run the simple concatenation fusion experiment.

    :param image_encoder_backbone: The name of the backbone model
                                    to use for the image encoder.
    :param text_encoder_backbone: The name of the backbone model
                                    to use for the text encoder.
    :param multimodal_backbone: The name of the multimodal backbone model.
    :param fusion: The name of the fusion model to use.
    :param dataset: The name of the dataset to use.
    :param epochs: The number of epochs to train for.
    :return: None.
    """
    # Ensure reproducibility.
    ensure_reproducibility()

    # Login to wandb.
    wandb.login()

    if not any([image_encoder_backbone, text_encoder_backbone, multimodal_backbone]):
        print("You must specify at least one backbone.")
        raise typer.Abort()

    # Prepare the backbones.
    backbones_data = prepare_backbones(
        {
            "image_backbone": image_encoder_backbone,
            "text_backbone": text_encoder_backbone,
            "multimodal_backbone": multimodal_backbone,
        }
    )

    # Get the fusion model factory.
    fusion_model_factory = fusions_registry.get(fusion)

    # Freeze the backbone models.
    freeze_model_parameters(backbones_data["image_encoder"])
    freeze_model_parameters(backbones_data["text_encoder"])

    # Initialize the logger.
    logger = get_lightning_logger(
        f"{fusion.value}-" f"{backbone_name_to_kebab_case(image_encoder_backbone.value)}-"
        if image_encoder_backbone
        else "" f"{backbone_name_to_kebab_case(text_encoder_backbone.value)}-"
        if text_encoder_backbone
        else "" f"{backbone_name_to_kebab_case(multimodal_backbone.value)}-"
        if multimodal_backbone
        else ""
    )

    # Initialize the answer space.
    answer_space = answer_space_registry.get(dataset)()

    # Get the collator class.
    collator_cls = collators_registry.get(dataset)

    # Get the datasets loading function.
    datasets_loading_fn = datasets_registry.get(dataset)

    if not issubclass(collator_cls, ClassificationCollator):
        print(f"Expected a classification collator for {dataset}, got {collator_cls}.")
        raise typer.Abort()

    # Create a trainer.
    # TODO: Try deterministic=True.
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        devices=1,
        num_nodes=1,
        strategy="ddp",
        max_epochs=epochs,
        callbacks=[
            PredictionSamplesCallback(
                answer_space=answer_space,
                dataset=dataset,
            ),
            get_model_checkpoint(
                dataset=dataset,
                image_encoder=image_encoder_backbone,
                text_encoder=text_encoder_backbone,
                multimodal_encoder=multimodal_backbone,
                fusion=fusion,
            ),
        ],
    )

    # Create a data module.
    data_module = MultiModalClassificationDataModule(
        image_processor=backbones_data["image_processor"],
        tokenizer=backbones_data["tokenizer"],
        answer_space=answer_space,
        image_encoder_config=backbones_data["image_backbone_config"],
        text_encoder_config=backbones_data["text_backbone_config"],
        collator_cls=cast(type[ClassificationCollator], collator_cls),
        datasets_loading_function=datasets_loading_fn,
        batch_size=64,
    )
    # TODO: Fix image_encoder_config and text_encoder_config names.
    # Create a model.
    model = MultiModalClassificationModule(
        fusion=fusion_model_factory(
            image_representation_size=backbones_data["image_representation_size"],
            text_representation_size=backbones_data["text_representation_size"],
            final_representation_size=768,
        ),
        classifier=default_classifier_factory(input_dim=768, classes_num=len(answer_space)),
        classes_num=len(answer_space),
        image_encoder=backbones_data["image_encoder"],
        text_encoder=backbones_data["text_encoder"],
        image_processor=backbones_data["image_processor"],
        tokenizer=backbones_data["tokenizer"],
        image_encoder_config=backbones_data["image_backbone_config"],
        text_encoder_config=backbones_data["text_backbone_config"],
    )
    # TODO: what are these exactly?
    # Train, validate, and test.
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


if __name__ == "__main__":
    typer.run(experiment)
