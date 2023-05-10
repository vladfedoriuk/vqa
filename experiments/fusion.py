"""
Late fusion experiments.

This experiment is a simple late fusion experiment.

The experiment can be parameterized using the following command line arguments:

- ``image_encoder_backbone``: The backbone to use for the image encoder.
- ``text_encoder_backbone``: The backbone to use for the text encoder.
- ``multimodal_backbone``: The multimodal backbone to use.
- ``fusion``: The fusion model to use.
- ``dataset``: The dataset to use.
- ``freeze_image_encoder_backbone``: Whether to freeze the image encoder backbone.
- ``freeze_text_encoder_backbone``: Whether to freeze the text encoder backbone.
- ``freeze_multimodal_backbone``: Whether to freeze the multimodal backbone.
- ``epochs``: The number of epochs to train for.
- ``batch_size``: The batch size to use.
"""
from typing import Optional, cast

import lightning.pytorch as pl
import typer
import wandb
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner import Tuner

from callbacks.checkpoints import get_model_checkpoint
from callbacks.sample import PredictionSamplesCallback
from collators import ClassificationCollator
from collators import registry as collators_registry
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
from utils.logger import compose_run_name
from utils.registry import initialize_registries
from utils.torch import ensure_reproducibility, freeze_model_parameters


@initialize_registries()
@load_env_config()
def experiment(
    image_encoder_backbone: Optional[AvailableBackbones] = None,
    text_encoder_backbone: Optional[AvailableBackbones] = None,
    multimodal_backbone: Optional[AvailableBackbones] = None,
    fusion: AvailableFusionModels = AvailableFusionModels.CAT,
    dataset: AvailableDatasets = AvailableDatasets.VQA_V2_SAMPLE,
    freeze_image_encoder_backbone: bool = True,
    freeze_text_encoder_backbone: bool = True,
    freeze_multimodal_backbone: bool = True,
    epochs: int = 10,
    batch_size: int = 64,
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
    :param freeze_image_encoder_backbone: Whether to freeze the image encoder backbone.
    :param freeze_text_encoder_backbone: Whether to freeze the text encoder backbone.
    :param freeze_multimodal_backbone: Whether to freeze the multimodal backbone.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :return: None.
    """
    # Ensure reproducibility.
    ensure_reproducibility()  # TODO Context decorator

    # Login to wandb.
    wandb.login()  # TODO Context decorator

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
    if image_encoder_backbone is not None and freeze_image_encoder_backbone:
        freeze_model_parameters(backbones_data["image_encoder"])
    if text_encoder_backbone is not None and freeze_text_encoder_backbone:
        freeze_model_parameters(backbones_data["text_encoder"])
    if multimodal_backbone is not None and freeze_multimodal_backbone:
        freeze_model_parameters(backbones_data["image_encoder"])
        freeze_model_parameters(backbones_data["text_encoder"])

    # Initialize the logger.
    run_name = compose_run_name(
        image_encoder_backbone=image_encoder_backbone,
        text_encoder_backbone=text_encoder_backbone,
        multimodal_backbone=multimodal_backbone,
        fusion=fusion,
        dataset=dataset,
        freeze_image_encoder_backbone=freeze_image_encoder_backbone,
        freeze_text_encoder_backbone=freeze_text_encoder_backbone,
        freeze_multimodal_backbone=freeze_multimodal_backbone,
        batch_size=batch_size,
        epochs=epochs,
    )
    logger = get_lightning_logger(run_name)

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
        strategy=DDPStrategy(
            find_unused_parameters=True
        ),  # TODO: Read it from env vars - the env var will be set in scripts.
        # - Add a new config setting for the strategy.
        max_epochs=epochs,
        callbacks=[
            PredictionSamplesCallback(
                answer_space=answer_space,
                dataset=dataset,
            ),
            get_model_checkpoint(
                file_name=f"{run_name}.ckpt",
            ),
        ],
        accumulate_grad_batches=32,
    )
    # TODO: Fix image_encoder_config and text_encoder_config names.
    # Create a model.
    final_representation_size = min(
        backbones_data["image_representation_size"],
        backbones_data["text_representation_size"],
    )
    model = MultiModalClassificationModule(
        fusion=fusion_model_factory(
            image_representation_size=backbones_data["image_representation_size"],
            text_representation_size=backbones_data["text_representation_size"],
            final_representation_size=final_representation_size,
        ),
        classifier=default_classifier_factory(input_dim=final_representation_size, classes_num=len(answer_space)),
        image_encoder=backbones_data["image_encoder"],
        text_encoder=backbones_data["text_encoder"],
        image_processor=backbones_data["image_processor"],
        tokenizer=backbones_data["tokenizer"],
        image_encoder_config=backbones_data["image_backbone_config"],
        text_encoder_config=backbones_data["text_backbone_config"],
        answer_space=answer_space,
        collator_cls=cast(type[ClassificationCollator], collator_cls),
        datasets_loading_function=datasets_loading_fn,
        batch_size=batch_size,
    )
    tuner = Tuner(trainer)
    tuner.lr_find(model)
    # TODO: what are these exactly?
    # Train, validate, and test.
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    typer.run(experiment)
