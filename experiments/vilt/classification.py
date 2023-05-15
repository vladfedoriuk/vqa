"""
ViLT classification experiments.

This experiment is a fine-tuning experiment for ViLT to solve VQA
as classification task.

The experiment can be parameterized using the following command line arguments:

- ``dataset``: The dataset to use.
- ``epochs``: The number of epochs to train for.
- ``batch_size``: The batch size to use.
"""
from typing import cast

import lightning.pytorch as pl
import typer
import wandb
from lightning.pytorch.strategies import DDPStrategy

from callbacks.checkpoints import get_model_checkpoint
from callbacks.sample import PredictionSamplesCallback
from collators import ClassificationCollator
from collators import registry as collators_registry
from lightning_modules.vilt.classification import ViLTClassificationModule
from loggers.wandb import get_lightning_logger
from models.backbones import AvailableBackbones
from models.backbones import registry as backbones_registry
from utils.config import load_env_config
from utils.datasets import AvailableDatasets
from utils.datasets import registry as datasets_registry
from utils.datasets.answer_space import registry as answer_space_registry
from utils.logger import compose_vilt_classification_experiment_run_name
from utils.registry import initialize_registries
from utils.torch import ensure_reproducibility


@initialize_registries()
@load_env_config()
def experiment(
    vilt_backbone: AvailableBackbones = AvailableBackbones.ViLT_MLM,
    dataset: AvailableDatasets = AvailableDatasets.VQA_V2_SAMPLE,
    epochs: int = 10,
    batch_size: int = 64,
):
    """
    Run the simple concatenation fusion experiment.

    :param vilt_backbone: The ViLT backbone to use.
    :param dataset: The name of the dataset to use.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :return: None.
    """
    # Ensure reproducibility.
    ensure_reproducibility()  # TODO Context decorator

    # Login to wandb.
    wandb.login()  # TODO Context decorator

    # Initialize the answer space.
    answer_space = answer_space_registry.get(dataset)()

    # Get the collator class.
    collator_cls = collators_registry.get(dataset)

    # Get the datasets loading function.
    datasets_loading_fn = datasets_registry.get(dataset)

    # Initialize the logger.
    run_name = compose_vilt_classification_experiment_run_name(
        vilt_backbone=vilt_backbone,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
    )

    logger = get_lightning_logger(run_name)

    if not issubclass(collator_cls, ClassificationCollator):
        print(f"Expected a classification collator for {dataset}, got {collator_cls}.")
        raise typer.Abort()

    if vilt_backbone not in (AvailableBackbones.ViLT_MLM, AvailableBackbones.ViLT_VQA):
        print(f"Expected a ViLT backbone, got {vilt_backbone}.")
        raise typer.Abort()

    vilt_backbone_config = backbones_registry.get(vilt_backbone)

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
        accumulate_grad_batches=4,
    )
    model = ViLTClassificationModule(
        vilt_backbone_config=vilt_backbone_config,
        answer_space=answer_space,
        collator_cls=cast(type[ClassificationCollator], collator_cls),
        datasets_loading_function=datasets_loading_fn,
        batch_size=batch_size,
    )
    # TODO: what are these exactly?
    # Train, validate, and test.
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    typer.run(experiment)
