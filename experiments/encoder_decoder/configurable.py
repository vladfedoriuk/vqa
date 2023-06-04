"""
ViT - GPT2 Causal Language Modeling Experiment.

This experiment is a fine-tuning experiment for ViT - GPT2 to solve VQA
as Causal Language Modeling task.
The model is an Encoder-Decoder model, where the encoder is a ViT and the decoder is a GPT2.

The experiment can be parameterized using the following command line arguments:

- ``epochs``: The number of epochs to train for.
- ``batch_size``: The batch size to use.
- ``dataset``: The dataset to use.
- ``encoder_decoder_backbone``: The encoder-decoder backbone to use.
"""
import lightning.pytorch as pl
import typer
import wandb

from callbacks.checkpoints import get_model_checkpoint
from callbacks.sample import CausalLanguageModelingPredictionSamplesCallback
from collators.daquar import DaquarDataCollatorForLanguageModeling
from collators.vqa_v2 import VqaV2DataCollatorForLanguageModeling
from lightning_modules.encoder_decoder.configurable import (
    ConfigurableEncoderDecoderModule,
)
from loggers.wandb import get_lightning_logger
from models.backbones import AvailableBackbones
from models.backbones import registry as backbones_registry
from utils.config import load_env_config
from utils.datasets import AvailableDatasets
from utils.datasets import registry as datasets_registry
from utils.logger import compose_encoder_decoder_experiment_run_name
from utils.registry import initialize_registries
from utils.torch import ensure_reproducibility, get_lightning_trainer_strategy


@initialize_registries()
@load_env_config()
def experiment(
    epochs: int = 10,
    batch_size: int = 64,
    dataset: AvailableDatasets = AvailableDatasets.DAQUAR,
    encoder_decoder_backbone: AvailableBackbones = AvailableBackbones.VIT_GPT2,
):
    """
    Run the configurable causal language modeling experiment.

    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :param dataset: The dataset to use.
    :param encoder_decoder_backbone: The encoder-decoder backbone to use.
    :return: None.
    """
    # Ensure reproducibility.
    ensure_reproducibility()  # TODO Context decorator

    # Login to wandb.
    wandb.login()  # TODO Context decorator

    # Initialize the logger.
    run_name = compose_encoder_decoder_experiment_run_name(
        encoder_decoder_backbone=encoder_decoder_backbone,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
    )

    logger = get_lightning_logger(run_name)

    # Get the datasets loading function.
    dataset_loading_fn = datasets_registry.get(dataset)

    # Get the config for the encoder-decoder backbone.
    config = backbones_registry.get(encoder_decoder_backbone)

    # Get the collator class.
    dataset_to_collator_cls = {
        AvailableDatasets.DAQUAR: DaquarDataCollatorForLanguageModeling,
        AvailableDatasets.VQA_V2: VqaV2DataCollatorForLanguageModeling,
        AvailableDatasets.VQA_V2_SAMPLE: VqaV2DataCollatorForLanguageModeling,
    }
    collator_cls = dataset_to_collator_cls[dataset]

    # Create a trainer.
    # TODO: Try deterministic=True.
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        devices=1,
        num_nodes=1,
        strategy=get_lightning_trainer_strategy(),
        max_epochs=epochs,
        callbacks=[
            get_model_checkpoint(file_name=run_name),
            CausalLanguageModelingPredictionSamplesCallback(dataset=dataset),
        ],
        accumulate_grad_batches=8,
    )
    model = ConfigurableEncoderDecoderModule(
        collator_cls=collator_cls,
        dataset_loading_function=dataset_loading_fn,
        batch_size=batch_size,
        config=config,
    )
    # Train, validate, and test.
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    typer.run(experiment)
