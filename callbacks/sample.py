"""
The predictions samples Lightning modules callbacks.

To know more about the Lightning modules callbacks,
check out the documentation:

https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
"""
from collections.abc import Callable
from typing import Any, cast

import torch
from lightning import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from lightning_modules.encoder_decoder.configurable import (
    ConfigurableEncoderDecoderModule,
)
from lightning_modules.fusion.classification import MultiModalFusionClassificationModule
from lightning_modules.mixins import VQAClassificationMixin
from lightning_modules.vilt.masked_language_modeling import (
    ViLTMaskedLanguageModelingModule,
)
from loggers.wandb import (
    log_causal_language_modeling_predictions_as_table,
    log_daquar_predictions_as_table,
    log_vilt_mlm_predictions_as_table,
    log_vqa_v2_predictions_as_table,
)
from utils.batch import convert_batch_to_sequence_of_mappings
from utils.datasets import AvailableDatasets
from utils.datasets.answer_space import AnswerSpace


class PredictionSamplesCallback(Callback):
    """The callback to log the prediction samples fom the validation set."""

    def __init__(self, num_samples: int = 20):
        """
        Initialize the callback.

        :param num_samples: The number of samples to log.
        """
        super().__init__()
        self.num_samples = num_samples

    def _select_num_samples(self, batch):
        """
        Select the number of samples to log.

        :param batch: The batch.
        :return: The number of samples to log.
        """
        return torch.randperm(len(batch))[: self.num_samples]

    def _prepare_batch_subset(self, batch):
        """
        Prepare the batch subset.

        :param batch: The batch.
        :return: The batch subset.
        """
        batch = convert_batch_to_sequence_of_mappings(batch)
        return [batch[idx] for idx in self._select_num_samples(batch)]


class ClassificationPredictionSamplesCallback(PredictionSamplesCallback):
    """
    The callback to log the classification prediction samples fom the validation set.

    The callback is meant to be used with
    :class:`lightning_modules.fusion.classification.MultiModalFusionClassificationModule`.

    The callback logs the prediction samples to Weights & Biases.
    """

    _DATASET_TO_LOGGER_FN: dict[
        AvailableDatasets, Callable[[WandbLogger, AnswerSpace, list[dict[str, Any]], dict[str, Any]], None]
    ] = {
        AvailableDatasets.DAQUAR: log_daquar_predictions_as_table,
        AvailableDatasets.VQA_V2: log_vqa_v2_predictions_as_table,
        AvailableDatasets.VQA_V2_SAMPLE: log_vqa_v2_predictions_as_table,
    }  # TODO: Create a registry for this. Use metadata, unify the logging
    #    reuse some code for logging and maybe get rid of this?

    def __init__(self, answer_space: AnswerSpace, dataset: AvailableDatasets, num_samples: int = 20):
        """
        Initialize the callback.

        :param answer_space: The answer space to use.
        :param dataset: The dataset to use.
        :param num_samples: The number of samples to log.
        """
        super().__init__(num_samples=num_samples)
        self.answer_space = answer_space
        self.dataset = dataset

    def on_validation_batch_end(
        self, trainer, pl_module: MultiModalFusionClassificationModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Call when the validation batch ends.

        :param trainer: The trainer.
        :param pl_module: The Lightning module.
        :param outputs: The outputs.
        :param batch: The batch.
        :param batch_idx: The batch index.
        :param dataloader_idx: The dataloader index.
        """
        assert isinstance(pl_module, VQAClassificationMixin), (
            f"The {self.__class__!r} callback is meant to be used with " f"{VQAClassificationMixin!r}."
        )
        if batch_idx != 0:
            return
        batch = self._prepare_batch_subset(batch)
        logger = cast(WandbLogger, pl_module.logger)
        self._DATASET_TO_LOGGER_FN[self.dataset](logger, self.answer_space, batch, outputs)


class CausalLanguageModelingPredictionSamplesCallback(PredictionSamplesCallback):
    """
    The callback to log the language modeling prediction samples fom the validation set.

    The callback is meant to be used with
    :class:`lightning_modules.vilt.masked_language_modeling.ViLTMaskedLanguageModelingModule`.

    The callback logs the prediction samples to Weights & Biases.
    """

    _DATASET_TO_METADATA: dict[AvailableDatasets, dict[str, Any]] = {
        AvailableDatasets.DAQUAR: {
            "answer_batch_property": "answer",
            "image_batch_property": "image",
            "original_question_batch_property": "original_question",
        },
        AvailableDatasets.VQA_V2: (
            vqa_v2_metadata := {
                "answer_batch_property": "multiple_choice_answer",
                "image_batch_property": "image",
                "original_question_batch_property": "original_question",
            }
        ),
        AvailableDatasets.VQA_V2_SAMPLE: vqa_v2_metadata,
    }  # TODO: Use a registry for this.

    def __init__(self, dataset: AvailableDatasets, num_samples: int = 20):
        """
        Initialize the callback.

        :param dataset: The dataset to use.
        :param num_samples: The number of samples to log.
        """
        super().__init__(num_samples=num_samples)
        self.dataset = dataset

    def on_validation_batch_end(
        self, trainer, pl_module: ConfigurableEncoderDecoderModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Call when the validation batch ends.

        :param trainer: The trainer.
        :param pl_module: The Lightning module.
        :param outputs: The outputs.
        :param batch: The batch.
        :param batch_idx: The batch index.
        :param dataloader_idx: The dataloader index.
        """
        assert isinstance(pl_module, ConfigurableEncoderDecoderModule), (
            f"The {self.__class__!r} callback is meant to be used with " f"{ConfigurableEncoderDecoderModule!r}."
        )
        if batch_idx != 0:
            return
        batch = self._prepare_batch_subset(batch)
        logger = cast(WandbLogger, pl_module.logger)  # TODO: Move all up to this to the superclass.
        log_causal_language_modeling_predictions_as_table(
            pl_module=pl_module,
            logger=logger,
            batch=batch,
            **self._DATASET_TO_METADATA[self.dataset],
        )


class MaskedLanguageModelingPredictionSamplesCallback(CausalLanguageModelingPredictionSamplesCallback):
    """
    The callback to log the masked language modeling prediction samples fom the validation set.

    The callback is meant to be used with
    :class:`lightning_modules.vilt.masked_language_modeling.ViLTMaskedLanguageModelingModule`.

    The callback logs the prediction samples to Weights & Biases.
    """  # TODO: Make them dataset and model agnostic.

    def on_validation_batch_end(
        self, trainer, pl_module: ViLTMaskedLanguageModelingModule, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Call when the validation batch ends.

        :param trainer: The trainer.
        :param pl_module: The Lightning module.
        :param outputs: The outputs.
        :param batch: The batch.
        :param batch_idx: The batch index.
        :param dataloader_idx: The dataloader index.
        """
        assert isinstance(pl_module, ViLTMaskedLanguageModelingModule), (
            f"The {self.__class__!r} callback is meant to be used with " f"{ViLTMaskedLanguageModelingModule!r}."
        )
        if batch_idx != 0:
            return
        batch = self._prepare_batch_subset(batch)
        logger = cast(WandbLogger, pl_module.logger)
        log_vilt_mlm_predictions_as_table(
            pl_module=pl_module,
            logger=logger,
            batch=batch,
            **self._DATASET_TO_METADATA[self.dataset],
        )
