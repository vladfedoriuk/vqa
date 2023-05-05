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

from loggers.wandb import (
    log_daquar_predictions_as_table,
    log_vqa_v2_predictions_as_table,
)
from utils.datasets import AvailableDatasets, convert_batch_to_list_of_dicts
from utils.datasets.answer_space import AnswerSpace


class PredictionSamplesCallback(Callback):
    """
    The callback to log the classification prediction samples fom the validation set.

    The callback is meant to be used with
    :class:`lightningmodules.classification.MultiModalClassificationModule`.

    The callback logs the prediction samples to Weights & Biases.
    """

    _DATASET_TO_LOGGER_FN: dict[
        AvailableDatasets, Callable[[WandbLogger, AnswerSpace, list[dict[str, Any]], dict[str, Any]], None]
    ] = {
        AvailableDatasets.DAQUAR: log_daquar_predictions_as_table,
        AvailableDatasets.VQA_V2: log_vqa_v2_predictions_as_table,
        AvailableDatasets.VQA_V2_SAMPLE: log_vqa_v2_predictions_as_table,
    }

    def __init__(self, answer_space: AnswerSpace, dataset: AvailableDatasets, num_samples: int = 20):
        """
        Initialize the callback.

        :param answer_space: The answer space to use.
        :param dataset: The dataset to use.
        :param num_samples: The number of samples to log.
        """
        super().__init__()
        self.answer_space = answer_space
        self.num_samples = num_samples
        self.dataset = dataset

    def _select_num_samples(self, batch):
        """
        Select the number of samples to log.

        :param batch: The batch.
        :return: The number of samples to log.
        """
        if len(batch) < self.num_samples:
            return batch
        return torch.randperm(len(batch))[: self.num_samples]

    def _prepare_batch_subset(self, batch):
        """
        Prepare the batch subset.

        :param batch: The batch.
        :return: The batch subset.
        """
        batch = convert_batch_to_list_of_dicts(batch)
        return [batch[idx] for idx in self._select_num_samples(batch)]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Call when the validation batch ends.

        :param trainer: The trainer.
        :param pl_module: The Lightning module.
        :param outputs: The outputs.
        :param batch: The batch.
        :param batch_idx: The batch index.
        :param dataloader_idx: The dataloader index.
        """
        if batch_idx != 0:
            return
        batch = self._prepare_batch_subset(batch)
        logger = cast(WandbLogger, pl_module.logger)
        self._DATASET_TO_LOGGER_FN[self.dataset](logger, self.answer_space, batch, outputs)
