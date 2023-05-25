"""The mixins that are used in the LightningModules."""
import abc
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict, cast

import datasets
import lightning.pytorch as pl
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, confusion_matrix
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from collators import ClassificationCollator
from config.env import NUM_WORKERS
from loggers.wandb import log_confusion_matrix
from models.backbones import BackboneConfig
from utils.batch import batch_to_device
from utils.datasets import DatasetsLoadingFunctionType
from utils.datasets.answer_space import AnswerSpace
from utils.types import (
    BatchImageTransformsType,
    BatchTextTransformsType,
    BatchType,
    SingeTextTransformsType,
    SingleImageTransformsType,
    StageType,
)


class Metrics(TypedDict):
    """The metrics returned by the shared evaluation step."""

    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


class VQAClassificationMixin:
    """
    A mixin for VQA classification.

    The mixin is used to define the common functionality for VQA classification.
    It should be used in conjunction with a LightningModule.
    """

    # A function that loads the datasets.
    datasets_loading_function: DatasetsLoadingFunctionType
    # The batch size.
    batch_size: int
    # The tokenizer.
    tokenizer: PreTrainedTokenizer
    # The image processor.
    image_processor: BaseImageProcessor
    # The image encoder configuration.
    image_encoder_config: type[BackboneConfig]
    # The text encoder configuration.
    text_encoder_config: type[BackboneConfig]
    # The answer space.
    answer_space: AnswerSpace
    # The number of classes.
    classes_num: int

    # The single image transforms.
    single_image_transforms: Mapping[StageType, SingleImageTransformsType]
    # The batch image transforms.
    batch_image_transforms: Mapping[StageType, BatchImageTransformsType]
    # The batch text transforms.
    batch_text_transforms: Mapping[StageType, BatchTextTransformsType]
    # The single text transforms.
    single_text_transforms: Mapping[StageType, SingeTextTransformsType]
    # The collator class.

    # The collator class.
    _collator_cls: type[ClassificationCollator]
    # The data to be used for training.
    _data: datasets.DatasetDict[datasets.Split, datasets.Dataset] | None
    # The collator function.
    _collator_fn: Callable[[list[Any]], Any | None] | None

    def prepare_data(self) -> None:
        """
        Prepare the data.

        :return: None.
        """
        self.datasets_loading_function()

    @abc.abstractmethod
    def _shared_eval_step(self, batch: BatchType) -> Metrics:
        """
        Perform a shared evaluation step.

        :param batch: The batch.
        :return: The loss and logits.
        """
        raise NotImplementedError

    def _log_metrics(
        self, loss: torch.Tensor, logits: torch.Tensor, batch: BatchType, prefix: Literal["train", "val", "test"]
    ) -> None:
        """
        Log the metrics.

        :param loss: The loss.
        :param logits: The logits.
        :param batch: The batch.
        :param prefix: The prefix.
        :return: None.
        """
        instance = cast(pl.LightningModule, self)
        instance.log_dict(
            {
                f"{prefix}_loss": loss,
                f"{prefix}_acc": accuracy(
                    logits,
                    batch["answer_label"],
                    task="multiclass",
                    num_classes=self.classes_num,
                ),
            },
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def validation_epoch_end(self, outputs: list[Metrics]):
        """
        Perform validation epoch end.

        :param outputs: The outputs.
        :return: None.
        """
        instance = cast(pl.LightningModule, self)
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        cm = confusion_matrix(
            logits,
            labels,
            num_classes=self.classes_num,
            normalize="true",
            task="multiclass",
        )
        log_confusion_matrix(cast(WandbLogger, instance.logger), cm, key="val_confusion_matrix")

    def training_step(self, batch: BatchType, batch_idx: int):
        """
        Perform a training step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        eval_step = self._shared_eval_step(batch)
        self._log_metrics(eval_step["loss"], eval_step["logits"], batch, prefix="train")
        return eval_step["loss"]

    def validation_step(self, batch: BatchType, batch_idx: int):
        """
        Perform a validation step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss and logits.
        """
        eval_step = self._shared_eval_step(batch)
        self._log_metrics(eval_step["loss"], eval_step["logits"], batch, prefix="val")
        return eval_step

    def test_step(self, batch: BatchType, batch_idx: int):
        """
        Perform a test step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss and logits.
        """
        eval_step = self._shared_eval_step(batch)
        self._log_metrics(eval_step["loss"], eval_step["logits"], batch, prefix="test")
        return eval_step

    def train_dataloader(self):
        """
        Get the training dataloader.

        :return: The training dataloader.
        """
        return DataLoader(
            self._data[datasets.Split.TRAIN],
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        Get the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            self._data[datasets.Split.VALIDATION],
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def test_dataloader(self):
        """
        Get the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            self._data[datasets.Split.TEST],
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            drop_last=True,
        )

    def setup(self, stage: StageType) -> None:
        """
        Set up the datasets.

        :param stage:
        :return: None.
        """
        self._data = self.datasets_loading_function()
        self._collator_fn = self._collator_cls(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            image_encoder_config=self.image_encoder_config,
            text_encoder_config=self.text_encoder_config,
            single_image_transforms=self.single_image_transforms[stage],
            single_text_transforms=self.single_text_transforms[stage],
            batch_image_transforms=self.batch_image_transforms[stage],
            batch_text_transforms=self.batch_text_transforms[stage],
            answer_space=self.answer_space,
        )

    def on_after_batch_transfer(self, batch: BatchType, dataloader_idx: int):
        """
        Perform a batch transfer.

        :param batch: The batch.
        :param dataloader_idx: The dataloader index.
        :return: The batch.
        """
        instance = cast(pl.LightningModule, self)
        return batch_to_device(self._collator_fn(batch_to_device(batch, instance.device)), instance.device)
