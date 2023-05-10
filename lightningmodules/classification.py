"""
The module contains the base class for classification modules.

To know more about the classification modules,
check out the documentation:

https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
"""
from typing import Literal

import datasets
import lightning.pytorch as pl
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from collators import ClassificationCollator
from models.backbones import BackboneConfig
from transforms.image import BatchVQAImageAugmentationModule
from transforms.noop import noop
from transforms.text import QuestionAugmentationModule
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


class MultiModalClassificationModule(pl.LightningModule):
    """Base class for classification modules."""

    def __init__(
        self,
        fusion: nn.Module,
        classifier: nn.Module,
        answer_space: AnswerSpace,
        collator_cls: type[ClassificationCollator],
        datasets_loading_function: DatasetsLoadingFunctionType,
        image_encoder: nn.Module,
        image_processor: BaseImageProcessor,
        text_encoder: nn.Module,
        tokenizer: PreTrainedTokenizer,
        image_encoder_config: type[BackboneConfig],
        text_encoder_config: type[BackboneConfig],
        batch_size: int = 64,
        lr: float = 1e-4,
    ):
        """
        Initialize the module.

        :param fusion: The fusion.
        :param classifier: The classifier.
        :param answer_space: The answer space.
        :param collator_cls: The collator class.
        :param datasets_loading_function: The datasets loading function.
        :param image_encoder: The image encoder.
        :param image_processor: The image processor.
        :param tokenizer: The tokenizer.
        :param text_encoder: The text encoder.
        :param image_encoder_config: The image encoder config.
        :param text_encoder_config: The text encoder config.
        :param batch_size: The batch size.
        :param lr: The learning rate.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "text_encoder",
                "fusion",
                "classifier",
                "image_processor",
                "tokenizer",
                "image_encoder_config",
                "text_encoder_config",
                "collator_cls",
                "datasets_loading_function",
            ]
        )
        self.fusion = fusion
        self.classifier = classifier

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.answer_space = answer_space
        self.classes_num = len(answer_space)

        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.image_encoder_config = image_encoder_config
        self.text_encoder_config = text_encoder_config

        self.loss = torch.nn.CrossEntropyLoss()

        self.train_batch_image_augmentation = BatchVQAImageAugmentationModule()
        self.train_question_augmentation = QuestionAugmentationModule()
        self.single_image_transforms: dict[StageType, SingleImageTransformsType] = {
            "fit": noop,
            "validate": noop,
            "test": noop,
            "predict": noop,
        }

        self.single_text_transforms: dict[StageType, SingeTextTransformsType] = {
            "fit": noop,
            "validate": noop,
            "test": noop,
            "predict": noop,
        }

        # TODO: Add the Noop module for consistency.
        # TODO: Use ModelDict for consistency.
        self.batch_image_transforms: dict[StageType, BatchImageTransformsType] = {
            "fit": self.train_batch_image_augmentation,
            "validate": noop,
            "test": noop,
            "predict": noop,
        }

        self.batch_text_transforms: dict[StageType, BatchTextTransformsType] = {
            "fit": self.train_question_augmentation,
            "validate": noop,
            "test": noop,
            "predict": noop,
        }

        self.batch_size = batch_size
        self.collator_cls = collator_cls
        self.datasets_loading_function = datasets_loading_function
        self._collator_fn = None
        self._data = None
        self.learning_rate = lr

    def _get_embeddings(self, batch: BatchType):
        """
        Get the embeddings.

        :param batch: The batch.
        :return: The embeddings.
        """
        return {
            "image_emb": (
                self.image_encoder_config.get_image_representation_from_preprocessed(
                    model=self.image_encoder,
                    processor_output=batch,
                )
            ),
            "text_emb": (
                self.text_encoder_config.get_text_representation_from_tokenized(
                    model=self.text_encoder,
                    tokenizer_output=batch,
                )
            ),
        }

    def _shared_eval_step(self, batch: BatchType, prefix: Literal["train", "val", "test"]):
        """
        Perform a shared evaluation step.

        :param batch: The batch.
        :param prefix: The prefix.
        :return: The loss and logits.
        """
        embeddings = self._get_embeddings(batch)
        fused_repr = self.fusion(embeddings["image_emb"], embeddings["text_emb"])
        logits = self.classifier(fused_repr)
        loss = self.loss(logits, batch["answer_label"])
        self._log_metrics(loss, logits, batch, prefix=prefix)
        return {
            "loss": loss,
            "logits": logits,
        }

    def _log_metrics(
        self, loss: torch.Tensor, logits: torch.Tensor, batch: BatchType, prefix: Literal["train", "val", "test"]
    ):
        """
        Log the metrics.

        :param loss: The loss.
        :param logits: The logits.
        :param batch: The batch.
        :param prefix: The prefix.
        :return: None.
        """
        self.log_dict(
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

    def training_step(self, batch: BatchType, batch_idx: int):
        """
        Perform a training step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        eval_step = self._shared_eval_step(batch, prefix="train")
        return eval_step["loss"]

    def configure_optimizers(self):
        """
        Configure the optimizers.

        The optimizer might leverage the learning rate finder:
        https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html

        :return: The optimizer.
        """
        lr = self.hparams.get("lr", self.hparams.get("learning_rate", self.learning_rate))
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

    def validation_step(self, batch: BatchType, batch_idx: int):
        """
        Perform a validation step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss and logits.
        """
        return self._shared_eval_step(batch, prefix="val")

    def test_step(self, batch: BatchType, batch_idx: int):
        """
        Perform a test step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss and logits.
        """
        return self._shared_eval_step(batch, prefix="test")

    def prepare_data(self) -> None:
        """
        Prepare the data.

        :return: None.
        """
        self.datasets_loading_function()

    def train_dataloader(self):
        """
        Get the training dataloader.

        :return: The training dataloader.
        """
        return DataLoader(
            self._data[datasets.Split.TRAIN],
            batch_size=self.batch_size,
            num_workers=6,
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
            num_workers=6,
            drop_last=True,
        )

    def test_dataloader(self):
        """
        Get the test dataloader.

        :return: The test dataloader.
        """
        data = self.datasets_loading_function()
        return DataLoader(
            data[datasets.Split.TEST],
            batch_size=self.batch_size,
            num_workers=6,
            drop_last=True,
        )

    def setup(self, stage: StageType) -> None:
        """
        Set up the datasets.

        :param stage:
        :return: None.
        """
        self._data = self.datasets_loading_function()
        self._collator_fn = self.collator_cls(
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
        return self._collator_fn(batch_to_device(batch, self.device))
