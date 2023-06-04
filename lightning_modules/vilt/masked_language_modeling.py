"""The masked language modeling lightning module."""
from functools import partial
from typing import Any, cast

import datasets
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import ViltForMaskedLM

from collators import ClassificationCollator
from collators.daquar import DaquarDataCollatorForMaskedLanguageModeling
from collators.vqa_v2 import VqaV2DataCollatorForMaskedLanguageModeling
from config.env import NUM_WORKERS
from models.backbones.configs import ViLTMLMConfig
from transforms.noop import default_noop_transforms_factory
from utils.batch import (
    batch_to_device,
    convert_batch_to_mapping_of_features,
    default_collator,
)
from utils.datasets import DatasetsLoadingFunctionType
from utils.types import BatchType, StageType


class ViLTMaskedLanguageModelingModule(pl.LightningModule):
    """The ViLT masked language modeling module."""

    def __init__(
        self,
        collator_cls: type[DaquarDataCollatorForMaskedLanguageModeling]
        | type[VqaV2DataCollatorForMaskedLanguageModeling],
        dataset_loading_function: DatasetsLoadingFunctionType,
        batch_size: int = 64,
    ):
        """
        Initialize the module.

        The ViLT classification module.
        It uses the ViLT backbone to extract the multimodal embedding and then
        uses a classifier to classify the multimodal embedding.

        :param batch_size: The batch size.
        """
        super().__init__()
        self.vilt_backbone_config = ViLTMLMConfig
        self.vilt = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")
        self.tokenizer = self.vilt_backbone_config.get_tokenizer()
        self.image_processor = self.vilt_backbone_config.get_image_processor()
        self.batch_size = batch_size

        self.batch_image_transforms = default_noop_transforms_factory()
        self.single_image_transforms = default_noop_transforms_factory()
        self.batch_text_transforms = default_noop_transforms_factory()
        self.single_text_transforms = default_noop_transforms_factory()

        self._collator_cls = collator_cls
        self._dataset_loading_function = dataset_loading_function

        self._data: dict[datasets.Split, datasets.Dataset] = {}
        self._collator_fn: ClassificationCollator | None = None
        self._default_collator_fn = partial(default_collator, self._collator_cls.IMAGE_BATCH_PROPERTY)

    def configure_optimizers(self) -> Any:
        """Configure the optimizers."""
        return torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)

    def _shared_eval_step(self, batch: BatchType, prefix: str) -> dict[str, Any]:
        """
        Perform a shared evaluation step.

        :param batch: The batch.
        :param prefix: The prefix.
        :return: The loss and logits.
        """
        outputs = self.vilt(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            return_dict=True,
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log_dict(
            {f"{prefix}_loss": loss},
            batch_size=self.batch_size,
            sync_dist=True,
        )
        return {"loss": loss, "logits": outputs.logits}

    def training_step(self, batch: BatchType, batch_idx: int):
        """
        Perform a training step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        eval_step = self._shared_eval_step(batch, prefix="train")
        return eval_step["loss"]

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
            collate_fn=self._default_collator_fn,
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
            collate_fn=self._default_collator_fn,
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
            collate_fn=self._default_collator_fn,
        )

    def setup(self, stage: StageType) -> None:
        """
        Set up the datasets.

        :param stage:
        :return: None.
        """
        self._data = self._dataset_loading_function()
        self._collator_fn = self._collator_cls(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            image_encoder_config=self.vilt_backbone_config,
            text_encoder_config=self.vilt_backbone_config,
            single_image_transforms=self.single_image_transforms[stage],
            single_text_transforms=self.single_text_transforms[stage],
            batch_image_transforms=self.batch_image_transforms[stage],
            batch_text_transforms=self.batch_text_transforms[stage],
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

    def make_masked_answer_prediction(
        self,
        batch: BatchType,
    ):
        """
        Make the masked answer prediction.

        :param batch: The batch.
        :return: The masked answer prediction.
        """
        batch = convert_batch_to_mapping_of_features(batch)
        questions = batch[self._collator_cls.ORIGINAL_QUESTION_BATCH_PROPERTY]
        processor = ViLTMLMConfig.get_processor()
        tokenizer = ViLTMLMConfig.get_tokenizer()
        masked_questions = [
            f"answer the following question: {question} answer: {tokenizer.mask_token}" for question in questions
        ]
        inputs = ViLTMLMConfig.get_processed_text_and_image(
            processor=processor,
            text=masked_questions,
            image=batch[self._collator_cls.IMAGE_BATCH_PROPERTY],
        )
        inputs = batch_to_device(inputs, self.device)
        mask_token_indices = inputs["input_ids"] == tokenizer.mask_token_id
        outputs = self.vilt(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            pixel_mask=inputs["pixel_mask"],
            return_dict=True,
        )
        logits = outputs.logits
        masked_answer_predictions = logits.argmax(dim=-1)
        masked_answer_predictions = masked_answer_predictions[mask_token_indices]
        return [tokenizer.decode(prediction) for prediction in masked_answer_predictions]
