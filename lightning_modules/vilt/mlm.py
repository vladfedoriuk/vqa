"""The masked language modeling lightning module."""
from typing import Any, cast

import datasets
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import ViltForMaskedLM

from collators import ClassificationCollator
from collators.daquar import DaquarDataCollatorForLanguageModeling
from config.env import NUM_WORKERS
from models.backbones.configs import ViLTMLMConfig
from transforms.noop import noop
from utils.batch import batch_to_device, convert_batch_to_dict_of_features
from utils.datasets.daquar import load_daquar_datasets
from utils.types import BatchType, StageType


class ViLTMaskedLanguageModelingModule(pl.LightningModule):
    """The ViLT masked language modeling module."""

    def __init__(
        self,
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

        self._data: dict[datasets.Split, datasets.Dataset] = {}
        self._collator_fn: ClassificationCollator | None = None

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
        self.log(f"{prefix}_loss", loss)
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
        self._data = load_daquar_datasets()
        self._collator_fn = DaquarDataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            image_encoder_config=self.vilt_backbone_config,
            text_encoder_config=self.vilt_backbone_config,
            single_image_transforms=noop,
            single_text_transforms=noop,
            batch_image_transforms=noop,
            batch_text_transforms=noop,
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
        batch = convert_batch_to_dict_of_features(batch)
        questions = batch[DaquarDataCollatorForLanguageModeling.ORIGINAL_QUESTION_BATCH_PROPERTY]
        processor = ViLTMLMConfig.get_processor()
        tokenizer = ViLTMLMConfig.get_tokenizer()
        masked_questions = [f"question: {question} answer: {tokenizer.mask_token}" for question in questions]
        inputs = ViLTMLMConfig.get_processed_text_and_image(
            processor=processor,
            text=masked_questions,
            image=batch[DaquarDataCollatorForLanguageModeling.IMAGE_BATCH_PROPERTY],
        )
        inputs = batch_to_device(inputs, self.device)
        mask_token_indices = inputs["input_ids"] == tokenizer.mask_token_id
        outputs = self.vilt(**inputs)
        logits = outputs.logits
        masked_answer_predictions = logits.argmax(dim=-1)
        masked_answer_predictions = masked_answer_predictions[mask_token_indices]
        return [tokenizer.decode(prediction) for prediction in masked_answer_predictions]
