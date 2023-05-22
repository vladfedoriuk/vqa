"""The language generation Encoder-Decoder module using ViT and GPT-2 pretrained backbone."""
from typing import Any, cast

import datasets
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers.generation_utils import GenerationMixin

from collators import ClassificationCollator
from collators.daquar import DaquarDataCollatorForLanguageModeling
from config.env import NUM_WORKERS
from models.backbones.configs import ViTGPT2Config
from transforms.noop import noop
from utils.batch import batch_to_device, convert_batch_to_mapping_of_features
from utils.datasets.daquar import load_daquar_datasets
from utils.types import BatchType, StageType


class ViTGPT2EncoderDecoderModule(pl.LightningModule):
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
        self.backbone_config = ViTGPT2Config
        self.backbone_model = self.backbone_config.get_model()
        self.tokenizer = self.backbone_config.get_tokenizer()
        self.image_processor = self.backbone_config.get_image_processor()
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
        outputs = self.backbone_model(
            pixel_values=batch["pixel_values"],
            return_dict=True,
            labels=batch["labels"],
            # Automatically figures out the decoder input_ids, attention_mask, etc.
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
            image_encoder_config=self.backbone_config,
            text_encoder_config=self.backbone_config,
            single_image_transforms=noop,
            single_text_transforms=noop,
            batch_image_transforms=noop,
            batch_text_transforms=noop,  # TODO: Use augmentations
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

    def make_generated_answer_prediction(
        self,
        batch: BatchType,
    ):
        """
        Make the answer prediction by generating the answer.

        :param batch: The batch.
        :return: The generated answer.
        """
        batch = convert_batch_to_mapping_of_features(batch)
        questions = batch[DaquarDataCollatorForLanguageModeling.ORIGINAL_QUESTION_BATCH_PROPERTY]
        tokenizer = self.backbone_config.get_tokenizer()
        prompts = [f"answer the following question: {question} answer: " for question in questions]
        decoder_inputs = batch_to_device(
            self.backbone_config.get_tokenized_text(
                tokenizer=tokenizer,
                text=prompts,
            ),
            self.device,
        )
        encoder_inputs = batch_to_device(
            self.backbone_config.get_processed_image(
                processor=self.image_processor, image=batch[DaquarDataCollatorForLanguageModeling.IMAGE_BATCH_PROPERTY]
            ),
            self.device,
        )
        backbone_model = cast(GenerationMixin, self.backbone_model)
        outputs = backbone_model.generate(
            **encoder_inputs,
            decoder_input_ids=decoder_inputs["input_ids"],
            return_dict_in_generate=True,
            max_length=100,
            do_sample=True,
            temperature=0.1,
        )
        return tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
