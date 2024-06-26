"""Transforms for text data."""
from collections.abc import Mapping
from typing import cast

import torch
from torch import nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationMixin,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from transforms.noop import Noop
from utils.batch import batch_to_device
from utils.torch import DeviceAwareModule
from utils.types import BatchTextTransformsType, BatchType, StageType


class QuestionAugmentationModule(DeviceAwareModule):
    """Perform question augmentation."""

    def __init__(
        self,
        to_model_name: str = "Helsinki-NLP/opus-mt-en-de",
        from_model_name: str = "Helsinki-NLP/opus-mt-de-en",
    ):
        """Initialize the module."""
        super().__init__()
        self.to_tokenizer = AutoTokenizer.from_pretrained(to_model_name)
        self.from_tokenizer = AutoTokenizer.from_pretrained(from_model_name)

        self.to_model = AutoModelForSeq2SeqLM.from_pretrained(to_model_name)
        self.from_model = AutoModelForSeq2SeqLM.from_pretrained(from_model_name)

    @staticmethod
    def _ensure_list_of_questions(questions):
        if isinstance(questions, str):
            questions = [questions]
        return questions

    @torch.no_grad()
    def forward(self, batch: BatchType) -> BatchType:
        """
        Augments the questions in the batch.

        :param batch: The batch to augment.
        :return: The augmented batch.
        """
        questions = self._ensure_list_of_questions(batch["question"])

        translated_questions = self.translate_one_step_batched(questions, self.to_tokenizer, self.to_model)
        translated_questions = self.translate_one_step_batched(
            translated_questions, self.from_tokenizer, self.from_model
        )

        batch["question"] = translated_questions
        return batch

    @torch.no_grad()
    def translate_one_step_batched(
        self,
        text: list[str],
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
    ):
        """
        Perform one step of translation for a batch of texts.

        :param text: The texts to translate.
        :param tokenizer: The tokenizer to use.
        :param model: The model to use.
        :return: The translated texts.
        """
        # TODO: Cleanup these .to() calls and batch_to_device() calls.
        #  Make sure that the model is on the correct device.
        processed_data = tokenizer(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model.to(self.device)
        model = cast(GenerationMixin, model)
        outputs = model.generate(
            input_ids=batch_to_device(processed_data["input_ids"], device=self.device),
            do_sample=True,
            top_k=100,
            top_p=0.95,
            max_length=100,  # TODO: Review these parameters.
        )
        return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]  # TODO: Maybe batch_decode()?


def default_text_batch_transforms_factory() -> Mapping[StageType, BatchTextTransformsType]:
    """
    Get the default text batch transforms.

    :return: The default text batch transforms.
    """
    return nn.ModuleDict(
        {
            "fit": QuestionAugmentationModule(),
            "validate": Noop(),
            "test": Noop(),
            "predict": Noop(),
        }
    )
