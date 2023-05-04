"""The backbone models registry."""
from collections.abc import Sequence
from typing import Final

import torch
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature

from utils.registry import Registry, RegistryKey
from utils.types import ImageType


class AvailableBackbones(str, RegistryKey):
    """The available backbones."""

    # Image backbones
    VIT = "google/vit-base-patch16-224-in21k"
    DINO = "facebook/dino-vitb16"
    BEIT = "microsoft/beit-base-patch16-224-pt22k"
    DEIT = "facebook/deit-base-distilled-patch16-224"
    RESNET = "microsoft/resnet-50"

    # Text backbones
    BERT = "bert-base-uncased"
    DISTILBERT = "distilbert-base-uncased"
    ROBERTA = "roberta-base"
    ALBERT = "albert-base-v2"


class BackboneConfig:
    """The backbone model config."""

    @classmethod
    def get_model(cls) -> torch.nn.Module:
        """Get the model."""
        raise NotImplementedError

    @classmethod
    def get_image_processor(cls) -> BaseImageProcessor:
        """Get the image processor."""
        raise NotImplementedError

    @classmethod
    def get_processed_image(cls, processor: BaseImageProcessor, image: ImageType | Sequence[ImageType]) -> BatchFeature:
        """Get the image features from processor."""
        raise NotImplementedError

    @classmethod
    def get_image_representation(
        cls, model: torch.nn.Module, processor: BaseImageProcessor, image: ImageType
    ) -> torch.Tensor:
        """Get the image representation."""
        raise NotImplementedError

    @classmethod
    def get_image_representation_from_preprocessed(
        cls, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the image representation."""
        raise NotImplementedError

    @classmethod
    def get_image_representation_size(cls) -> int:
        """Get the image representation size."""
        raise NotImplementedError

    @classmethod
    def get_tokenizer(cls) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        raise NotImplementedError

    @classmethod
    def get_tokenized_text(cls, tokenizer: PreTrainedTokenizer, text: str) -> BatchEncoding:
        """Get the tokenized text."""
        raise NotImplementedError

    @classmethod
    def get_text_representation_size(cls) -> int:
        """Get the text representation size."""
        raise NotImplementedError

    @classmethod
    def get_text_representation(cls, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, text: str) -> torch.Tensor:
        """Get the text representation."""
        raise NotImplementedError

    @classmethod
    def get_text_representation_from_tokenized(
        cls,
        model: torch.nn.Module,
        tokenizer_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        raise NotImplementedError


class ImageEncoderMixin:
    """The image encoder mixin."""

    @classmethod
    def get_processed_image(cls, processor: BaseImageProcessor, image: ImageType | Sequence[ImageType]) -> BatchFeature:
        """Get the image features from processor."""
        return processor(image, return_tensors="pt")

    @classmethod
    def get_image_representation_from_preprocessed(
        cls, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the image representation."""
        return model(processor_output["pixel_values"]).pooler_output

    @classmethod
    def get_image_representation(
        cls, model: torch.nn.Module, processor: BaseImageProcessor, image: ImageType
    ) -> torch.Tensor:
        """Get the image representation."""
        return cls.get_image_representation_from_preprocessed(
            model=model,
            processor_output=cls.get_processed_image(processor=processor, image=image),
        )


class TextEncoderMixin:
    """The text encoder mixin."""

    @classmethod
    def get_tokenized_text(cls, tokenizer: PreTrainedTokenizer, text: str) -> BatchEncoding:
        """Get the tokenized text."""
        return tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_token_type_ids=("token_type_ids" in tokenizer.model_input_names),
            return_attention_mask=True,
        )

    @classmethod
    def get_text_representation_from_tokenized(
        cls,
        model: torch.nn.Module,
        tokenizer_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        return model(**tokenizer_output).pooler_output

    @classmethod
    def get_text_representation(cls, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, text: str) -> torch.Tensor:
        """Get the text representation."""
        return cls.get_text_representation_from_tokenized(
            model=model,
            tokenizer_output=cls.get_tokenized_text(tokenizer=tokenizer, text=text),
        )


class BackbonesRegistry(Registry[AvailableBackbones, type[BackboneConfig]]):
    """The backbones model registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        import models.backbones.configs  # noqa: F401


registry: Final[BackbonesRegistry] = BackbonesRegistry()
