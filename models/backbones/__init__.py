"""The backbone models registry."""
from typing import Final, Protocol

import torch
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature

from utils.registry import Registry, RegistryKey
from utils.types import ImageType


class AvailableBackbones(RegistryKey):
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


class BackboneConfig(Protocol):
    """The backbone model config."""

    def get_model(self) -> torch.nn.Module:
        """Get the model."""
        raise NotImplementedError

    def get_image_processor(self) -> BaseImageProcessor:
        """Get the image processor."""
        ...

    def get_processed_image(self, processor: BaseImageProcessor, image: ImageType) -> BatchFeature:
        """Get the image features from processor."""
        ...

    def get_image_representation(
        self, model: torch.nn.Module, processor: BaseImageProcessor, image: ImageType
    ) -> torch.Tensor:
        """Get the image representation."""
        ...

    def get_image_representation_from_preprocessed(
        self, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the image representation."""
        ...

    def get_image_representation_size(self) -> int:
        """Get the image representation size."""
        ...

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        ...

    def get_tokenized_text(self, tokenizer: PreTrainedTokenizer, text: str) -> BatchEncoding:
        """Get the tokenized text."""
        ...

    def get_text_representation_size(self) -> int:
        """Get the text representation size."""
        ...

    def get_text_representation(
        self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, text: str
    ) -> torch.Tensor:
        """Get the text representation."""
        ...

    def get_text_representation_from_tokenized(
        self,
        model: torch.nn.Module,
        tokenizer_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        ...


class ImageEncoderMixin:
    """The image encoder mixin."""

    @staticmethod
    def get_processed_image(processor: BaseImageProcessor, image: ImageType) -> BatchFeature:
        """Get the image features from processor."""
        return processor(image, return_tensors="pt")

    @staticmethod
    def get_image_representation_from_preprocessed(
        model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the image representation."""
        return model(processor_output["pixel_values"]).pooler_output

    def get_image_representation(
        self, model: torch.nn.Module, processor: BaseImageProcessor, image: ImageType
    ) -> torch.Tensor:
        """Get the image representation."""
        return self.get_image_representation_from_preprocessed(
            model=model,
            processor_output=self.get_processed_image(processor=processor, image=image),
        )


class TextEncoderMixin:
    """The text encoder mixin."""

    @staticmethod
    def get_tokenized_text(tokenizer: PreTrainedTokenizer, text: str) -> BatchEncoding:
        """Get the tokenized text."""
        return tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_token_type_ids=("token_type_ids" in tokenizer.model_input_names),
            return_attention_mask=True,
        )

    @staticmethod
    def get_text_representation_from_tokenized(
        model: torch.nn.Module,
        tokenizer_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        return model(**tokenizer_output).pooler_output

    def get_text_representation(
        self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, text: str
    ) -> torch.Tensor:
        """Get the text representation."""
        return self.get_text_representation_from_tokenized(
            model=model,
            tokenizer_output=self.get_tokenized_text(tokenizer=tokenizer, text=text),
        )


class BackbonesRegistry(Registry[AvailableBackbones, type[BackboneConfig]]):
    """The backbones model registry."""

    def get_by_name(self, name: str) -> type[BackboneConfig] | None:
        """Get the backbone config by name."""
        return self.get(AvailableBackbones(name))

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        import models.backbones.configs  # noqa: F401


registry: Final[BackbonesRegistry] = BackbonesRegistry()
