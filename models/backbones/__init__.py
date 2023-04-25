"""The backbone models registry."""
from typing import Final, Protocol, cast

import torch
from transformers import AutoImageProcessor, AutoTokenizer

from utils.registry import Registry, RegistryKey
from utils.types import ImageType, is_callable


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

    def get_image_processor(self) -> AutoImageProcessor:
        """Get the image processor."""
        ...

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer."""
        ...

    def get_image_representation_size(self) -> int:
        """Get the image representation size."""
        ...

    def get_text_representation_size(self) -> int:
        """Get the text representation size."""
        ...

    def get_image_representation(
        self, model: torch.nn.Module, image: ImageType
    ) -> torch.Tensor:
        """Get the image representation."""
        ...

    def get_text_representation(
        self, model: torch.nn.Module, text: str
    ) -> torch.Tensor:
        """Get the text representation."""
        ...


class ImageEncoderMixin:
    """The image encoder mixin."""

    def get_image_representation(
        self, model: torch.nn.Module, image: ImageType
    ) -> torch.Tensor:
        """Get the image representation."""
        instance = cast(BackboneConfig, self)
        image_processor = instance.get_image_processor()
        assert is_callable(image_processor), "The image processor must be callable."
        image = image_processor(image, return_tensors="pt")
        return model(image["pixel_values"]).pooler_output


class TextEncoderMixin:
    """The text encoder mixin."""

    def get_text_representation(
        self, model: torch.nn.Module, text: str
    ) -> torch.Tensor:
        """Get the text representation."""
        instance = cast(BackboneConfig, self)
        tokenizer = instance.get_tokenizer()
        assert is_callable(tokenizer), "The tokenizer must be callable."
        inputs = tokenizer(text, return_tensors="pt")
        return model(**inputs).pooler_output


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
