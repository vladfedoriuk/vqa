"""The backbone models registry."""
from collections.abc import Sequence
from typing import Final, TypedDict, cast

import torch
from torch import classproperty
from transformers import BatchEncoding, PreTrainedTokenizer, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature

from utils.registry import Registry, RegistryKey
from utils.types import ImageType

# TODO: Refactor the backbones - use Interface segregation principle


class AvailableBackbones(str, RegistryKey):
    """The available backbones."""

    # Image backbones
    # Model from https://huggingface.co/google/vit-base-patch16-224-in21k
    VIT = "google/vit-base-patch16-224-in21k"
    # Model from https://huggingface.co/facebook/dino-vitb16
    DINO = "facebook/dino-vitb16"
    # Model from https://huggingface.co/microsoft/beit-base-patch16-224-pt22k
    BEIT = "microsoft/beit-base-patch16-224-pt22k"
    # Model from https://huggingface.co/facebook/deit-base-distilled-patch16-224
    DEIT = "facebook/deit-base-distilled-patch16-224"
    # Model from https://huggingface.co/microsoft/resnet-50
    RESNET = "microsoft/resnet-50"

    # Text backbones
    # Model from https://huggingface.co/bert-base-uncased
    BERT = "bert-base-uncased"
    # Model from https://huggingface.co/distilbert-base-uncased
    DISTILBERT = "distilbert-base-uncased"
    # Model from https://huggingface.co/roberta-base
    ROBERTA = "roberta-base"
    # Model from https://huggingface.co/albert-base-v2
    ALBERT = "albert-base-v2"

    # Multi-modal backbones
    # Model from https://huggingface.co/openai/clip-vit-base-patch16
    CLIP = "openai/clip-vit-base-patch16"
    # Model from https://huggingface.co/dandelin/vilt-b32-mlm
    ViLT_MLM = "dandelin/vilt-b32-mlm"
    # Model from https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
    ViLT_VQA = "dandelin/vilt-b32-finetuned-vqa"
    # Model from https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
    VIT_GPT2 = "nlpconnect/vit-gpt2-image-captioning"

    # TODO: differentiate between encoder backbones and encoder-decoder backbones
    @classproperty
    def image_backbones(cls) -> tuple["AvailableBackbones", ...]:  # TODO: Refactor this - move to variables
        """Get the image backbones."""
        return (
            cls.VIT,
            cls.DINO,
            cls.BEIT,
            cls.DEIT,
            cls.RESNET,
        )

    @classproperty
    def text_backbones(cls) -> tuple["AvailableBackbones", ...]:
        """Get the text backbones."""
        return (
            cls.BERT,
            cls.DISTILBERT,
            cls.ROBERTA,
            cls.ALBERT,
        )

    @classproperty
    def multimodal_backbones(cls) -> tuple["AvailableBackbones", ...]:
        """Get the multimodal backbones."""
        return (
            cls.CLIP,
            cls.ViLT_MLM,
            cls.ViLT_VQA,
            cls.VIT_GPT2,
        )


class BackboneConfig:
    """The backbone model config."""

    @classmethod
    def get_model(cls) -> torch.nn.Module:
        """Get the model."""
        raise NotImplementedError

    @classmethod
    def get_processor(cls) -> ProcessorMixin:
        """Get the universal processor."""
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
    def get_tokenized_text(cls, tokenizer: PreTrainedTokenizer, text: str | list[str]) -> BatchEncoding:
        """Get the tokenized text."""
        raise NotImplementedError

    @classmethod
    def get_text_representation_size(cls) -> int:
        """Get the text representation size."""
        raise NotImplementedError

    @classmethod
    def get_text_representation(
        cls, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, text: str | list[str]
    ) -> torch.Tensor:
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

    @classmethod
    def get_multimodal_representation_size(cls) -> int:
        """Get the multimodal representation size."""
        raise NotImplementedError

    @classmethod
    def get_multimodal_representation_from_preprocessed(
        cls, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the multimodal representation."""
        raise NotImplementedError

    @classmethod
    def get_multimodal_representation(
        cls,
        model: torch.nn.Module,
        processor: BaseImageProcessor,
        image: ImageType,
        tokenizer: PreTrainedTokenizer,
        text: str | list[str],
    ) -> torch.Tensor:
        """Get the multimodal representation."""
        raise NotImplementedError


class ImageEncoderMixin:
    """The image encoder mixin."""

    @classmethod
    def get_processed_image(cls, processor: BaseImageProcessor, image: ImageType | Sequence[ImageType]) -> BatchFeature:
        """Get the image features from processor."""
        return processor(images=image, return_tensors="pt")

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
    def get_tokenized_text(cls, tokenizer: PreTrainedTokenizer, text: str | list[str]) -> BatchEncoding:
        """Get the tokenized text."""
        return tokenizer(
            text=text,
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
    def get_text_representation(
        cls, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, text: str | list[str]
    ) -> torch.Tensor:
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


class BackbonesConfigs(TypedDict, total=False):
    """The backbones configs."""

    image_backbone: AvailableBackbones | None
    text_backbone: AvailableBackbones | None
    multimodal_backbone: AvailableBackbones | None


class BackbonesConfigValues(TypedDict, total=False):
    """The backbones config values."""

    image_encoder: torch.nn.Module
    text_encoder: torch.nn.Module
    tokenizer: PreTrainedTokenizer
    image_processor: BaseImageProcessor
    image_representation_size: int
    text_representation_size: int
    image_backbone_config: type[BackboneConfig]
    text_backbone_config: type[BackboneConfig]


def prepare_backbones(backbones: BackbonesConfigs) -> BackbonesConfigValues:
    """
    Prepare the backbones.

    :param backbones: The backbones.
    :return: The backbones config classes, models, tokenizers, and processors.
    """
    image_backbone = backbones.get("image_backbone")
    text_backbone = backbones.get("text_backbone")
    multimodal_backbone = backbones.get("multimodal_backbone")

    if not any([image_backbone, text_backbone, multimodal_backbone]):
        raise ValueError("At least one backbone must be specified.")

    config_values = {}

    if multimodal_backbone is not None:
        config_values |= _handle_multimodal_backbone_config_values(multimodal_backbone)
    if image_backbone is not None:
        config_values |= _handle_image_backbone_config_values(image_backbone)
    if text_backbone is not None:
        config_values |= _handle_text_backbone_config_values(text_backbone)
    return config_values


def _handle_multimodal_backbone_config_values(backbone: AvailableBackbones) -> BackbonesConfigValues:
    """Handle the multimodal backbone config values."""
    backbone_config = registry.get(backbone)
    if backbone_config is None:
        raise ValueError(f"Backbone {backbone} is not supported.")
    if backbone not in AvailableBackbones.multimodal_backbones:
        raise ValueError(f"Backbone {backbone} is not multimodal.")

    multimodal_model = backbone_config.get_model()
    multimodal_processor = backbone_config.get_processor()
    return {
        "image_encoder": multimodal_model,
        "text_encoder": multimodal_model,
        "tokenizer": cast(PreTrainedTokenizer, multimodal_processor),
        "image_processor": cast(BaseImageProcessor, multimodal_processor),
        "image_representation_size": backbone_config.get_image_representation_size(),
        "text_representation_size": backbone_config.get_text_representation_size(),
        "image_backbone_config": backbone_config,
        "text_backbone_config": backbone_config,
    }


def _handle_image_backbone_config_values(backbone: AvailableBackbones) -> BackbonesConfigValues:
    """Handle the image backbone config values."""
    backbone_config = registry.get(backbone)
    if backbone_config is None:
        raise ValueError(f"Backbone {backbone} is not supported.")
    if backbone not in AvailableBackbones.image_backbones:
        raise ValueError(f"Backbone {backbone} is not a vision backbone.")
    image_model = backbone_config.get_model()
    image_processor = backbone_config.get_image_processor()
    return {
        "image_encoder": image_model,
        "image_processor": image_processor,
        "image_representation_size": backbone_config.get_image_representation_size(),
        "image_backbone_config": backbone_config,
    }


def _handle_text_backbone_config_values(backbone: AvailableBackbones) -> BackbonesConfigValues:
    """Handle the text backbone config values."""
    backbone_config = registry.get(backbone)
    if backbone_config is None:
        raise ValueError(f"Backbone {backbone} is not supported.")
    if backbone not in AvailableBackbones.text_backbones:
        raise ValueError(f"Backbone {backbone} is not a text backbone.")
    text_model = backbone_config.get_model()
    tokenizer = backbone_config.get_tokenizer()
    return {
        "text_encoder": text_model,
        "tokenizer": tokenizer,
        "text_representation_size": backbone_config.get_text_representation_size(),
        "text_backbone_config": backbone_config,
    }
