"""The backbone configurations."""
import dataclasses
from collections.abc import Sequence

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
    BeitModel,
    CLIPModel,
    CLIPProcessor,
    DeiTModel,
    GPT2TokenizerFast,
    PreTrainedTokenizer,
    ViltModel,
    ViltProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)
from transformers.image_processing_utils import BaseImageProcessor

from models.backbones import (
    AvailableBackbones,
    BackboneConfig,
    ImageEncoderMixin,
    TextEncoderMixin,
    registry,
)
from utils.types import ImageType


@registry.register(AvailableBackbones.DINO)
@dataclasses.dataclass(frozen=True)
class DINOConfig(ImageEncoderMixin, BackboneConfig):
    """The DINO backbone."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return AutoModel.from_pretrained("facebook/dino-vitb16")

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("facebook/dino-vitb16")

    @classmethod
    def get_image_representation_size(cls) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.BEIT)
@dataclasses.dataclass(frozen=True)
class BEITConfig(ImageEncoderMixin, BackboneConfig):
    """The BEiT backbone."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

    @classmethod
    def get_image_representation_size(cls) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.DEIT)
@dataclasses.dataclass(frozen=True)
class DEITConfig(ImageEncoderMixin, BackboneConfig):
    """The DeiT backbone."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

    @classmethod
    def get_image_representation_size(cls) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.RESNET)
@dataclasses.dataclass(frozen=True)
class RESNETConfig(ImageEncoderMixin, BackboneConfig):
    """The ResNet backbone."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return AutoModel.from_pretrained("microsoft/resnet-50")

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    @classmethod
    def get_image_representation_size(cls) -> int:
        """Get the image representation size."""
        return 2048

    @classmethod
    def get_image_representation_from_preprocessed(
        cls, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the image representation."""
        return model(processor_output["pixel_values"]).pooler_output.squeeze().expand(1, -1)


@registry.register(AvailableBackbones.VIT)
@dataclasses.dataclass(frozen=True)
class VITConfig(ImageEncoderMixin, BackboneConfig):
    """The ViT model is a transformer-based model."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    @classmethod
    def get_image_representation_size(cls) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.BERT)
@dataclasses.dataclass(frozen=True)
class BERTConfig(TextEncoderMixin, BackboneConfig):
    """The BERT model is a transformer-based model."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return AutoModel.from_pretrained("bert-base-uncased")

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    @classmethod
    def get_text_representation_size(cls) -> int:
        """Get the text representation size."""
        return 768


class TextEncoderLastHiddenStateMixin(TextEncoderMixin):
    """The text encoder last hidden state mixin."""

    @classmethod
    def get_text_representation_from_tokenized(
        cls,
        model: torch.nn.Module,
        tokenizer_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        model_kwargs = {
            "input_ids": tokenizer_output["input_ids"],
            "attention_mask": tokenizer_output["attention_mask"],
        }
        if "token_type_ids" in tokenizer_output:
            model_kwargs["token_type_ids"] = tokenizer_output["token_type_ids"]
        return model(**model_kwargs).last_hidden_state[:, 0, :]


@registry.register(AvailableBackbones.ROBERTA)
@dataclasses.dataclass(frozen=True)
class ROBERTAConfig(TextEncoderLastHiddenStateMixin, BackboneConfig):
    """The RoBERTa model is a transformer-based model."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return AutoModel.from_pretrained("roberta-base")

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("roberta-base")

    @classmethod
    def get_text_representation_size(cls) -> int:
        """Get the text representation size."""
        return 768


@registry.register(AvailableBackbones.DISTILBERT)
@dataclasses.dataclass(frozen=True)
class DISTILBERTConfig(TextEncoderLastHiddenStateMixin, BackboneConfig):
    """The DistilBERT model is a transformer-based model."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return AutoModel.from_pretrained("distilbert-base-uncased")

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @classmethod
    def get_text_representation_size(cls) -> int:
        """Get the text representation size."""
        return 768

    @classmethod
    def get_text_representation_from_tokenized(
        cls,
        model: torch.nn.Module,
        tokenizer_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        text_encoder_kwargs = {
            "input_ids": tokenizer_output["input_ids"],
            "attention_mask": tokenizer_output["attention_mask"],
        }
        return model(**text_encoder_kwargs).last_hidden_state[:, 0, :]


@registry.register(AvailableBackbones.ALBERT)
@dataclasses.dataclass(frozen=True)
class ALBERTConfig(TextEncoderLastHiddenStateMixin, BackboneConfig):
    """The ALBERT model is a transformer-based model."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return AutoModel.from_pretrained("albert-base-v2")

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("albert-base-v2")

    @classmethod
    def get_text_representation_size(cls) -> int:
        """Get the text representation size."""
        return 768


@registry.register(AvailableBackbones.CLIP)
@dataclasses.dataclass(frozen=True)
class CLIPConfig(ImageEncoderMixin, TextEncoderMixin, BackboneConfig):
    """CLIP is a transformer-based model."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    @classmethod
    def get_processor(cls):
        """Get the processor."""
        return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer."""
        return cls.get_processor()

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return cls.get_processor()

    @classmethod
    def get_representation_size(cls) -> int:
        """Get the representation size."""
        return 512

    @classmethod
    def get_text_representation_size(cls) -> int:
        """Get the text representation size."""
        return cls.get_representation_size()

    @classmethod
    def get_image_representation_size(cls) -> int:
        """Get the image representation size."""
        return cls.get_representation_size()

    @classmethod
    def get_text_representation_from_tokenized(
        cls,
        model: torch.nn.Module,
        processor_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        # TODO use processor_output instead of tokenizer_output everywhere
        return model(
            input_ids=processor_output["input_ids"],
            attention_mask=processor_output["attention_mask"],
            pixel_values=processor_output["pixel_values"],
        ).text_embeds

    @classmethod
    def get_image_representation_from_preprocessed(
        cls, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the image representation."""
        return model(
            input_ids=processor_output["input_ids"],
            attention_mask=processor_output["attention_mask"],
            pixel_values=processor_output["pixel_values"],
        ).image_embeds


@registry.register(AvailableBackbones.ViLT_MLM)
@dataclasses.dataclass(frozen=True)
class ViLTMLMConfig(BackboneConfig):  # TODO: Refactor the backbone configs - design proper inheritance and interfaces
    """
    ViLT-MLM is a transformer-based model for multimodal pretraining.

    It was pretrained to predict masked tokens in images and text.
    """

    @classmethod
    def get_model(cls):
        """Get the model."""
        return ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

    @classmethod
    def get_processor(cls) -> ViltProcessor:
        """Get the processor."""
        return ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    @classmethod
    def get_multimodal_representation_size(cls) -> int:
        """Get the multimodal representation size."""
        return 768

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer."""
        return cls.get_processor().tokenizer  # type: ignore

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return cls.get_processor().image_processor  # type: ignore

    @classmethod
    def get_processed_image(cls, processor: BaseImageProcessor, image: ImageType | Sequence[ImageType]) -> BatchFeature:
        """Get the image features from processor."""
        # TODO: rename the processor to image_processor
        return processor(
            images=image,
            return_tensors="pt",
            do_pad=True,
        )

    @classmethod
    def get_tokenized_text(cls, tokenizer: PreTrainedTokenizer, text: str | list[str]) -> BatchEncoding:
        """Get the tokenized text."""
        return tokenizer(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

    @classmethod
    def get_processed_text_and_image(
        cls,
        processor: ViltProcessor,
        text: str | list[str],
        image: ImageType | Sequence[ImageType],  # TODO: rename to images everywhere
    ) -> BatchEncoding:
        """Get the processed text and image."""
        return processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

    @classmethod
    def get_multimodal_representation_from_preprocessed(
        cls, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the multimodal representation."""
        return model(
            input_ids=processor_output["input_ids"],
            token_type_ids=processor_output["token_type_ids"],
            attention_mask=processor_output["attention_mask"],
            pixel_values=processor_output["pixel_values"],
            pixel_mask=processor_output["pixel_mask"],
            return_dict=True,
        ).pooler_output


@registry.register(AvailableBackbones.ViLT_VQA)
@dataclasses.dataclass(frozen=True)
class ViLTVQAConfig(ViLTMLMConfig):
    """ViLT fine-tuned on VQA."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    @classmethod
    def get_processor(cls) -> ViltProcessor:
        """Get the processor."""
        return ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


@registry.register(AvailableBackbones.VIT_GPT2)
@dataclasses.dataclass(frozen=True)
class ViTGPT2Config(BackboneConfig):
    """ViT-GPT2 is a transformer Encoder-Decoder model for image captioning."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer."""
        return GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    @classmethod
    def get_image_processor(cls):
        """Get the image processor."""
        return ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    @classmethod
    def get_processed_image(cls, processor: BaseImageProcessor, image: ImageType | Sequence[ImageType]) -> BatchFeature:
        """Get the image features from processor."""
        return processor(
            images=image,
            return_tensors="pt",
        )

    @classmethod
    def get_tokenized_text(cls, tokenizer: PreTrainedTokenizer, text: str | list[str]) -> BatchEncoding:
        """Get the tokenized text."""
        return tokenizer(
            text=text,
            return_tensors="pt",
            padding=True,
            max_length=40,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )


@registry.register(AvailableBackbones.ViT_BERT)
@dataclasses.dataclass(frozen=True)
class ViTBertConfig(VITConfig, BERTConfig):
    """DEIT-BERT is a transformer Encoder-Decoder model."""

    @classmethod
    def get_model(cls):
        """Get the model."""
        return VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "bert-base-uncased"
        )

    @classmethod
    def get_tokenized_text(cls, tokenizer: PreTrainedTokenizer, text: str | list[str]) -> BatchEncoding:
        """Get the tokenized text."""
        return tokenizer(
            text=text,
            return_tensors="pt",
            padding=True,
            max_length=40,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
