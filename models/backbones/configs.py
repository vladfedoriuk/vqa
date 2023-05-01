"""The backbone configurations."""
import dataclasses

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
    BeitModel,
    DeiTModel,
)

from models.backbones import (
    AvailableBackbones,
    BackboneConfig,
    ImageEncoderMixin,
    TextEncoderMixin,
    registry,
)


@registry.register(AvailableBackbones.DINO)
@dataclasses.dataclass(frozen=True)
class DINOConfig(ImageEncoderMixin, BackboneConfig):
    """The DINO backbone."""

    def get_model(self):
        """Get the model."""
        return AutoModel.from_pretrained("facebook/dino-vitb16")

    def get_image_processor(self):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("facebook/dino-vitb16")

    def get_image_representation_size(self) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.BEIT)
@dataclasses.dataclass(frozen=True)
class BEITConfig(ImageEncoderMixin, BackboneConfig):
    """The BEiT backbone."""

    def get_model(self):
        """Get the model."""
        return BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

    def get_image_processor(self):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

    def get_image_representation_size(self) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.DEIT)
@dataclasses.dataclass(frozen=True)
class DEITConfig(ImageEncoderMixin, BackboneConfig):
    """The DeiT backbone."""

    def get_model(self):
        """Get the model."""
        return DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

    def get_image_processor(self):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

    def get_image_representation_size(self) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.RESNET)
@dataclasses.dataclass(frozen=True)
class RESNETConfig(ImageEncoderMixin, BackboneConfig):
    """The ResNet backbone."""

    def get_model(self):
        """Get the model."""
        return AutoModel.from_pretrained("microsoft/resnet-50")

    def get_image_processor(self):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    def get_image_representation_size(self) -> int:
        """Get the image representation size."""
        return 2048

    def get_image_representation_from_preprocessed(
        self, model: torch.nn.Module, processor_output: BatchFeature
    ) -> torch.Tensor:
        """Get the image representation."""
        return model(processor_output["pixel_values"]).pooler_output.squeeze().expand(1, -1)


@registry.register(AvailableBackbones.VIT)
@dataclasses.dataclass(frozen=True)
class VITConfig(ImageEncoderMixin, BackboneConfig):
    """The ViT model is a transformer-based model."""

    def get_model(self):
        """Get the model."""
        return AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def get_image_processor(self):
        """Get the image processor."""
        return AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def get_image_representation_size(self) -> int:
        """Get the image representation size."""
        return 768


@registry.register(AvailableBackbones.BERT)
@dataclasses.dataclass(frozen=True)
class BERTConfig(TextEncoderMixin, BackboneConfig):
    """The BERT model is a transformer-based model."""

    def get_model(self):
        """Get the model."""
        return AutoModel.from_pretrained("bert-base-uncased")

    def get_tokenizer(self):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def get_text_representation_size(self) -> int:
        """Get the text representation size."""
        return 768


class TextEncoderLastHiddenStateMixin(TextEncoderMixin):
    """The text encoder last hidden state mixin."""

    @staticmethod
    def get_text_representation_from_tokenized(
        model: torch.nn.Module,
        tokenizer_output: BatchEncoding,
    ) -> torch.Tensor:
        """Get the text representation."""
        return model(**tokenizer_output).last_hidden_state[:, 0, :]


@registry.register(AvailableBackbones.ROBERTA)
@dataclasses.dataclass(frozen=True)
class ROBERTAConfig(TextEncoderLastHiddenStateMixin, BackboneConfig):
    """The RoBERTa model is a transformer-based model."""

    def get_model(self):
        """Get the model."""
        return AutoModel.from_pretrained("roberta-base")

    def get_tokenizer(self):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("roberta-base")

    def get_text_representation_size(self) -> int:
        """Get the text representation size."""
        return 768


@registry.register(AvailableBackbones.DISTILBERT)
@dataclasses.dataclass(frozen=True)
class DISTILBERTConfig(TextEncoderLastHiddenStateMixin, BackboneConfig):
    """The DistilBERT model is a transformer-based model."""

    def get_model(self):
        """Get the model."""
        return AutoModel.from_pretrained("distilbert-base-uncased")

    def get_tokenizer(self):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def get_text_representation_size(self) -> int:
        """Get the text representation size."""
        return 768

    @staticmethod
    def get_text_representation_from_tokenized(
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

    def get_model(self):
        """Get the model."""
        return AutoModel.from_pretrained("albert-base-v2")

    def get_tokenizer(self):
        """Get the tokenizer."""
        return AutoTokenizer.from_pretrained("albert-base-v2")

    def get_text_representation_size(self) -> int:
        """Get the text representation size."""
        return 768
