"""The module contains the code for the fusion-based models."""
import torch
from torch import nn

from utils.registry import Registry, RegistryKey
from utils.torch import initialize_linear_weights


class BaseFusionModel(nn.Module):
    """A base fusion model."""

    def __init__(
        self,
        image_representation_size: int,
        text_representation_size: int,
        final_representation_size: int,
    ):
        """
        Initialize the model.

        :param final_representation_size: The final representation size.
        :param image_representation_size: The image representation size.
        :param text_representation_size: The text representation size.
        """
        super().__init__()
        self.final_representation_size = final_representation_size
        self.image_representation_size = image_representation_size
        self.text_representation_size = text_representation_size

    def _initialize_weights(self):
        """
        Initialize the text projection, vision projection, fusion, and classifier weights.

        :return: None
        """
        initialize_linear_weights(self)

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ):
        """
        Perform forward pass.

        :param image_emb: The image embedding.
                            Shape: (batch_size, image_representation_size)
        :param text_emb: The text embedding.
                            Shape: (batch_size, text_representation_size)
        :return: The logits.
        """
        raise NotImplementedError


class AvailableFusionModels(str, RegistryKey):
    """The available fusions."""

    # Image backbones
    CAT = "concatenation-fusion"
    ATTENTION = "attention-fusion"
    DEEP_SET = "deep-set-fusion"
    DEEP_SET_TRANSFORMER = "deep-set-transformer-fusion"


class FusionModelsRegistry(Registry[AvailableFusionModels, type[BaseFusionModel]]):
    """The available fusion models registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from models.fusions import models  # noqa: F401


registry = FusionModelsRegistry()

__all__ = [
    "BaseFusionModel",
    "AvailableFusionModels",
    "FusionModelsRegistry",
    "registry",
]
