"""A module for  accumulating the fusion models."""
from models.fusions.attention import AttentionFusionModel  # noqa: F401
from models.fusions.concatenation import ConcatenationFusionModel  # noqa: F401
from models.fusions.deep_set import (  # noqa: F401
    DeepSetFusionModel,
    DeepSetTransformerFusionModel,
)

__all__ = ["ConcatenationFusionModel", "AttentionFusionModel", "DeepSetFusionModel", "DeepSetTransformerFusionModel"]
