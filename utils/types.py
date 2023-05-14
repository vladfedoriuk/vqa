"""Type definitions for the project."""
from collections.abc import Callable
from typing import Any, Literal

import PIL.Image
import torch.optim
from transformers import BatchEncoding, BatchFeature

ImageType = PIL.Image.Image | torch.Tensor

BatchType = dict[str, Any] | list[dict[str, Any]] | BatchEncoding | BatchFeature | torch.Tensor

SingleImageTransformsType = Callable[[PIL.Image.Image], ImageType]

SingeTextTransformsType = Callable[[str], str]

BatchImageTransformsType = Callable[[BatchType], BatchType]

BatchTextTransformsType = Callable[[BatchType], BatchType]

StageType = Literal["fit", "validate", "test", "predict"]
