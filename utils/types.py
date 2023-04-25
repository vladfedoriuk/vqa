"""Type definitions for the project."""
from collections.abc import Callable
from typing import Any, TypeGuard

import PIL.Image
import torch.optim

from utils.phase import Phase

ImageType = PIL.Image.Image | torch.Tensor

TransformsType = dict[Phase, Callable[[PIL.Image.Image], ImageType]]

SingleImageTransformsType = Callable[[PIL.Image.Image], ImageType]


def is_callable(obj: Any) -> TypeGuard[Callable]:
    """Check if the tokenizer is callable."""
    return callable(obj)
