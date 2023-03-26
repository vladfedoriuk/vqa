"""Type definitions for the project."""
from collections.abc import Callable

import PIL.Image
import torch.optim

from utils.phase import Phase

TransformsType = dict[
    Phase, Callable[[PIL.Image.Image], PIL.Image.Image | torch.Tensor]
]

SingleImageTransformsType = Callable[[PIL.Image.Image], PIL.Image.Image | torch.Tensor]
