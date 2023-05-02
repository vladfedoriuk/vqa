"""Type definitions for the project."""
from collections.abc import Callable

import PIL.Image
import torch.optim

from utils.phase import Phase

ImageType = PIL.Image.Image | torch.Tensor

TransformsType = dict[Phase, Callable[[PIL.Image.Image], ImageType]]

SingleImageTransformsType = Callable[[PIL.Image.Image], ImageType]
