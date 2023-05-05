"""Type definitions for the project."""
from collections.abc import Callable

import PIL.Image
import torch.optim

ImageType = PIL.Image.Image | torch.Tensor

SingleImageTransformsType = Callable[[PIL.Image.Image], ImageType]
