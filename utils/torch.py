"""
The module contains utility functions for PyTorch.

The module contains the following functions:

- :func:`ensure_reproducibility` - Seed all the random number generators.

The module contains the following constants:

- :const:`device` - A device to be used.
"""
import random
from typing import Final

import numpy as np
import torch
from torch.types import Device

# A device to be used
device: Final[Device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Seed all the random number generators.

    Refer to https://pytorch.org/docs/stable/notes/randomness.html for more details.

    :param seed: A seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
