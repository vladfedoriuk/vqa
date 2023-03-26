"""
The module contains utility functions for PyTorch.

The module contains the following functions:

- :func:`ensure_reproducibility` - Seed all the random number generators.

The module contains the following constants:

- :const:`device` - A device to be used.
"""
from typing import Any, Final

import torch
from lightning import seed_everything
from torch.types import Device

# A device to be used
device: Final[Device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Seed all the random number generators.

    Refer to https://pytorch.org/docs/stable/notes/randomness.html for more details.
    Also, refer to the Pytorch Lightning documentation for more details:

    https://pytorch-lightning.readthedocs.io/en/1.6.5/api/pytorch_lightning.utilities.seed.html

    :param seed: A seed to use.
    """
    seed_everything(seed, workers=True)


def squeeze_dict_of_tensors(
    dict_of_tensors: dict[Any, torch.Tensor]
) -> dict[Any, torch.Tensor]:
    """
    Squeeze the dict of tensors.

    :param dict_of_tensors: The dict of tensors.
    :return: The squeezed dict of tensors.
    """
    return {key: value.squeeze() for key, value in dict_of_tensors.items()}
