"""
The module contains utility functions for PyTorch.

The module contains the following functions:

- :func:`ensure_reproducibility` - Seed all the random number generators.

The module contains the following constants:

- :const:`device` - A device to be used.
"""
from collections.abc import Callable, Sequence
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


def freeze_model_parameters(model, excludes: Sequence[Callable[[str], bool]] = ()):
    """
    Freezes some model parameters.

    One can specify which types of parameters to freeze by providing excludes callables.
    An 'exclude' callable accepts a name of a parameter and is expected to return
    a boolean value, meaning whether the parameter should require gradient or not.

    :param model: A model to partially freeze.
    :param excludes: A sequence of 'exclude' callables.
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
        if any(exclude(name) for exclude in excludes):
            param.requires_grad = True
