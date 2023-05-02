"""
The module contains utility functions for PyTorch.

The module contains the following functions:

- :func:`ensure_reproducibility` - Seed all the random number generators.
"""
from collections.abc import Callable, Sequence
from typing import Any

import torch
from lightning import seed_everything
from torch import nn
from transformers import BatchEncoding, BatchFeature


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
    dict_of_tensors: dict[Any, torch.Tensor] | BatchEncoding | BatchFeature
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


def backbone_name_to_kebab_case(backbone_name: str) -> str:
    """
    Convert the backbone name to kebab case.

    :param backbone_name: The name of the backbone.
    :return: The backbone name in kebab case.
    """
    return backbone_name.replace("/", "-")


def initialize_linear_weights(model: nn.Module):
    """
    Initialize the linear weights.

    :param model: A model.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
