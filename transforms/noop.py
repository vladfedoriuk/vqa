"""No-op transform."""
from typing import TypeVar

import torch
from torch import nn

DataType = TypeVar("DataType")


def noop(data: DataType) -> DataType:
    """
    No-op transform.

    :param data: Any data.
    :return: The same data.
    """
    return data


class Noop(nn.Module):
    """
    The no-op Pytorch Module.

    Can be used with e.g :py:class:`~torch.nn.ModuleDict`
    """

    @torch.no_grad()
    def forward(self, data: DataType) -> DataType:
        """
        Forward pass.

        :param data: Any data.
        :return: The same data.
        """
        return data


def default_noop_transforms_factory():
    """
    Create default no-op transforms.

    :return: The no-op transforms.
    """
    return nn.ModuleDict(
        {
            "fit": Noop(),
            "validate": Noop(),
            "test": Noop(),
            "predict": Noop(),
        }
    )
