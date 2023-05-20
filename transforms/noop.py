"""No-op transform."""
from typing import TypeVar

import torch
from torch import nn

from utils.batch import batch_to_device
from utils.torch import DeviceAwareModule

DataType = TypeVar("DataType")


def noop(data: DataType) -> DataType:
    """
    No-op transform.

    :param data: Any data.
    :return: The same data.
    """
    return data


class Noop(DeviceAwareModule):
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
        return batch_to_device(data, device=self.device)


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
