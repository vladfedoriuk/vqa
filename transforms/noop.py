"""No-op transform."""
from typing import Any

import torch
from torch import nn

from utils.torch import DeviceAwareModule


def noop(data: Any) -> Any:
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
    def forward(self, data: Any) -> Any:
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
