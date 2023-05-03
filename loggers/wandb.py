"""
A module contains the logger functionalities for Weights & Biases.

For more information, see:

- https://docs.wandb.ai/guides/integrations/lightning
- https://lightning.ai/docs/pytorch/latest/extensions/generated/pytorch_lightning.loggers.WandbLogger.html
"""  # noqa: E501
from functools import lru_cache

import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb.sdk.wandb_run import Run

from config.env import WANDB_PROJECT


@lru_cache(maxsize=1)
def get_lightning_logger(
    run_name: str,
) -> WandbLogger:
    """
    Get the logger for Weights & Biases.

    .. note::
        The logger is meant to be used with Pytorch Lightning.

        See:
        https://docs.wandb.ai/guides/integrations/lightning#using-pytorch-lightnings-wandblogger

    :return: The logger.
    """
    return WandbLogger(
        project=WANDB_PROJECT,
        name=run_name,
        log_model="all",
    )


def log_confusion_matrix(run: Run, cm: torch.Tensor, caption: str):
    """
    Log the confusion matrix to Weights & Biases.

    :param run: The run.
    :param cm: The confusion matrix (as obtained from :py:func:`torchmetrics.functional.confusion_matrix`).
    :param caption: The caption.
    :return:
    """
    cm = cm.cpu().numpy().squeeze()
    image = wandb.Image(cm, caption=caption)
    run.log({"confusion_matrix": image})
