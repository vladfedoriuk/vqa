"""
A module contains the logger functionalities for Weights & Biases.

For more information, see:

- https://docs.wandb.ai/guides/integrations/lightning
- https://lightning.ai/docs/pytorch/latest/extensions/generated/pytorch_lightning.loggers.WandbLogger.html
"""  # noqa: E501

from lightning.pytorch.loggers.wandb import WandbLogger

from config.env import WANDB_PROJECT


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
