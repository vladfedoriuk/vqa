"""
The ModelCheckpoint callback.

To know more about the ModelCheckpoint callback,
check out the documentation:

https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html
"""
from lightning.pytorch.callbacks import ModelCheckpoint

from config.env import SAVE_ARTIFACTS_PATH


def get_model_checkpoint(
    file_name: str,
) -> ModelCheckpoint:
    """
    Get the model checkpoint.

    :param file_name: The name of the file to save the model checkpoint to.
    :return: The model checkpoint.
    """
    return ModelCheckpoint(
        dirpath=SAVE_ARTIFACTS_PATH,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        filename=file_name,
    )
