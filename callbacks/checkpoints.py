"""
The ModelCheckpoint callback.

To know more about the ModelCheckpoint callback,
check out the documentation:

https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html
"""
from lightning.pytorch.callbacks import ModelCheckpoint

from config.env import SAVE_ARTIFACTS_PATH
from models.backbones import AvailableBackbones
from models.fusions import AvailableFusionModels
from utils.datasets import AvailableDatasets
from utils.torch import backbone_name_to_kebab_case


def get_model_checkpoint(
    image_encoder: AvailableBackbones,
    text_encoder: AvailableBackbones,
    fusion: AvailableFusionModels,
    dataset: AvailableDatasets,
) -> ModelCheckpoint:
    """
    Get the model checkpoint.

    :param image_encoder: The name of the backbone image model
    :param text_encoder: The name of the backbone text model
    :param fusion: The name of the fusion model
    :param dataset: The name of the dataset
    :return: The model checkpoint.
    """
    return ModelCheckpoint(
        dirpath=SAVE_ARTIFACTS_PATH,
        monitor="val_loss",
        mode="min",
        filename=(
            f"{backbone_name_to_kebab_case(image_encoder.value)}-"
            f"{backbone_name_to_kebab_case(text_encoder.value)}-"
            f"{fusion.value}-"
            f"{dataset.value}-"
            "{epoch:02d}-{val_loss:.2f}"
        ),
        save_top_k=1,
        every_n_epochs=1,
    )
