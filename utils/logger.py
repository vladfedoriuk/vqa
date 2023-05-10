"""The logger utilities."""

from models.backbones import AvailableBackbones
from models.fusions import AvailableFusionModels
from utils.datasets import AvailableDatasets


def compose_run_name(
    fusion: AvailableFusionModels,
    dataset: AvailableDatasets,
    freeze_image_encoder_backbone: bool,
    freeze_text_encoder_backbone: bool,
    freeze_multimodal_backbone: bool,
    image_encoder_backbone: AvailableBackbones | None = None,
    text_encoder_backbone: AvailableBackbones | None = None,
    multimodal_backbone: AvailableBackbones | None = None,
    epochs: int = 10,
    batch_size: int = 64,
):
    """
    Compose the run name.

    :param fusion: The name of the fusion model to use.
    :param dataset: The name of the dataset to use.
    :param freeze_image_encoder_backbone: Whether to freeze the image encoder backbone.
    :param freeze_text_encoder_backbone: Whether to freeze the text encoder backbone.
    :param freeze_multimodal_backbone: Whether to freeze the multimodal backbone.
    :param image_encoder_backbone: The name of the backbone model
                                    to use for the image encoder.
    :param text_encoder_backbone: The name of the backbone model
                                    to use for the text encoder.
    :param multimodal_backbone: The name of the multimodal backbone model.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :return: The run name.
    """
    run_name = f"{fusion.value}_{dataset.value}_"
    if image_encoder_backbone is not None:
        run_name += f"image_encoder_backbone_{image_encoder_backbone.value}_"
        if freeze_image_encoder_backbone:
            run_name += "frozen_"
    if text_encoder_backbone is not None:
        run_name += f"text_encoder_backbone_{text_encoder_backbone.value}_"
        if freeze_text_encoder_backbone:
            run_name += "frozen_"
    if multimodal_backbone is not None:
        run_name += f"multimodal_backbone_{multimodal_backbone.value}_"
        if freeze_multimodal_backbone:
            run_name += "frozen_"
    run_name += f"epochs_{epochs}_batch_size_{batch_size}"
    return run_name
