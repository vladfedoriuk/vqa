"""The logger utilities."""
from typing import Literal

from models.backbones import AvailableBackbones
from models.fusions import AvailableFusionModels
from utils.datasets import AvailableDatasets
from utils.torch import backbone_name_to_kebab_case


def compose_fusion_classification_experiment_run_name(
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
    Compose the run name for a fusion classification experiment.

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
    run_name = f"{fusion.value}-classification-{dataset.value}-"
    if image_encoder_backbone is not None:
        run_name += f"{backbone_name_to_kebab_case(image_encoder_backbone)}-"
        if freeze_image_encoder_backbone:
            run_name += "frozen-"
    if text_encoder_backbone is not None:
        run_name += f"{backbone_name_to_kebab_case(text_encoder_backbone)}-"
        if freeze_text_encoder_backbone:
            run_name += "frozen-"
    if multimodal_backbone is not None:
        run_name += f"{backbone_name_to_kebab_case(multimodal_backbone)}-"
        if freeze_multimodal_backbone:
            run_name += "frozen-"
    run_name += f"epochs-{epochs}-batch-size-{batch_size}"
    return run_name


def compose_vilt_experiment_run_name(
    vilt_backbone: AvailableBackbones,
    dataset: AvailableDatasets,
    epochs: int = 10,
    batch_size: int = 64,
    type_: Literal["masked-language-modeling", "classification"] = "classification",
):
    """
    Compose the run name for a ViLT classification experiment.

    :param vilt_backbone: The ViLT backbone to use.
    :param dataset: The name of the dataset to use.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :param type_: The type of experiment to run.
    :return: The run name.
    """
    return "-".join(
        (
            f"{backbone_name_to_kebab_case(vilt_backbone)}",
            f"{type_}-{dataset.value}",
            f"epochs-{epochs}",
            f"batch-size-{batch_size}",
        )
    )
