"""
A module contains the logger functionalities for Weights & Biases.

For more information, see:

- https://docs.wandb.ai/guides/integrations/lightning
- https://lightning.ai/docs/pytorch/latest/extensions/generated/pytorch_lightning.loggers.WandbLogger.html

For some best practices, see:
https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1

"""  # noqa: E501
from functools import lru_cache
from typing import Any, Literal

import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb.sdk.wandb_run import Run

from collators.daquar import DaquarDataCollatorForLanguageModeling
from config.env import WANDB_PROJECT
from lightning_modules.vilt.masked_language_modeling import (
    ViLTMaskedLanguageModelingModule,
)
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace


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
        log_model=False,
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


def log_vqa_v2_predictions_as_table(
    logger: WandbLogger,
    answer_space: VqaV2SampleAnswerSpace,
    batch: list[dict[str, Any]],
    outputs: dict[Literal["logits"], torch.Tensor],
) -> None:
    """
    Log the VQA v2 predictions as a table to Weights & Biases.

    :param logger: The logger.
    :param answer_space: The answer space.
    :param batch: The batch.
    :param outputs: The outputs.
    :return: None.
    """
    from utils.datasets.answer_space import get_predicted_answer

    columns = [
        "Image",
        "Question",
        "Multiple Choice Answer",
        "Possible alternative answers",
        "Model prediction",
    ]
    data = [
        [
            wandb.Image(data_point["image"]),
            data_point["question"],
            data_point["multiple_choice_answer"],
            ", ".join(answer if isinstance(answer, str) else answer["answer"] for answer in data_point["answers"]),
            get_predicted_answer(answer_space=answer_space, output_logits=output_logits),
        ]
        for data_point, output_logits in zip(batch, outputs["logits"])
    ]
    logger.log_table(
        key="sample_predictions",
        columns=columns,
        data=data,
    )


def log_daquar_predictions_as_table(
    logger: WandbLogger,
    answer_space: VqaV2SampleAnswerSpace,
    batch: list[dict[str, Any]],
    outputs: dict[Literal["logits"], torch.Tensor],
):
    """
    Log the Daquar predictions as a table to Weights & Biases.

    :param logger: The logger.
    :param answer_space: The answer space.
    :param batch: The batch.
    :param outputs: The outputs.
    :return: None.
    """
    from utils.datasets.answer_space import get_predicted_answer

    columns = [
        "Image",
        "Question",
        "Answer",
        "Model prediction",
    ]
    data = [
        [
            wandb.Image(data_point["image"]),
            data_point["question"],
            data_point["answer"],
            get_predicted_answer(answer_space=answer_space, output_logits=output_logits),
        ]
        for data_point, output_logits in zip(batch, outputs["logits"])
    ]
    logger.log_table(
        key="sample_predictions",
        columns=columns,
        data=data,
    )


def log_daquar_vilt_mlm_predictions_as_table(
    logger: WandbLogger,
    batch: list[dict[str, Any]],
    pl_module: ViLTMaskedLanguageModelingModule,
):
    """
    Log the Daquar predictions as a table to Weights & Biases.

    :param logger: The logger.
    :param batch: The batch.
    :param pl_module: The module.
    :return: None.
    """
    columns = [
        "Image",
        "Question",
        "Answer",
        "Model prediction",
    ]
    data = [
        [
            wandb.Image(data_point["image"]),
            data_point[DaquarDataCollatorForLanguageModeling.ORIGINAL_QUESTION_BATCH_PROPERTY],
            data_point["answer"],
            predicted_answer,
        ]
        for data_point, predicted_answer in zip(batch, pl_module.make_masked_answer_prediction(batch))
    ]
    logger.log_table(
        key="sample_predictions",
        columns=columns,
        data=data,
    )
