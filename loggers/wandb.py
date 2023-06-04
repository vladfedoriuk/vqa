"""
A module contains the logger functionalities for Weights & Biases.

For more information, see:

- https://docs.wandb.ai/guides/integrations/lightning
- https://lightning.ai/docs/pytorch/latest/extensions/generated/pytorch_lightning.loggers.WandbLogger.html

For some best practices, see:
https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1

"""  # noqa: E501
from functools import lru_cache
from typing import Literal

import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger

from config.env import WANDB_PROJECT
from lightning_modules.encoder_decoder.configurable import (
    ConfigurableEncoderDecoderModule,
)
from lightning_modules.vilt.masked_language_modeling import (
    ViLTMaskedLanguageModelingModule,
)
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace
from utils.types import BatchType


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


def log_confusion_matrix(logger: WandbLogger, cm: torch.Tensor, key: str):
    """
    Log the confusion matrix to Weights & Biases.

    :param logger: The logger.
    :param cm: The confusion matrix (as obtained from :py:func:`torchmetrics.functional.confusion_matrix`).
    :param key: The key.
    :return:
    """
    cm = cm.cpu().numpy().squeeze()
    logger.experiment.log(
        {
            key: wandb.Image(cm, caption="Confusion matrix"),
        }
    )


def log_vqa_v2_predictions_as_table(
    logger: WandbLogger,
    answer_space: VqaV2SampleAnswerSpace,
    batch: BatchType,
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
    batch: BatchType,
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


def log_vilt_mlm_predictions_as_table(
    logger: WandbLogger,
    batch: BatchType,
    pl_module: ViLTMaskedLanguageModelingModule,
    answer_batch_property: str = "answer",
    image_batch_property: str = "image",
    original_question_batch_property: str = "original_question",
):  # TODO: Refactor the logging functions to reuse some table logging code.
    """
    Log the Daquar predictions as a table to Weights & Biases.

    :param logger: The logger.
    :param batch: The batch.
    :param pl_module: The module.
    :param answer_batch_property: The answer batch property.
    :param image_batch_property: The image batch property.
    :param original_question_batch_property: The original question batch property.
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
            wandb.Image(data_point[image_batch_property]),
            data_point[original_question_batch_property],
            data_point[answer_batch_property],
            predicted_answer,
        ]
        for data_point, predicted_answer in zip(batch, pl_module.make_masked_answer_prediction(batch))
    ]
    logger.log_table(
        key="sample_predictions",
        columns=columns,
        data=data,
    )


def log_causal_language_modeling_predictions_as_table(
    logger: WandbLogger,
    batch: BatchType,
    pl_module: ConfigurableEncoderDecoderModule,
    answer_batch_property: str = "answer",
    image_batch_property: str = "image",
    original_question_batch_property: str = "original_question",
):
    """
    Log the Daquar predictions as a table to Weights & Biases.

    :param logger: The logger.
    :param batch: The batch.
    :param pl_module: The module.
    :param answer_batch_property: The answer batch property.
    :param image_batch_property: The image batch property.
    :param original_question_batch_property: The original question batch property.
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
            wandb.Image(data_point[image_batch_property]),
            data_point[original_question_batch_property],
            data_point[answer_batch_property],
            predicted_answer,
        ]
        for data_point, predicted_answer in zip(batch, pl_module.make_generated_answer_prediction(batch))
    ]
    logger.log_table(
        key="sample_predictions",
        columns=columns,
        data=data,
    )
