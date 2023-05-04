"""
The classification Lightning modules callbacks.

To know more about the Lightning modules callbacks,
check out the documentation:

https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
"""
from typing import cast

import torch
import wandb
from lightning import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.datasets import convert_batch_to_list_of_dicts
from utils.datasets.answer_space import AnswerSpace


class LogClassificationPredictionSamplesCallback(Callback):
    """
    The callback to log the classification prediction samples fom the validation set.

    The callback is meant to be used with
    :class:`lightningmodules.classification.MultiModalClassificationModule`.

    The callback logs the prediction samples to Weights & Biases.
    """

    def __init__(self, answer_space: AnswerSpace, num_samples: int = 20):
        """
        Initialize the callback.

        :param answer_space: The answer space to use.
        :param num_samples: The number of samples to log.
        """
        super().__init__()
        self.answer_space = answer_space
        self.num_samples = num_samples

    def get_predicted_answer(self, output_logits: torch.Tensor):
        """
        Get the predicted answer for the given datapoint and model output logits.

        :param output_logits: The output logits.
        :return: The caption.
        """
        # Get the model prediction
        predicted_answer_id = self.answer_space.logits_to_answer_id(output_logits)
        # Retrieve the answer from the answer space
        return self.answer_space.answer_id_to_answer(predicted_answer_id)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Call when the validation batch ends.

        :param trainer: The trainer.
        :param pl_module: The Lightning module.
        :param outputs: The outputs.
        :param batch: The batch.
        :param batch_idx: The batch index.
        :param dataloader_idx: The dataloader index.
        """
        batch = convert_batch_to_list_of_dicts(batch)
        random_samples = torch.randperm(len(batch))[: self.num_samples]
        batch = [batch[idx] for idx in random_samples]
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
                self.get_predicted_answer(output_logits),
            ]
            for data_point, output_logits in zip(batch, outputs["logits"])
        ]
        logger = cast(WandbLogger, pl_module.logger)
        logger.log_table(
            key="sample_predictions",
            columns=columns,
            data=data,
        )
