"""The classification Lightning modules callbacks."""
from typing import Any

import torch
from lightning import Callback
from pytorch_lightning.loggers import WandbLogger

from utils.datasets import AnswerSpace


class LogClassificationPredictionSamplesCallback(Callback):
    """
    The callback to log the classification prediction samples fom the validation set.

    The callback is meant to be used with
    :class:`lightningmodules.classification.MultiModalClassificationModule`.

    The callback logs the prediction samples to Weights & Biases.
    """

    def __init__(
        self, logger: WandbLogger, answer_space: AnswerSpace, num_samples: int = 20
    ):
        """
        Initialize the callback.

        :param logger: The logger to use.
        :param answer_space: The answer space to use.
        :param num_samples: The number of samples to log.
        """
        super().__init__()
        self.logger = logger
        self.answer_space = answer_space
        self.num_samples = num_samples

    def get_caption(self, data_point: dict[str, Any], output_logits: torch.Tensor):
        """
        Get the caption for the image prediction.

        :param data_point: The data point.
        :param output_logits: The output logits.
        :return: The caption.
        """
        # Get the model prediction
        predicted_answer_id = self.answer_space.logits_to_answer_id(output_logits)
        predicted_answer = self.answer_space.answer_id_to_answer(predicted_answer_id)
        return "\n".join(
            (
                f"Question: {data_point['question']}",
                f"Multiple Choice Answer: {data_point['multiple_choice_answer']}",
                "Possible alternative answers:",
                "\n".join(
                    {
                        answer if isinstance(answer, str) else answer["answer"]
                        for answer in data_point["answers"]
                        if answer["answer"] != data_point["multiple_choice_answer"]
                    }
                ),
                f"Model prediction: {predicted_answer}",
            )
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Call when the validation batch ends."""
        if batch_idx == 0:
            images = [img["image"] for img in batch[: self.num_samples]]
            captions = [
                self.get_caption(data_point, output_logits)
                for data_point, output_logits in zip(
                    batch[: self.num_samples], outputs["logits"][: self.num_samples]
                )
            ]
            self.logger.log_image(key="sample_images", images=images, caption=captions)
