"""The VQA V2 collator."""
import dataclasses
import functools
import operator

import torch

from collators import ClassificationCollator, registry
from utils.datasets import AvailableDatasets, convert_batch_to_dict_of_features
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace
from utils.torch import squeeze_dict_of_tensors


@registry.register(AvailableDatasets.VQA_V2)
@registry.register(AvailableDatasets.VQA_V2_SAMPLE)
@dataclasses.dataclass(frozen=True)
class VqaV2Collator(ClassificationCollator):
    """The VQA V2 collator."""

    #: The answer space.
    answer_space: VqaV2SampleAnswerSpace = dataclasses.field(default_factory=VqaV2SampleAnswerSpace)

    def get_answer_labels(self, batch):
        """
        Get the answer labels.

        :param batch: The batch.
        :return:
        """
        return {
            "answer_label": torch.tensor(
                [self.answer_space.answer_to_answer_id(answer) for answer in batch["multiple_choice_answer"]],
                dtype=torch.int64,
            ).squeeze()
        }

    def get_image_features(self, batch):
        """
        Get the image features.

        :param batch: The batch.
        :return: The image features.
        """
        assert callable(self.image_processor), "The image processor is not callable."
        return squeeze_dict_of_tensors(
            self.image_encoder_config.get_processed_image(
                self.image_processor,
                image=[self.image_transforms(image.convert("RGB")) for image in batch["image"]],
            )
        )

    def get_text_features(self, batch):
        """
        Get the text features.

        :param batch: The batch.
        :return: The text features.
        """
        return squeeze_dict_of_tensors(
            self.text_encoder_config.get_tokenized_text(
                self.tokenizer,
                text=batch["question"],
            )
        )

    def __call__(self, batch):
        """
        Call the collator.

        :param batch: The batch.
        :return: The features and labels for each sample in the batch.
        """
        batch = convert_batch_to_dict_of_features(batch)
        features = (
            batch,  # We include the batch as a feature so
            # that we can access the image and question later.
            self.get_image_features(batch),
            self.get_text_features(batch),
            self.get_answer_labels(batch),
        )
        return functools.reduce(operator.or_, features, {})
