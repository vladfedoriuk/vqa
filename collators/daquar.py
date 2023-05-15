"""The Daquar collators."""
import dataclasses
from typing import ClassVar

from transformers import DataCollatorForLanguageModeling
from transformers.image_processing_utils import BaseImageProcessor

from collators import (
    ClassificationCollator,
    MultiModalCollator,
    VQAClassificationCollatorMixin,
    VQAMultimodalCollatorMixin,
    registry,
)
from models.backbones import BackboneConfig
from utils.batch import (
    convert_batch_to_dict_of_features,
    convert_batch_to_list_of_dicts,
)
from utils.datasets import AvailableDatasets
from utils.datasets.daquar import DaquarAnswerSpace, load_images_for_batch
from utils.torch import expand_first_dim_dict_of_tensors, squeeze_dict_of_tensors
from utils.types import BatchImageTransformsType, BatchType, SingleImageTransformsType


class DaquarImageFeaturesCollatorMixin:
    """
    The Daquar image features collator mixin.

    This mixin is used to get the image features.
    """

    image_processor: BaseImageProcessor
    image_encoder_config: type[BackboneConfig]
    single_image_transforms: SingleImageTransformsType
    batch_image_transforms: BatchImageTransformsType

    IMAGE_BATCH_PROPERTY: ClassVar[str] = "image"

    def get_image_features(self, batch: BatchType) -> BatchType:
        """
        Get the image features.

        :param batch: The batch.
        :return: The image features.
        """
        images = load_images_for_batch(batch)
        features = self.image_encoder_config.get_processed_image(
            self.image_processor,
            image=[self.single_image_transforms(image) for image in images],
        )
        features[self.IMAGE_BATCH_PROPERTY] = images
        features = self.batch_image_transforms(features)
        return features


@registry.register(AvailableDatasets.DAQUAR)
@dataclasses.dataclass
class DaquarClassificationCollator(
    DaquarImageFeaturesCollatorMixin, VQAClassificationCollatorMixin, ClassificationCollator
):
    """The Daquar collator."""

    #: The answer space.
    answer_space: DaquarAnswerSpace = dataclasses.field(default_factory=DaquarAnswerSpace)


@dataclasses.dataclass
class DaquarDataCollatorForLanguageModeling(
    DaquarImageFeaturesCollatorMixin,
    VQAMultimodalCollatorMixin,
    MultiModalCollator,
):
    """The Daquar data collator for language modeling."""

    ANSWER_BATCH_PROPERTY: ClassVar[str] = "answer"

    data_collator_for_language_modeling: DataCollatorForLanguageModeling = dataclasses.field(init=False)

    def __post_init__(self):
        """Initialize the data collator for language modeling."""
        self.data_collator_for_mlm = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
        )

    def _create_question_answer_prompt(self, batch: BatchType) -> BatchType:
        """
        Create the question/answer prompt.

        :param batch: The batch.
        :return: The batch with the question/answer prompt.
        """
        # Create the question/answer prompt
        batch[self.QUESTION_BATCH_PROPERTY] = [
            f"question: {question} answer: {answer}"
            for question, answer in zip(batch[self.QUESTION_BATCH_PROPERTY], batch[self.ANSWER_BATCH_PROPERTY])
        ]
        return batch

    def __call__(self, batch: BatchType) -> BatchType:
        """
        Collate the batch.

        :param batch: The batch.
        :return: The collated batch.
        """
        batch = convert_batch_to_dict_of_features(batch)
        # Create the question/answer prompt
        batch = self._create_question_answer_prompt(batch)
        # Do the pre-processing stuff with tokenization and image processing
        batch = super().__call__(batch)
        # Add the masked tokens, labels and token masks
        features = self.data_collator_for_mlm(
            convert_batch_to_list_of_dicts(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "special_tokens_mask": batch["special_tokens_mask"],
                    "token_type_ids": batch["token_type_ids"],
                }
            )
        )
        batch |= features
        batch = squeeze_dict_of_tensors(batch)
        if len(batch[self.QUESTION_BATCH_PROPERTY]) == 1:
            return expand_first_dim_dict_of_tensors(batch)
        return batch
