"""The VQA V2 collator."""
import dataclasses
from typing import ClassVar

from transformers import DataCollatorForLanguageModeling

from collators import (
    ClassificationCollator,
    MultiModalCollator,
    VQAClassificationCollatorMixin,
    VQAMultimodalCollatorMixin,
    registry,
)
from utils.batch import (
    convert_batch_to_mapping_of_features,
    convert_batch_to_sequence_of_mappings,
)
from utils.datasets import AvailableDatasets
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace
from utils.torch import expand_first_dim_dict_of_tensors, squeeze_dict_of_tensors
from utils.types import BatchType


@registry.register(AvailableDatasets.VQA_V2)
@registry.register(AvailableDatasets.VQA_V2_SAMPLE)
@dataclasses.dataclass
class VqaV2ClassificationCollator(VQAClassificationCollatorMixin, ClassificationCollator):
    """The VQA V2 collator."""

    #: The answer space.
    answer_space: VqaV2SampleAnswerSpace = dataclasses.field(default_factory=VqaV2SampleAnswerSpace)

    ANSWER_BATCH_PROPERTY = "multiple_choice_answer"


@dataclasses.dataclass
class VqaV2DataCollatorForLanguageModeling(
    VQAMultimodalCollatorMixin,
    MultiModalCollator,
):
    """The VQA V2 data collator for language modeling."""

    ANSWER_BATCH_PROPERTY: ClassVar[str] = "multiple_choice_answer"
    ORIGINAL_QUESTION_BATCH_PROPERTY: ClassVar[str] = "original_question"
    QUESTION_ANSWER_PROMPT_BATCH_PROPERTY: ClassVar[str] = VQAMultimodalCollatorMixin.QUESTION_BATCH_PROPERTY

    data_collator_for_language_modeling: DataCollatorForLanguageModeling = dataclasses.field(init=False)

    def __post_init__(self):
        """Initialize the data collator for language modeling."""
        self.data_collator_for_language_modeling = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, return_tensors="pt"
        )

    def _backup_question(self, batch: BatchType) -> BatchType:
        """
        Backup the question.

        :param batch: The batch.
        :return: The batch with the backed up question.
        """
        batch[self.ORIGINAL_QUESTION_BATCH_PROPERTY] = batch[self.QUESTION_BATCH_PROPERTY].copy()
        return batch

    def _create_question_answer_prompt(self, batch: BatchType) -> BatchType:
        """
        Create the question/answer prompt.

        :param batch: The batch.
        :return: The batch with the question/answer prompt.
        """
        # Create the question/answer prompt
        batch[self.QUESTION_ANSWER_PROMPT_BATCH_PROPERTY] = [
            f"answer the following question: {question} answer: {answer}"
            for question, answer in zip(batch[self.QUESTION_BATCH_PROPERTY], batch[self.ANSWER_BATCH_PROPERTY])
        ]
        return batch

    def __call__(self, batch: BatchType) -> BatchType:
        """
        Collate the batch.

        :param batch: The batch.
        :return: The collated batch.
        """
        # Convert the batch to a dict of features
        batch = convert_batch_to_mapping_of_features(batch)
        # Backup the question
        batch = self._backup_question(batch)
        # Create the question/answer prompt
        batch = self._create_question_answer_prompt(batch)
        # Do the pre-processing stuff with tokenization and image processing
        batch = super().__call__(batch)
        # Add the masked tokens, labels and token masks
        features = self.data_collator_for_language_modeling(
            convert_batch_to_sequence_of_mappings(
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


@dataclasses.dataclass
class VqaV2DataCollatorForMaskedLanguageModeling(VqaV2DataCollatorForLanguageModeling):
    """The Daquar data collator for masked language modeling."""

    def __post_init__(self):
        """Initialize the data collator for language modeling."""
        self.data_collator_for_language_modeling = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
        )
