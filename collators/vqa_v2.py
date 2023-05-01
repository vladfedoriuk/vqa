"""The VQA V2 collator."""
import dataclasses
import functools
import operator

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from collators import AvailableCollators, ClassificationCollator, registry
from utils.datasets import convert_batch_to_dict_of_features
from utils.datasets.vqa_v2 import (
    VqaV2SampleAnswerSpace,
    load_vqa_v2_sample_test_dataset,
    load_vqa_v2_sample_train_dataset,
    load_vqa_v2_sample_val_dataset,
)
from utils.phase import Phase
from utils.torch import squeeze_dict_of_tensors
from utils.types import TransformsType


@registry.register(AvailableCollators.VQA_V2_SAMPLE)
@dataclasses.dataclass(frozen=True)
class VqaV2SampleCollator(ClassificationCollator):
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
            self.image_processor(
                images=[self.image_transforms(image.convert("RGB")) for image in batch["image"]],
                return_tensors="pt",
            )
        )

    def get_text_features(self, batch):
        """
        Get the text features.

        :param batch: The batch.
        :return: The text features.
        """
        return squeeze_dict_of_tensors(
            self.tokenizer(
                text=batch["question"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids="token_type_ids" in self.tokenizer.model_input_names,
                return_attention_mask=True,
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

    @classmethod
    def get_dataloaders(
        cls,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor,
        image_transforms: TransformsType,
        answer_space: VqaV2SampleAnswerSpace,
        batch_size: int = 64,
    ):
        """
        Get the data loaders.

        :param tokenizer: The tokenizer.
        :param image_processor: The preprocessor.
        :param image_transforms: The image transforms to apply to the images.
        :param batch_size: The batch size.
        :param answer_space: The answer space.
        :return: The data loaders.
        """
        train_data, val_data, test_data = (
            load_vqa_v2_sample_train_dataset(),
            load_vqa_v2_sample_val_dataset(),
            load_vqa_v2_sample_test_dataset(),
        )

        dataloader_train = DataLoader(
            train_data,
            sampler=RandomSampler(train_data),
            collate_fn=cls(
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_transforms=image_transforms[Phase.TRAIN],
                answer_space=answer_space,
            ),
            batch_size=batch_size,
        )

        dataloader_validation = DataLoader(
            val_data,
            sampler=SequentialSampler(val_data),
            collate_fn=cls(
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_transforms=image_transforms[Phase.EVAL],
                answer_space=answer_space,
            ),
            batch_size=batch_size,
        )

        dataloader_test = DataLoader(
            test_data,
            sampler=SequentialSampler(test_data),
            collate_fn=cls(
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_transforms=image_transforms[Phase.TEST],
                answer_space=answer_space,
            ),
            batch_size=batch_size,
        )
        return {
            Phase.TRAIN: dataloader_train,
            Phase.EVAL: dataloader_validation,
            Phase.TEST: dataloader_test,
        }
