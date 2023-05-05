"""The package contains the code defining the collators."""
import abc
import dataclasses
import functools
import operator
from collections.abc import Callable
from typing import Any, ClassVar, Final, Generic, ParamSpec, TypeVar

import torch
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from models.backbones import BackboneConfig
from transforms.noop import noop
from utils.datasets import AvailableDatasets, convert_batch_to_dict_of_features
from utils.datasets.answer_space import AnswerSpace
from utils.registry import Registry
from utils.torch import squeeze_dict_of_tensors
from utils.types import SingleImageTransformsType

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class MultiModalCollator(Generic[T], abc.ABC):
    """The multi-modal collator."""

    tokenizer: PreTrainedTokenizer
    image_processor: BaseImageProcessor
    image_encoder_config: type[BackboneConfig]
    text_encoder_config: type[BackboneConfig]
    image_transforms: SingleImageTransformsType = dataclasses.field(default_factory=noop)

    @abc.abstractmethod
    def __call__(self, batch: list[T]) -> Any | None:
        """
        Collate the batch.

        :param batch: The batch.
        :return: The collated batch.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ClassificationCollator(MultiModalCollator, abc.ABC):
    """The classification collator."""

    answer_space: AnswerSpace = dataclasses.field(default_factory=AnswerSpace)

    @abc.abstractmethod
    def get_answer_labels(self, batch):
        """
        Get the answer labels.

        :param batch: The batch.
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_image_features(self, batch):
        """
        Get the image features.

        :param batch: The batch.
        :return: The image features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_features(self, batch):
        """
        Get the text features.

        :param batch: The batch.
        :return: The text features.
        """
        raise NotImplementedError

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


P = ParamSpec("P")
R = TypeVar("R")


class VQACollatorMixin:
    """The VQA collator mixin."""

    tokenizer: PreTrainedTokenizer
    image_processor: BaseImageProcessor
    image_encoder_config: type[BackboneConfig]
    text_encoder_config: type[BackboneConfig]
    image_transforms: SingleImageTransformsType

    answer_space: AnswerSpace

    ANSWER_BATCH_PROPERTY: ClassVar[str] = "answer"
    IMAGE_BATCH_PROPERTY: ClassVar[str] = "image"
    QUESTION_BATCH_PROPERTY: ClassVar[str] = "question"

    def get_answer_labels(self, batch):
        """
        Get the answer labels.

        :param batch: The batch.
        :return:
        """
        return {
            "answer_label": torch.tensor(
                [self.answer_space.answer_to_answer_id(answer) for answer in batch[self.ANSWER_BATCH_PROPERTY]],
                dtype=torch.int64,
            ).squeeze()
        }

    @staticmethod
    def handle_features_dim(func: Callable[[P], R]) -> Callable[[P], R]:
        """
        Handle the features dimension.

        :param func: The function.
        :return: The wrapped function.
        """
        # noqa: D202
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return squeeze_dict_of_tensors(func(*args, **kwargs))

        return wrapper

    @handle_features_dim
    def get_image_features(self, batch):
        """
        Get the image features.

        :param batch: The batch.
        :return: The image features.
        """
        return self.image_encoder_config.get_processed_image(
            self.image_processor,
            image=[self.image_transforms(image.convert("RGB")) for image in batch[self.IMAGE_BATCH_PROPERTY]],
        )

    @handle_features_dim
    def get_text_features(self, batch):
        """
        Get the text features.

        :param batch: The batch.
        :return: The text features.
        """
        return self.text_encoder_config.get_tokenized_text(
            self.tokenizer,
            text=batch[self.QUESTION_BATCH_PROPERTY],
        )


class MultiModalCollatorRegistry(Registry[AvailableDatasets, type[MultiModalCollator]]):
    """The multi-modal collator registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from collators import daquar, vqa_v2  # noqa: F401


registry: Final[MultiModalCollatorRegistry] = MultiModalCollatorRegistry()
