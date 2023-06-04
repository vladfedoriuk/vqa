"""The package contains the code defining the collators."""
import abc
import dataclasses
import functools
import operator
from typing import ClassVar, Final, Generic, ParamSpec, TypeVar, cast

import PIL.Image
import torch
from torchvision.transforms import transforms
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from models.backbones import BackboneConfig
from transforms.noop import noop
from utils.batch import convert_batch_to_mapping_of_features
from utils.datasets import AvailableDatasets
from utils.datasets.answer_space import AnswerSpace
from utils.registry import Registry
from utils.torch import expand_first_dim_dict_of_tensors, squeeze_dict_of_tensors
from utils.types import (
    BatchImageTransformsType,
    BatchTextTransformsType,
    BatchType,
    SingeTextTransformsType,
    SingleImageTransformsType,
)

T = TypeVar("T")


@dataclasses.dataclass
class MultiModalCollator(Generic[T], abc.ABC):
    """The multi-modal collator."""

    tokenizer: PreTrainedTokenizer
    image_processor: BaseImageProcessor
    image_encoder_config: type[BackboneConfig]
    text_encoder_config: type[BackboneConfig]
    single_image_transforms: SingleImageTransformsType = dataclasses.field(default=noop)
    single_text_transforms: SingeTextTransformsType = dataclasses.field(default=noop)
    batch_image_transforms: BatchImageTransformsType = dataclasses.field(default=noop)
    batch_text_transforms: BatchTextTransformsType = dataclasses.field(default=noop)

    @abc.abstractmethod
    def get_image_features(self, batch: BatchType) -> BatchType:
        """
        Get the image features.

        :param batch: The batch.
        :return: The image features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_features(self, batch: BatchType) -> BatchType:
        """
        Get the text features.

        :param batch: The batch.
        :return: The text features.
        """
        raise NotImplementedError

    def __call__(self, batch: BatchType) -> BatchType:
        """
        Collate the batch.

        :param batch: The batch.
        :return: The collated batch.
        """
        batch = convert_batch_to_mapping_of_features(batch)
        # We include the batch as a feature so
        # that we can access the image and question later.
        return functools.reduce(
            operator.or_,
            (
                self.get_image_features(batch),
                self.get_text_features(batch),
            ),
            batch,
        )


@dataclasses.dataclass
class ClassificationCollator(MultiModalCollator, abc.ABC):
    """The classification collator."""

    answer_space: AnswerSpace = dataclasses.field(default_factory=AnswerSpace)

    @abc.abstractmethod
    def get_answer_labels(self, batch: BatchType) -> BatchType:
        """
        Get the answer labels.

        :param batch: The batch.
        :return:
        """
        raise NotImplementedError

    def __call__(self, batch: BatchType) -> BatchType:
        """
        Call the collator.

        :param batch: The batch.
        :return: The features and labels for each sample in the batch.
        """
        batch = super().__call__(batch)
        return batch | self.get_answer_labels(batch)


P = ParamSpec("P")
R = TypeVar("R")


class VQAMultimodalCollatorMixin:
    """The VQA collator mixin."""

    tokenizer: PreTrainedTokenizer
    image_processor: BaseImageProcessor
    image_encoder_config: type[BackboneConfig]
    text_encoder_config: type[BackboneConfig]
    single_image_transforms: SingleImageTransformsType
    single_text_transforms: SingeTextTransformsType
    batch_image_transforms: BatchImageTransformsType
    batch_text_transforms: BatchTextTransformsType

    IMAGE_BATCH_PROPERTY: ClassVar[str] = "image"
    QUESTION_BATCH_PROPERTY: ClassVar[str] = "question"

    @staticmethod
    def _ensure_image_is_pil(image: PIL.Image.Image | torch.Tensor) -> PIL.Image.Image:
        """
        Ensure the image is PIL.

        :param image: The image.
        :return: The image.
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.cpu())
        return image

    @staticmethod
    def _ensure_image_is_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Ensure the image is RGB.

        :param image: The image.
        :return: The image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def get_image_features(self, batch: BatchType) -> BatchType:
        """
        Get the image features.

        :param batch: The batch.
        :return: The image features.
        """
        return self.batch_image_transforms(
            self.image_encoder_config.get_processed_image(
                self.image_processor,
                image=[
                    self.single_image_transforms(self._ensure_image_is_rgb(self._ensure_image_is_pil(image)))
                    for image in batch[self.IMAGE_BATCH_PROPERTY]
                ],
            )
        )

    def get_text_features(self, batch: BatchType) -> BatchType:
        """
        Get the text features.

        :param batch: The batch.
        :return: The text features.
        """
        batch[self.QUESTION_BATCH_PROPERTY] = [
            self.single_text_transforms(question) for question in batch[self.QUESTION_BATCH_PROPERTY]
        ]
        batch = self.batch_text_transforms(batch)
        return self.text_encoder_config.get_tokenized_text(self.tokenizer, text=batch[self.QUESTION_BATCH_PROPERTY])

    def __call__(self, batch: BatchType) -> BatchType:
        """
        Call the collator.

        :param batch: The batch.
        :return: The collated batch.
        """
        batch = cast(MultiModalCollator, super()).__call__(batch)
        batch = squeeze_dict_of_tensors(batch)
        if len(batch[self.IMAGE_BATCH_PROPERTY]) != len(batch[self.QUESTION_BATCH_PROPERTY]):
            raise ValueError(
                f"The number of images ({len(batch[self.IMAGE_BATCH_PROPERTY])}) and questions "
                f"({len(batch[self.QUESTION_BATCH_PROPERTY])}) must be equal."
            )
        if len(batch[self.QUESTION_BATCH_PROPERTY]) == 1:
            return expand_first_dim_dict_of_tensors(batch)
        return batch


class VQAClassificationCollatorMixin(VQAMultimodalCollatorMixin):
    """The VQA classification collator mixin."""

    answer_space: AnswerSpace

    ANSWER_BATCH_PROPERTY: ClassVar[str] = "answer"

    def get_answer_labels(self, batch: BatchType) -> BatchType:
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


class MultiModalCollatorRegistry(Registry[AvailableDatasets, type[MultiModalCollator]]):
    # TODO: Change the value type to (type, role) - role: classification, masked_lm, casual_lm, ...
    """The multi-modal collator registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from collators import daquar, vqa_v2  # noqa: F401


registry: Final[MultiModalCollatorRegistry] = MultiModalCollatorRegistry()
