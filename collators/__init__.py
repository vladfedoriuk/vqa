"""The package contains the code defining the collators."""
import dataclasses
from typing import Any, Final, Generic, TypeVar

from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from models.backbones import BackboneConfig
from transforms.noop import noop
from utils.datasets import AvailableDatasets
from utils.datasets.answer_space import AnswerSpace
from utils.registry import Registry
from utils.types import SingleImageTransformsType

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class MultiModalCollator(Generic[T]):
    """The multi-modal collator."""

    tokenizer: PreTrainedTokenizer
    image_processor: BaseImageProcessor
    image_encoder_config: type[BackboneConfig]
    text_encoder_config: type[BackboneConfig]
    image_transforms: SingleImageTransformsType = dataclasses.field(default_factory=noop)

    def __call__(self, batch: list[T]) -> Any | None:
        """
        Collate the batch.

        :param batch: The batch.
        :return: The collated batch.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ClassificationCollator(MultiModalCollator):
    """The classification collator."""

    answer_space: AnswerSpace = dataclasses.field(default_factory=AnswerSpace)


class MultiModalCollatorRegistry(Registry[AvailableDatasets, type[MultiModalCollator]]):
    """The multi-modal collator registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from collators import vqa_v2  # noqa: F401


registry: Final[MultiModalCollatorRegistry] = MultiModalCollatorRegistry()
