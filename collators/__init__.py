"""The package contains the code defining the collators."""
import dataclasses
from typing import Final

from transformers import AutoImageProcessor, AutoTokenizer

from transforms.noop import noop
from utils.datasets import AnswerSpace
from utils.registry import Registry, RegistryKey
from utils.types import SingleImageTransformsType, TransformsType


@dataclasses.dataclass(frozen=True)
class MultiModalCollator:
    """The multi-modal collator."""

    tokenizer: AutoTokenizer
    image_processor: AutoImageProcessor
    image_transforms: SingleImageTransformsType = dataclasses.field(
        default_factory=noop
    )


@dataclasses.dataclass(frozen=True)
class ClassificationCollator(MultiModalCollator):
    """The classification collator."""

    answer_space: AnswerSpace = dataclasses.field(default_factory=AnswerSpace)

    @classmethod
    def get_dataloaders(
        cls,
        tokenizer: AutoTokenizer,
        image_processor: AutoImageProcessor,
        image_transforms: TransformsType,
        answer_space: AnswerSpace,
        batch_size: int = 64,
    ):
        """
        Get the data loaders.

        :param tokenizer: The tokenizer.
        :param image_processor: The preprocessor.
        :param image_transforms: The image transforms
                                    to apply to the images
                                    (per phase).
        :param answer_space: The answer space.
        :param batch_size: The batch size.
        :return: The data loaders.
        """
        raise NotImplementedError


class AvailableCollators(RegistryKey):
    """The available collators."""

    VQA_V2_SAMPLE = "vqa_v2_sample"


class MultiModalCollatorRegistry(
    Registry[AvailableCollators, type[MultiModalCollator]]
):
    """The multi-modal collator registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from collators import vqa_v2  # noqa: F401


registry: Final[Registry] = MultiModalCollatorRegistry()
