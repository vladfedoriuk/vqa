"""The package contains the code defining the collators."""
import dataclasses
from typing import Final

from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from models.backbones import BackboneConfig
from transforms.noop import noop
from utils.datasets import AnswerSpace
from utils.registry import Registry, RegistryKey
from utils.types import SingleImageTransformsType, TransformsType


@dataclasses.dataclass(frozen=True)
class MultiModalCollator:
    """The multi-modal collator."""

    tokenizer: PreTrainedTokenizer
    image_processor: BaseImageProcessor
    image_encoder_config: type[BackboneConfig]
    text_encoder_config: type[BackboneConfig]
    image_transforms: SingleImageTransformsType = dataclasses.field(default_factory=noop)


@dataclasses.dataclass(frozen=True)
class ClassificationCollator(MultiModalCollator):
    """The classification collator."""

    answer_space: AnswerSpace = dataclasses.field(default_factory=AnswerSpace)

    @classmethod
    def get_dataloaders(
        cls,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor,
        image_encoder_config: type[BackboneConfig],
        text_encoder_config: type[BackboneConfig],
        image_transforms: TransformsType,
        answer_space: AnswerSpace,
        batch_size: int = 64,
    ):
        """
        Get the data loaders.

        :param tokenizer: The tokenizer.
        :param image_processor: The preprocessor.
        :param image_encoder_config: The image encoder config.
        :param text_encoder_config: The text encoder config.
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


class MultiModalCollatorRegistry(Registry[AvailableCollators, type[MultiModalCollator]]):
    """The multi-modal collator registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from collators import vqa_v2  # noqa: F401


registry: Final[Registry] = MultiModalCollatorRegistry()
