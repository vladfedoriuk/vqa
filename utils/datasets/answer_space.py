"""
The answer spaces module.

The answer spaces are used to convert the answers to answer ids
and vice versa.
"""
from typing import ClassVar, Final, Protocol

import torch

from utils.datasets import AvailableDatasets
from utils.registry import Registry


class AnswerSpace(Protocol):
    """
    The answer space protocol.

    It can be used to convert the answers to answer ids
    and vice versa.

    The protocol is used for type hinting.
    """

    #: The fake answer to fill the missing IDs
    _ANSWER_NOT_FOUND: ClassVar[str] = "<answer not found>"

    def clean_answer(self, answer: str) -> str:
        """
        Clean the answer.

        :param answer: The answer to clean.
        :return: The cleaned answer.
        """
        if not answer:
            return self._ANSWER_NOT_FOUND
        answers = answer.split(",")
        answers = [answer.strip().lower() for answer in answers]
        answers = [answer for answer in answers if answer != ""]
        return answers[0] if answers else self._ANSWER_NOT_FOUND

    def __len__(self) -> int:
        """
        Get the length of the answer space.

        :return: The length of the answer space.
        """
        ...

    def answer_id_to_answer(self, answer_id: int) -> str:
        """
        Convert the answer id to the answer.

        :param answer_id: The answer id.
        :return: The answer.
        """
        ...

    def answer_to_answer_id(self, answer: str) -> int:
        """
        Convert the answer to the answer id.

        :param answer: The answer.
        :return: The answer id.
        """
        ...

    @staticmethod
    def logits_to_answer_id(logits: torch.Tensor) -> int:
        """
        Convert the logits to the answer id.

        :param logits: The logits.
        :return: The answer id.
        """
        return logits.argmax(-1).detach().cpu().numpy()


class AnswerSpaceRegistry(Registry[AvailableDatasets, type[AnswerSpace]]):
    """The multi-modal collator registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from utils.datasets.vqa_v2 import (  # noqa: F401
            VqaV2AnswerSpace,
            VqaV2SampleAnswerSpace,
        )


registry: Final[AnswerSpaceRegistry] = AnswerSpaceRegistry()
