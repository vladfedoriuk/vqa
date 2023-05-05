"""
The answer spaces module.

The answer spaces are used to convert the answers to answer ids
and vice versa.
"""
from typing import ClassVar, Final, Protocol

import pandas as pd
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


class PandasAnswerSpace(AnswerSpace):
    """The pandas answer space."""

    def __init__(self):
        """Initialize the answer space."""
        self._answers_space = None

    def _add_fake_answer(self):
        """
        Add a fake answer to the answers space.

        It is needed for the evaluation of the VQA V2 sample dataset.
        """
        self._answers_space = pd.concat(
            [
                self._answers_space,
                pd.DataFrame(
                    {
                        "answer": [self._ANSWER_NOT_FOUND],
                        "answer_id": [len(self._answers_space)],
                    }
                ),
            ]
        )

    def _do_load_answers_space(self) -> pd.DataFrame:
        """
        Load the answers space.

        :return: The answers space.
        """
        raise NotImplementedError

    @property
    def answers_space(self) -> pd.DataFrame:
        """
        Get the answers space.

        Lazy loads the answers space from the file.

        :return: The answers space.
        """
        if self._answers_space is None:
            self._answers_space = self._do_load_answers_space()
            self._add_fake_answer()
        return self._answers_space

    def __len__(self):
        """
        Get the number of answers in the answers space.

        :return: The number of answers in the answers space.
        """
        return len(self.answers_space)

    def answer_id_to_answer(self, answer_id):
        """
        Convert the answer id to the answer.

        :param answer_id: The answer id.
        :return: The answer.
        """
        try:
            return self.answers_space["answer"].iloc[answer_id]
        except IndexError:
            return self._ANSWER_NOT_FOUND

    def __get_answer_id(self, answer):
        """
        Get the answer ID.

        :param answer: a given answer
        :return: The ID of the answer in the processed dataset
        """
        return self.answers_space[self.answers_space["answer"] == answer].index[0]

    def answer_to_answer_id(self, answer):
        """
        Convert the answer to the answer id.

        :param answer: The answer.
        :return: The answer id.
        """
        answer = self.clean_answer(answer)
        try:
            return self.__get_answer_id(answer)
        except IndexError:
            return self.__get_answer_id(self._ANSWER_NOT_FOUND)


class AnswerSpaceRegistry(Registry[AvailableDatasets, type[AnswerSpace]]):
    """The answer space registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry."""
        from utils.datasets.daquar import DaquarAnswerSpace  # noqa: F401
        from utils.datasets.vqa_v2 import (  # noqa: F401
            VqaV2AnswerSpace,
            VqaV2SampleAnswerSpace,
        )


registry: Final[AnswerSpaceRegistry] = AnswerSpaceRegistry()


def get_predicted_answer(answer_space: AnswerSpace, output_logits: torch.Tensor):
    """
    Get the predicted answer for the given datapoint and model output logits.

    :param answer_space: The answer space.
    :param output_logits: The output logits.
    :return: The caption.
    """
    # Get the model prediction
    predicted_answer_id = answer_space.logits_to_answer_id(output_logits)
    # Retrieve the answer from the answer space
    return answer_space.answer_id_to_answer(predicted_answer_id)
