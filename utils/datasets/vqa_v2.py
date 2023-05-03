"""VQA V2 dataset."""
from functools import lru_cache
from typing import ClassVar, Literal

import datasets
import pandas as pd

from config.datasets.vqa_v2 import VQA_V2_SAMPLE_ANSWERS_SPACE_PATH
from utils.datasets import AnswerSpace


def load_vqa_v2_sample_train_dataset() -> datasets.Dataset:
    """
    Load the VQA V2 sample train dataset.

    It can be used for testing the models
    before training them on the full dataset.


    :return: The VQA V2 sample train dataset.
    """
    return datasets.load_dataset(
        "Multimodal-Fatima/VQAv2_sample_train",
        split="train",
    )


def load_vqa_v2_sample_val_dataset() -> datasets.Dataset:
    """
    Load the VQA V2 sample validation dataset.

    It can be used for testing the models
    before training them on the full dataset.

    :return: The VQA V2 sample validation dataset.
    """
    return datasets.load_dataset(
        "Multimodal-Fatima/VQAv2_sample_validation",
        split="validation",
    )


def load_vqa_v2_sample_test_dataset() -> datasets.Dataset:
    """
    Load the VQA V2 sample test dataset.

    It can be used for testing the models
    before training them on the full dataset.

    :return: The VQA V2 sample test dataset.
    """
    return datasets.load_dataset(
        "Multimodal-Fatima/VQAv2_sample_testdev",
        split="testdev",
    )


def load_vqa_v2(split: Literal["train", "validation", "test"]) -> datasets.Dataset:
    """
    Load the VQA V2 dataset.

    :param split: The split of the dataset to load.
    :return: The VQA V2 dataset.
    """
    return datasets.load_dataset(
        "HuggingFaceM4/VQAv2",
        split=split,
    )


@lru_cache(maxsize=1)
def load_vqa_v2_sample_answers_space() -> pd.DataFrame:
    """
    Load the VQA V2 answers space.

    :return: The VQA V2 answers space.
    """
    return pd.read_json(
        VQA_V2_SAMPLE_ANSWERS_SPACE_PATH,
        orient="records",
    )


# TODO: Maybe some registry for the answer spaces?
class VqaV2SampleAnswerSpace(AnswerSpace):
    """
    The VQA V2 sample answers space.

    It can be used to convert the answers to answer ids
    and vice versa.
    """

    #: The fake answer to fill the missing IDs
    _ANSWER_NOT_FOUND: ClassVar[str] = "<answer not found>"

    def __init__(self):
        """
        Initialize the VQA V2 sample answers space.

        It loads the answers space from the file.
        """
        self._answers_space = None

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

    def _add_fake_answer(self):
        """
        Add a fake answer to the answers space.

        It is needed for the evaluation of the VQA V2 sample dataset.
        """
        self._answers_space = self.answers_space.append(
            {"answer": self._ANSWER_NOT_FOUND},
            ignore_index=True,
        )

    @property
    def answers_space(self) -> pd.DataFrame:
        """
        Get the answers space.

        Lazy loads the answers space from the file.

        :return: The answers space.
        """
        if self._answers_space is None:
            self._answers_space = load_vqa_v2_sample_answers_space()
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
