"""VQA V2 dataset."""
from functools import lru_cache
from typing import Literal

import datasets
import pandas as pd

from config.datasets.vqa_v2 import (
    VQA_V2_ANSWERS_SPACE_PATH,
    VQA_V2_SAMPLE_ANSWERS_SPACE_PATH,
)
from utils.datasets import AvailableDatasets
from utils.datasets import registry as datasets_registry
from utils.datasets.answer_space import AnswerSpace
from utils.datasets.answer_space import registry as answer_space_registry


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


@lru_cache(maxsize=1)
def load_vqa_v2_answers_space() -> pd.DataFrame:
    """
    Load the VQA V2 answers space.

    :return: The VQA V2 answers space.
    """
    return pd.read_json(
        VQA_V2_ANSWERS_SPACE_PATH,
        orient="records",
    )


@answer_space_registry.register(AvailableDatasets.VQA_V2_SAMPLE)
class VqaV2SampleAnswerSpace(AnswerSpace):
    """
    The VQA V2 sample answers space.

    It can be used to convert the answers to answer ids
    and vice versa.
    """

    def __init__(self):
        """
        Initialize the VQA V2 sample answers space.

        It loads the answers space from the file.
        """
        self._answers_space = None

    def _add_fake_answer(self):
        """
        Add a fake answer to the answers space.

        It is needed for the evaluation of the VQA V2 sample dataset.
        """
        self._answers_space = self.answers_space.append(
            {"answer": self._ANSWER_NOT_FOUND},
            ignore_index=True,
        )

    @staticmethod
    def _do_load_answers_space() -> pd.DataFrame:
        """
        Load the answers space.

        :return: The answers space.
        """
        return load_vqa_v2_sample_answers_space()

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


@answer_space_registry.register(AvailableDatasets.VQA_V2)
class VqaV2AnswerSpace(VqaV2SampleAnswerSpace):
    """
    VQA V2 answers space.

    It can be used to convert the answers to answer ids
    and vice versa.
    """

    @staticmethod
    def _do_load_answers_space() -> pd.DataFrame:
        """
        Load the answers space.

        :return: The answers space.
        """
        return load_vqa_v2_answers_space()


@datasets_registry.register(AvailableDatasets.VQA_V2_SAMPLE)
def load_vqa_v2_sample_datasets() -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Load the VQA V2 sample datasets.

    :return: The VQA V2 sample datasets.
    """
    return (
        load_vqa_v2_sample_train_dataset(),
        load_vqa_v2_sample_val_dataset(),
        load_vqa_v2_sample_test_dataset(),
    )


@datasets_registry.register(AvailableDatasets.VQA_V2)
def load_vqa_v2_datasets() -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Load the VQA V2 datasets.

    :return: The VQA V2 datasets.
    """
    return (
        load_vqa_v2("train"),
        load_vqa_v2("validation"),
        load_vqa_v2("test"),
    )
