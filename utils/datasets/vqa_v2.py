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
from utils.datasets.answer_space import PandasAnswerSpace
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
class VqaV2SampleAnswerSpace(PandasAnswerSpace):
    """
    The VQA V2 sample answers space.

    It can be used to convert the answers to answer ids
    and vice versa.
    """

    def _do_load_answers_space(self) -> pd.DataFrame:
        """
        Load the answers space.

        :return: The answers space.
        """
        return load_vqa_v2_sample_answers_space()


@answer_space_registry.register(AvailableDatasets.VQA_V2)
class VqaV2AnswerSpace(VqaV2SampleAnswerSpace):
    """
    VQA V2 answers space.

    It can be used to convert the answers to answer ids
    and vice versa.
    """

    def _do_load_answers_space(self) -> pd.DataFrame:
        """
        Load the answers space.

        :return: The answers space.
        """
        return load_vqa_v2_answers_space()


@datasets_registry.register(AvailableDatasets.VQA_V2_SAMPLE)
def load_vqa_v2_sample_datasets() -> datasets.DatasetDict:
    """
    Load the VQA V2 sample datasets.

    :return: The VQA V2 sample datasets.
    """
    return datasets.DatasetDict(
        {
            datasets.Split.TRAIN: load_vqa_v2_sample_train_dataset(),
            datasets.Split.VALIDATION: load_vqa_v2_sample_val_dataset(),
            datasets.Split.TEST: load_vqa_v2_sample_test_dataset(),
        }
    )


@datasets_registry.register(AvailableDatasets.VQA_V2)
def load_vqa_v2_datasets() -> datasets.DatasetDict:
    """
    Load the VQA V2 datasets.

    :return: The VQA V2 datasets.
    """
    return datasets.DatasetDict(
        {
            datasets.Split.TRAIN: load_vqa_v2("train"),
            datasets.Split.VALIDATION: load_vqa_v2("validation"),
            datasets.Split.TEST: load_vqa_v2("test"),
        }
    )
