"""
Generate the raw answers space for the VQA V2 dataset.

The raw answers space is the set of all the answers in the dataset.
"""
import itertools
import logging
from shutil import copyfileobj

import datasets
import pandas as pd

from config.datasets.vqa_v2 import (
    VQA_V2_ANSWERS_SPACE_EXAMPLE_PATH,
    VQA_V2_RAW_ANSWERS_SPACE_PATH,
)
from utils.datasets.vqa_v2 import load_vqa_v2

logger = logging.getLogger(__name__)


def _extract_multiple_choice_answers(dataset: datasets.Dataset) -> pd.Series:
    """
    Extract the multiple choice answers from the dataset.

    :param dataset: The dataset.
    :return: The multiple choice answers.
    """
    return pd.Series(
        dataset["multiple_choice_answer"],
    )


def _extract_answers(dataset: datasets.Dataset) -> pd.Series:
    """
    Extract the answers from the dataset.

    :param dataset: The dataset.
    :return: The answers.
    """
    return pd.Series(
        itertools.chain.from_iterable([data["answer"] for data in data_point] for data_point in dataset["answers"])
    )


def _get_datasets() -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Get the VQA V2 datasets.

    :return: The VQA V2 datasets.
    """
    return (
        load_vqa_v2("train"),
        load_vqa_v2("validation"),
        load_vqa_v2("test"),
    )


def create_raw_answers_space(datasets_: tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]) -> pd.DataFrame:
    """
    Create the raw answers space for the VQA V2 dataset.

    :param datasets_: The VQA V2 datasets.
    :return: The raw answers space.
    """
    train_data, val_data, test_data = datasets_
    return pd.DataFrame(
        pd.concat(
            [
                _extract_multiple_choice_answers(train_data),
                _extract_multiple_choice_answers(val_data),
                _extract_multiple_choice_answers(test_data),
                _extract_answers(train_data),
                _extract_answers(val_data),
            ]
        ).unique(),
        columns=["answer"],
    )


def save_raw_answers_space(raw_answers_space: pd.DataFrame):
    """
    Save the raw answers space for the VQA V2 dataset.

    :param raw_answers_space: The raw answers space.
    """
    raw_answers_space.to_json(
        VQA_V2_RAW_ANSWERS_SPACE_PATH,
        orient="records",
        indent=4,
    )


def copy_example_raw_answers_space():
    """Copy an example raw answers space for the VQA V2 dataset."""
    logger.info("Copying a sample raw answers space for the VQA V2 dataset...")
    with open(VQA_V2_RAW_ANSWERS_SPACE_PATH, "w") as raw_answers_space_f, open(
        VQA_V2_ANSWERS_SPACE_EXAMPLE_PATH
    ) as example_raw_answers_space_f:
        copyfileobj(example_raw_answers_space_f, raw_answers_space_f)


def main():
    """Generate the raw answers space for the VQA V2 dataset."""
    import dvc.api

    params = dvc.api.params_show()
    if params["vqa_v2"]["answers_space"]["raw"]["copy_example"]:
        logger.info("Skipping the generation of the raw answers space for the VQA V2 dataset...")
        logger.info("Copying a sample raw answers space for the VQA V2 dataset...")
        copy_example_raw_answers_space()
        logger.info("Done!")
        return

    logger.info("Loading the VQA V2 dataset...")
    datasets_ = _get_datasets()
    logger.info("Generating the raw answers space for the VQA V2 dataset...")
    raw_answers_space = create_raw_answers_space(datasets_)
    logger.info("Saving the raw answers space for the VQA V2 dataset...")
    save_raw_answers_space(raw_answers_space)
    logger.info("Done!")


if __name__ == "__main__":
    main()
