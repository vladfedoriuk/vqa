"""
Answers space processing for the VQA V2 sample dataset.

The full VQAv2 dataset can be found:

https://visualqa.org/index.html
"""
import itertools
import logging
from functools import lru_cache
from pathlib import Path

import datasets
import pandas as pd

from config.datasets.vqa_v2 import VQA_V2_SAMPLE_ANSWERS_SPACE_PATH
from utils.datasets.pipelines import flatten_multiple_answers
from utils.datasets.vqa_v2 import (
    load_vqa_v2_sample_test_dataset,
    load_vqa_v2_sample_train_dataset,
    load_vqa_v2_sample_val_dataset,
)

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


def _extract_alternative_answers(dataset: datasets.Dataset) -> pd.Series:
    """
    Extract the alternative answers from the dataset.

    :param dataset: The dataset.
    :return: The alternative answers.
    """
    return pd.Series(itertools.chain.from_iterable(dataset["answers"]))


def _get_alternative_answers_if_needed(dataset: datasets.Dataset) -> pd.Series:
    """
    Get the alternative answers if needed.

    :param dataset: The dataset.
    :return: The alternative answers if needed.
    """
    return _extract_alternative_answers(dataset) if include_alternative_answers() else pd.Series()


@lru_cache(maxsize=1)
def include_alternative_answers() -> bool:
    """
    Return whether to include alternative answers.

    :return: Whether to include alternative answers.
    """
    import dvc.api

    params = dvc.api.params_show()
    return params["vqa_v2_sample"]["answers_space"]["include_alternative_answers"]


def get_answers_space(
    train_data: datasets.Dataset,
    val_data: datasets.Dataset,
    test_data: datasets.Dataset,
):
    """
    Get the answers space for the VQA V2 sample dataset.

    :param train_data: The train dataset.
    :param val_data: The validation dataset.
    :param test_data: The test dataset.
    :return: The answers space.
    """
    logger.info("Getting the answers space...")
    answers = pd.DataFrame(
        pd.concat(
            [
                _extract_multiple_choice_answers(train_data),
                _extract_multiple_choice_answers(val_data),
                _extract_multiple_choice_answers(test_data),
                *(
                    (
                        _get_alternative_answers_if_needed(train_data),
                        _get_alternative_answers_if_needed(val_data),
                    )
                    if include_alternative_answers()
                    else ()
                ),
            ]
        ),
        columns=["answer"],
    )
    # Split the answers containing multiple answers into multiple rows.
    return flatten_multiple_answers(answers)


def save_answers_space(answers_space: pd.DataFrame, path: Path):
    """
    Save the answers space for the VQA V2 sample dataset.

    :param answers_space: The answers space.
    :param path: The path to save the answers space to.
    :return: None
    """
    logger.info("Saving the answers space...")
    answers_space.to_json(
        path,
        orient="records",
        indent=4,
    )


def main() -> None:
    """
    Generate the answers space for the VQA V2 sample dataset.

    :return: None
    """
    logger.info("Generating the answers space...")
    train_data = load_vqa_v2_sample_train_dataset()
    val_data = load_vqa_v2_sample_val_dataset()
    test_data = load_vqa_v2_sample_test_dataset()
    logger.info("Loaded the datasets.")
    answers_space = get_answers_space(train_data, val_data, test_data)
    logger.info("Got the answers space.")
    save_answers_space(answers_space, VQA_V2_SAMPLE_ANSWERS_SPACE_PATH)
    logger.info("Saved the answers space.")


if __name__ == "__main__":
    main()
