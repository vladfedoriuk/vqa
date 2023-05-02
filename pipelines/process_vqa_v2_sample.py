"""
Answers space processing for the VQA V2 sample dataset.

The full VQAv2 dataset can be found:

https://visualqa.org/index.html
"""
import itertools
import logging
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
                pd.Series(
                    train_data["multiple_choice_answer"],
                ),
                pd.Series(
                    val_data["multiple_choice_answer"],
                ),
                pd.Series(
                    test_data["multiple_choice_answer"],
                ),
                pd.Series(itertools.chain.from_iterable(train_data["answers"])),
                pd.Series(itertools.chain.from_iterable(val_data["answers"])),
            ]
        ).unique(),
        columns=["answer"],
    )
    # Split the answers containing multiple answers into multiple rows.
    return flatten_multiple_answers(answers).rename_axis("answer_id").reset_index()


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


def generate_vqa_v2_sample_answers_space() -> None:
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
    generate_vqa_v2_sample_answers_space()
