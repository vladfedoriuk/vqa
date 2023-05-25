"""Pipeline to process the VQA V2 dataset."""
import logging

import pandas as pd

from config.datasets.vqa_v2 import (
    VQA_V2_ANSWERS_SPACE_PATH,
    VQA_V2_RAW_ANSWERS_SPACE_PATH,
)
from utils.datasets.pipelines import flatten_multiple_answers

logger = logging.getLogger(__name__)


def get_raw_answers_space() -> pd.DataFrame:
    """Return the raw answers space for the VQA V2 dataset."""
    return pd.read_json(
        VQA_V2_RAW_ANSWERS_SPACE_PATH,
        orient="records",
    )


def save_answers_space(answers_space: pd.DataFrame):
    """Save the flattened answers space for the VQA V2 dataset."""
    answers_space.to_json(
        VQA_V2_ANSWERS_SPACE_PATH,
        orient="records",
    )


def main():
    """Process the VQA V2 dataset."""
    logger.info("Reading the raw VQA V2 dataset...")
    answers_space = get_raw_answers_space()
    logger.info("Cleaning the answers space...")
    answers_space = flatten_multiple_answers(answers_space)
    logger.info("Saving the answers space...")
    save_answers_space(answers_space)
    logger.info("Done!")


if __name__ == "__main__":
    main()
