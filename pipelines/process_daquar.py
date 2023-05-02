"""
The processing pipeline for the DAQUAR dataset.

The original DAQUAR dataset can be found:

https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge/

The pipeline includes:

- Loading the dataset.
- Flattening the comma-separated answers.
- Saving the datasets as CSV files.
"""  # noqa: E501
import logging

import pandas as pd

from config.datasets.daquar import (
    PROCESSED_DAQUAR_DATA_FILES,
    PROCESSED_DAQUAR_PATH,
    RAW_DAQUAR_DATA_FILES,
    RAW_DAQUAR_PATH,
)
from utils.datasets.pipelines import flatten_multiple_answers

logger = logging.getLogger(__name__)


def load_daquar_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the DAQUAR dataset.

    :return: A tuple containing training and evaluation dataframes respectively.
    """
    logger.info("Loading the DAQUAR dataset...")
    train_data_file, eval_data_file = RAW_DAQUAR_DATA_FILES
    train_data = pd.read_csv(RAW_DAQUAR_PATH / train_data_file)
    eval_data = pd.read_csv(RAW_DAQUAR_PATH / eval_data_file)
    return train_data, eval_data


def flatten_data(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flatten multiple answers in a single row.

    :param train_data: The training data.
    :param eval_data: The evaluation data.
    :return: The flattened training and evaluation data.
    """
    logger.info("Flattening the DAQUAR dataset...")
    return flatten_multiple_answers(train_data), flatten_multiple_answers(eval_data)


def save_flatten_data(
    flattened_train_data: pd.DataFrame,
    flattened_eval_data: pd.DataFrame,
) -> None:
    """
    Save the flattened data.

    :param flattened_train_data: The flattened training data.
    :param flattened_eval_data: The flattened evaluation data.
    :return: None
    """
    logger.info("Saving the flattened DAQUAR dataset...")
    train_data_file, eval_data_file = PROCESSED_DAQUAR_DATA_FILES
    flattened_train_data.to_csv(PROCESSED_DAQUAR_PATH / train_data_file, index=False)
    flattened_eval_data.to_csv(PROCESSED_DAQUAR_PATH / eval_data_file, index=False)


def main():
    """Run the pipeline."""
    logger.info("Running the DAQUAR processing pipeline...")
    save_flatten_data(*flatten_data(*load_daquar_data()))
    logger.info("Finished running the DAQUAR processing pipeline.")


if __name__ == "__main__":
    main()
