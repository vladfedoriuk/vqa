"""
The processing pipeline for the DAQUAR dataset.

The original DAQUAR dataset can be found:

https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge/

The pipeline includes:

- Loading the dataset.
- Flattening the comma-separated answers.
- Saving the datasets as CSV files.
"""  # noqa: E501
import pandas as pd

from config.datasets.daquar import (
    PROCESSED_DAQUAR_DATA_FILES,
    PROCESSED_DAQUAR_PATH,
    RAW_DAQUAR_DATA_FILES,
    RAW_DAQUAR_PATH,
)


def load_daquar_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the DAQUAR dataset.

    :return: A tuple containing training and evaluation dataframes respectively.
    """
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
    train_data_file, eval_data_file = PROCESSED_DAQUAR_DATA_FILES
    flattened_train_data.to_csv(PROCESSED_DAQUAR_PATH / train_data_file, index=False)
    flattened_eval_data.to_csv(PROCESSED_DAQUAR_PATH / eval_data_file, index=False)


def flatten_multiple_answers(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flatten multiple, comma-separated answers into separate rows.

    For each row with multiple answers, the function creates a new row for each answer.
    The new rows are identical to the original row, except for the answer column.

    :param data: A dataframe with multiple answers in a single row.
    :return: A dataframe with a single answer in a single row.
    """
    flattened_data = data.iloc[data.index.repeat(data["answer"].str.count(",") + 1)]
    split_answers = (
        data["answer"]
        .str.split(",", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .rename("answer")
    )
    flattened_data["answer"] = split_answers.str.strip()
    flattened_data.reset_index(drop=True, inplace=True)
    return flattened_data


def main():
    """Run the pipeline."""
    save_flatten_data(*flatten_data(*load_daquar_data()))


if __name__ == "__main__":
    main()
