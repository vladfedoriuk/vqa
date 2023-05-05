"""The utilities to load and transform the datasets."""
import pandas as pd


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
    data.fillna("", inplace=True)
    flattened_data = data.iloc[data.index.repeat(data["answer"].str.count(",") + 1)]
    split_answers = (
        data["answer"]
        .str.split(",", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .rename("answer")
        .str.strip()
        .str.lower()
    )
    flattened_data.loc[:, "answer"] = split_answers
    flattened_data = flattened_data[flattened_data["answer"] != ""]
    flattened_data.reset_index(drop=True, inplace=True)
    return flattened_data
