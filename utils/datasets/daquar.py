"""The module contains the code to load the DAQUAR dataset."""
import datasets

from config.datasets.daquar import PROCESSED_DAQUAR_DATA_FILES, PROCESSED_DAQUAR_PATH


def load_daquar() -> datasets.Dataset:
    """
    Load the DAQUAR dataset.

    :return: The DAQUAR dataset.
    """
    train_data_file, eval_data_file = PROCESSED_DAQUAR_DATA_FILES
    return datasets.load_dataset(
        "csv",
        data_files={
            "train": str(PROCESSED_DAQUAR_PATH / train_data_file),
            "validation": str(PROCESSED_DAQUAR_PATH / eval_data_file),
        },
    )
