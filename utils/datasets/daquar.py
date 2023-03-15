"""The module contains the code to load the DAQUAR dataset."""
import datasets

from config import PROCESSED_DAQUAR_PATH


def load_daquar() -> datasets.Dataset:
    """
    Load the DAQUAR dataset.

    :return: The DAQUAR dataset.
    """
    return datasets.load_dataset(
        "csv",
        data_files={
            "train": str(PROCESSED_DAQUAR_PATH / "daquar_train_flattened.csv"),
            "validation": str(PROCESSED_DAQUAR_PATH / "daquar_eval_flattened.csv"),
        },
    )
