"""The module contains the configuration for the DAQUAR dataset."""
from pathlib import Path
from typing import Final

from config.env import DATA_PATH

# A directory where the raw DAQUAR dataset can be found.
# The DAQUAR dataset is tracked with DVC.
# If you don't have it locally, you can download it with:
# dvc pull
RAW_DAQUAR_PATH: Final[Path] = DATA_PATH / "raw" / "daquar"

# A tuple containing the names of the raw training and evaluation data files.
RAW_DAQUAR_DATA_FILES: Final[tuple[str, str]] = (
    "data_train.csv",
    "data_eval.csv",
)

# A tuple containing the names of the preprocessed training and evaluation data files.
PROCESSED_DAQUAR_DATA_FILES: Final[tuple[str, str]] = (
    "daquar_train_flattened.csv",
    "daquar_eval_flattened.csv",
)

# A directory where the processed DAQUAR dataset will be saved.
PROCESSED_DAQUAR_PATH: Final[Path] = DATA_PATH / "processed" / "daquar"
