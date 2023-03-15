"""Configuration variables."""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# A directory where the datasets can be found.
DATASETS_PATH: Final[Path] = Path(os.getenv("DATASETS_PATH", "/shared/sets/datasets/"))

# A directory to which the experiments artifacts are saved.
SAVE_ARTIFACTS_PATH: Final[Path] = Path(
    os.getenv("SAVE_ARTIFACTS_PATH", "/.local/share")
)

# A team name (entity) on the Weights and Biases
WANDB_ENTITY: Final[str] = os.getenv("WANDB_ENTITY", "")

# Datasets cache directory
HF_DATASETS_CACHE: Final[Path] = Path(
    os.getenv("HF_DATASETS_CACHE", DATASETS_PATH / "huggingface_cache")
)

# A directory where the DAQUAR dataset can be found.
# The DAQUAR dataset is tracked with DVC.
# If you don't have it locally, you can download it with:
# dvc pull
RAW_DAQUAR_PATH: Final[Path] = Path(__file__).parent / "data" / "raw" / "daquar"

# A directory where the processed DAQUAR dataset will be saved.
PROCESSED_DAQUAR_PATH: Final[Path] = (
    Path(__file__).parent / "data" / "processed" / "daquar"
)
