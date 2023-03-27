"""Configuration variables."""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# TODO: Move to the pydantic settings.
load_dotenv(Path(__file__).parent.parent / ".env")

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

# A project name on the Weights and Biases
WANDB_PROJECT: Final[str] = os.getenv("WANDB_PROJECT", "vqa")
