"""The phase module contains the code for the Phase enum."""
from enum import Enum


class Phase(Enum):
    """Phase of the model."""

    #: Training phase.
    TRAIN = "train"
    #: Evaluation phase.
    EVAL = "validation"
    #: Test phase.
    TEST = "test"
