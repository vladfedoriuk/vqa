"""The utilities for the datasets."""
from collections.abc import Callable
from typing import Final

import datasets

from utils.registry import Registry, RegistryKey


class AvailableDatasets(str, RegistryKey):
    """The available datasets."""

    VQA_V2_SAMPLE = "vqa_v2_sample"
    VQA_V2 = "vqa_v2"
    DAQUAR = "daquar"


DatasetsLoadingFunctionType = Callable[[], datasets.DatasetDict[datasets.Split, datasets.Dataset]]


class DatasetsRegistry(Registry[AvailableDatasets, DatasetsLoadingFunctionType]):
    """The datasets' registry."""

    @classmethod
    def initialize(cls) -> None:
        """Initialize the datasets loading functions."""
        from utils.datasets.daquar import load_daquar_datasets  # noqa: F401s
        from utils.datasets.vqa_v2 import (  # noqa: F401
            load_vqa_v2_datasets,
            load_vqa_v2_sample_datasets,
        )


registry: Final[DatasetsRegistry] = DatasetsRegistry()


__all__ = [
    "AvailableDatasets",
    "DatasetsLoadingFunctionType",
    "registry",
    "DatasetsRegistry",
]
