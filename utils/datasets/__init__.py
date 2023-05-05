"""The utilities for the datasets."""
from collections.abc import Callable
from typing import Any, Final

import datasets

from utils.registry import Registry, RegistryKey


def convert_batch_to_dict_of_features(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Convert the batch to a dict of features.

    If the batch is a list, then it is converted to a dict of features
    with the keys having the lists of respective feature values for each
    sample in the batch.

    :param batch: The batch.
    :return: The list.
    """
    if isinstance(batch, dict):
        return batch
    elif not batch:
        return {}
    else:
        return {key: [sample[key] for sample in batch] for key in batch[0].keys()}


def convert_batch_to_list_of_dicts(batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Convert the batch to a list of dicts.

    If the batch is a dict, then it is converted to a list of dicts
    with the keys having the values for each sample in the batch.

    :param batch: The batch.
    :return: The list.
    """
    if isinstance(batch, list):
        return batch
    elif not batch:
        return []
    else:
        some_value = next(iter(batch.values()))
        return [{key: value[i] for key, value in batch.items()} for i in range(len(some_value))]


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
    "convert_batch_to_dict_of_features",
    "convert_batch_to_list_of_dicts",
    "DatasetsLoadingFunctionType",
    "registry",
    "DatasetsRegistry",
]
