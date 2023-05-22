"""Batch-related utilities."""
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.types import Device
from transformers import BatchEncoding, BatchFeature

from utils.types import BatchType


def batch_to_device(batch: BatchType, device: Device) -> BatchType:
    """
    Move the batch to the device.

    :param batch: The batch.
    :param device: The device.
    :return: The batch on the device.
    """
    if isinstance(batch, (BatchEncoding, BatchFeature, torch.Tensor)):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    else:
        return batch


def convert_batch_to_mapping_of_features(batch: list[Mapping[str, Any]]) -> Mapping[str, list[Any]]:
    """
    Convert the batch to a dict of features.

    If the batch is a list, then it is converted to a dict of features
    with the keys having the lists of respective feature values for each
    sample in the batch.

    :param batch: The batch.
    :return: The list.
    """
    if isinstance(batch, Mapping):
        return batch
    elif not batch:
        return {}
    else:
        return {key: [sample[key] for sample in batch] for key in batch[0].keys()}


def convert_batch_to_sequence_of_mappings(batch: Mapping[str, Sequence[Any]]) -> Sequence[Mapping[str, Any]]:
    """
    Convert the batch to a list of dicts.

    If the batch is a dict, then it is converted to a list of dicts
    with the keys having the values for each sample in the batch.

    :param batch: The batch.
    :return: The list.
    """
    if isinstance(batch, Sequence):
        return batch
    elif not batch:
        return []
    else:
        some_value = next(iter(batch.values()))
        return [{key: value[i] for key, value in batch.items()} for i in range(len(some_value))]
