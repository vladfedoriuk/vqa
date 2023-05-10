"""No-op transform."""
from typing import TypeVar

DataType = TypeVar("DataType")


def noop(data: DataType) -> DataType:
    """
    No-op transform.

    :param data: Any data.
    :return: The same data.
    """
    return data
