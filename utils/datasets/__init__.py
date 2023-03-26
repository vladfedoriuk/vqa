"""The utilities for the datasets."""
from typing import Any, Protocol

import torch


def convert_batch_to_dict_of_features(
    batch: list[dict[str, Any]]
) -> dict[str, list[Any]]:
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


class AnswerSpace(Protocol):
    """
    The answer space protocol.

    It can be used to convert the answers to answer ids
    and vice versa.

    The protocol is used for type hinting.
    """

    def __len__(self) -> int:
        """
        Get the length of the answer space.

        :return: The length of the answer space.
        """
        ...

    def answer_id_to_answer(self, answer_id: int) -> str:
        """
        Convert the answer id to the answer.

        :param answer_id: The answer id.
        :return: The answer.
        """
        ...

    def answer_to_answer_id(self, answer: str) -> int:
        """
        Convert the answer to the answer id.

        :param answer: The answer.
        :return: The answer id.
        """
        ...

    @staticmethod
    def logits_to_answer_id(logits: torch.Tensor) -> int:
        """
        Convert the logits to the answer id.

        :param logits: The logits.
        :return: The answer id.
        """
        return logits.argmax(-1).detach().cpu().numpy()
