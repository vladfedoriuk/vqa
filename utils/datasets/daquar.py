"""The module contains the code to load the DAQUAR dataset."""
import datasets
import pandas as pd
from transformers.image_utils import load_image

from config.datasets.daquar import (
    PROCESSED_DAQUAR_DATA_FILES,
    PROCESSED_DAQUAR_PATH,
    RAW_DAQUAR_PATH,
)
from utils.datasets import AvailableDatasets
from utils.datasets import registry as datasets_registry
from utils.datasets.answer_space import PandasAnswerSpace
from utils.datasets.answer_space import registry as answer_space_registry


def load_daquar() -> datasets.Dataset:
    """
    Load the DAQUAR dataset.

    :return: The DAQUAR dataset.
    """
    train_data_file, eval_data_file = PROCESSED_DAQUAR_DATA_FILES
    return datasets.load_dataset(
        "csv",
        data_files={
            "train": str(PROCESSED_DAQUAR_PATH / train_data_file),
            "validation": str(PROCESSED_DAQUAR_PATH / eval_data_file),
        },
    )


def _split_validation_dataset(
    dataset: datasets.Dataset,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """
    Split the validation dataset into test and validation datasets.

    :param dataset: The validation dataset.
    :return: The test and validation datasets.
    """
    test_validation = dataset.train_test_split(
        test_size=0.2,
        shuffle=True,
        seed=42,
    )
    test_dataset = test_validation["test"]
    validation_dataset = test_validation["train"]
    return validation_dataset, test_dataset


def load_images_for_batch(batch):
    """
    Load the images for a batch.

    :param batch: The batch.
    :return: A list of images.
    """
    return [
        load_image(
            str(RAW_DAQUAR_PATH / "images" / f"{image_id}.png"),
        )
        for image_id in batch["image_id"]
    ]


@datasets_registry.register(AvailableDatasets.DAQUAR)
def load_daquar_datasets() -> datasets.DatasetDict:
    """
    Load the DAQUAR datasets.

    :return: The train, validation and test datasets.
    """
    dataset = load_daquar()
    validation_dataset, test_dataset = _split_validation_dataset(
        datasets.Dataset.from_list(dataset["validation"]),
    )
    return datasets.DatasetDict(
        {
            datasets.Split.TRAIN: dataset["train"],
            datasets.Split.VALIDATION: validation_dataset,
            datasets.Split.TEST: test_dataset,
        }
    )


@answer_space_registry.register(AvailableDatasets.DAQUAR)
class DaquarAnswerSpace(PandasAnswerSpace):
    """The answer space for the DAQUAR dataset."""

    def _do_load_answers_space(self) -> pd.DataFrame:
        """
        Load the answer space.

        :return: The answer space.
        """
        return (
            pd.DataFrame(
                pd.concat(
                    [
                        pd.read_csv(PROCESSED_DAQUAR_PATH / processed_file)
                        for processed_file in PROCESSED_DAQUAR_DATA_FILES
                    ],
                    ignore_index=True,
                )["answer"].drop_duplicates()
            )
            .reset_index(drop=True)
            .rename_axis("answer_id")
        )
