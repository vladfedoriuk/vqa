"""The module contains the code to load the DAQUAR dataset."""
import datasets
from transformers.image_utils import load_image

from config.datasets.daquar import (
    PROCESSED_DAQUAR_DATA_FILES,
    PROCESSED_DAQUAR_PATH,
    RAW_DAQUAR_PATH,
)


def load_daquar() -> datasets.Dataset:
    """
    Load the DAQUAR dataset.

    :return: The DAQUAR dataset.
    """
    train_data_file, eval_data_file = PROCESSED_DAQUAR_DATA_FILES
    dataset = datasets.load_dataset(
        "csv",
        data_files={
            "train": str(PROCESSED_DAQUAR_PATH / train_data_file),
            "validation": str(PROCESSED_DAQUAR_PATH / eval_data_file),
        },
    )
    return dataset.map(
        map_daquar_dataset_to_load_images,
        batched=True,
    )


def map_daquar_dataset_to_load_images(examples):
    """
    Map the DAQUAR dataset to load the images.

    :param examples: The examples of the dataset.
    :return: The examples of the dataset with the images loaded.
    """
    return {
        "image": [
            load_image(
                str(RAW_DAQUAR_PATH / "images" / f"{image_id}.png"),
            )
            for image_id in examples["image_id"]
        ]
    }
