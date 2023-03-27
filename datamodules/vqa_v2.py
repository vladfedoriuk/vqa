"""The VQA V2 data modules."""
from collections import defaultdict

import lightning.pytorch as pl
from transformers import AutoImageProcessor, AutoTokenizer

from collators.vqa_v2 import VqaV2SampleCollator
from transforms.noop import noop
from transforms.vqa_v2 import image_augmentation_for_vqa_v2
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace
from utils.phase import Phase


# TODO: Make this data module dataset-agnostic.
class VqaV2SampleDataModule(pl.LightningDataModule):
    """The VQA V2 sample data module."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        image_processor: AutoImageProcessor,
        answer_space: VqaV2SampleAnswerSpace,
        batch_size: int = 64,
    ):
        """
        Construct the VQA V2 sample data module.

        :param tokenizer: The tokenizer.
        :param image_processor: The image processor.
        :param answer_space: The answer space.
        :param batch_size: The batch size.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.train_transforms = {
            Phase.TRAIN: image_augmentation_for_vqa_v2,
        }
        self.answer_space = answer_space
        self.batch_size = batch_size
        self._dataloaders = None

    def setup(self, stage: str):
        """
        Set up the data module.

        :param stage: The stage to set up.
        :return: None.
        """
        image_transforms = defaultdict(lambda: noop)
        image_transforms |= self.train_transforms
        self._dataloaders = VqaV2SampleCollator.get_dataloaders(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            image_transforms=image_transforms,
            answer_space=self.answer_space,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        """
        Get the training dataloader.

        :return: The training dataloader.
        """
        return self._dataloaders[Phase.TRAIN]

    def val_dataloader(self):
        """
        Get the validation dataloader.

        :return: The validation dataloader.
        """
        return self._dataloaders[Phase.EVAL]

    def test_dataloader(self):
        """
        Get the test dataloader.

        :return: The test dataloader.
        """
        return self._dataloaders[Phase.TEST]
