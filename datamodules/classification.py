"""
The classification data module.

To know more about the classification data modules,
check out the documentation:

https://lightning.ai/docs/pytorch/stable/data/datamodule.html
https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
"""
import lightning.pytorch as pl
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from collators import ClassificationCollator
from models.backbones import BackboneConfig
from transforms.noop import noop
from transforms.vqa_v2 import image_augmentation_for_vqa_v2
from utils.datasets import DatasetsLoadingFunctionType
from utils.datasets.answer_space import AnswerSpace
from utils.phase import Phase


class MultiModalClassificationDataModule(pl.LightningDataModule):
    """The VQA V2 sample data module."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor,
        image_encoder_config: type[BackboneConfig],
        text_encoder_config: type[BackboneConfig],
        answer_space: AnswerSpace,
        collator_cls: type[ClassificationCollator],
        datasets_loading_function: DatasetsLoadingFunctionType,
        batch_size: int = 64,
    ):
        """
        Construct the VQA V2 sample data module.

        :param tokenizer: The tokenizer.
        :param image_processor: The image processor.
        :param answer_space: The answer space.
        :param collator_cls: The collator class.
        :param datasets_loading_function: The datasets loading function.
        :param batch_size: The batch size.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_encoder_config = image_encoder_config
        self.text_encoder_config = text_encoder_config
        self.image_transforms = {
            Phase.TRAIN: image_augmentation_for_vqa_v2,
            Phase.EVAL: noop,
            Phase.TEST: noop,
        }
        self.answer_space = answer_space
        self.batch_size = batch_size
        self.collator_cls = collator_cls
        self.dataset_loading_function = datasets_loading_function
        self._datasets = None

    def setup(self, stage: str):
        """
        Set up the data module.

        :param stage: The stage to set up.
        :return: None.
        """
        # TODO: Use ``stage`` parameter.
        self._datasets = dict(zip(Phase, self.dataset_loading_function()))

    def train_dataloader(self):
        """
        Get the training dataloader.

        :return: The training dataloader.
        """
        return DataLoader(
            self._datasets[Phase.TRAIN],
            sampler=RandomSampler(self._datasets[Phase.TRAIN]),
            collate_fn=self.collator_cls(
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                image_encoder_config=self.image_encoder_config,
                text_encoder_config=self.text_encoder_config,
                image_transforms=self.image_transforms[Phase.TRAIN],
                answer_space=self.answer_space,
            ),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        """
        Get the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            self._datasets[Phase.EVAL],
            sampler=SequentialSampler(self._datasets[Phase.EVAL]),
            collate_fn=self.collator_cls(
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                image_encoder_config=self.image_encoder_config,
                text_encoder_config=self.text_encoder_config,
                image_transforms=self.image_transforms[Phase.EVAL],
                answer_space=self.answer_space,
            ),
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        """
        Get the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            self._datasets[Phase.TEST],
            sampler=SequentialSampler(self._datasets[Phase.TEST]),
            collate_fn=self.collator_cls(
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                image_encoder_config=self.image_encoder_config,
                text_encoder_config=self.text_encoder_config,
                image_transforms=self.image_transforms[Phase.TEST],
                answer_space=self.answer_space,
            ),
            batch_size=self.batch_size,
        )

    def predict_dataloader(self):
        """
        Get the "predict" dataloader.

        :return: The "predict" dataloader.
        """
        return self.test_dataloader()
