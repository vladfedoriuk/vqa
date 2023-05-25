"""
The module contains the base class for classification modules.

To know more about the classification modules,
check out the documentation:

https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
"""
import lightning.pytorch as pl
import torch.optim
from torch import nn
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from collators import ClassificationCollator
from lightning_modules.mixins import VQAClassificationMixin
from models.backbones import BackboneConfig
from transforms.image import default_image_batch_transforms_factory
from transforms.noop import default_noop_transforms_factory
from transforms.text import default_text_batch_transforms_factory
from utils.datasets import DatasetsLoadingFunctionType
from utils.datasets.answer_space import AnswerSpace
from utils.types import BatchType


class MultiModalFusionClassificationModule(VQAClassificationMixin, pl.LightningModule):
    """Base class for classification modules."""

    def __init__(
        self,
        fusion: nn.Module,
        classifier: nn.Module,
        answer_space: AnswerSpace,
        collator_cls: type[ClassificationCollator],
        datasets_loading_function: DatasetsLoadingFunctionType,
        image_encoder: nn.Module,
        image_processor: BaseImageProcessor,
        text_encoder: nn.Module,
        tokenizer: PreTrainedTokenizer,
        image_encoder_config: type[BackboneConfig],
        text_encoder_config: type[BackboneConfig],
        batch_size: int = 64,
        lr: float = 1e-4,
    ):
        """
        Initialize the module.

        :param fusion: The fusion.
        :param classifier: The classifier.
        :param answer_space: The answer space.
        :param collator_cls: The collator class.
        :param datasets_loading_function: The datasets loading function.
        :param image_encoder: The image encoder.
        :param image_processor: The image processor.
        :param tokenizer: The tokenizer.
        :param text_encoder: The text encoder.
        :param image_encoder_config: The image encoder config.
        :param text_encoder_config: The text encoder config.
        :param batch_size: The batch size.
        :param lr: The learning rate.
        """
        super().__init__()
        self.fusion = fusion
        self.classifier = classifier

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.answer_space = answer_space
        self.classes_num = len(answer_space)

        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.image_encoder_config = image_encoder_config
        self.text_encoder_config = text_encoder_config

        self.batch_image_transforms = default_image_batch_transforms_factory()
        self.single_image_transforms = default_noop_transforms_factory()
        self.batch_text_transforms = default_text_batch_transforms_factory()
        self.single_text_transforms = default_noop_transforms_factory()

        self.loss = torch.nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.datasets_loading_function = datasets_loading_function
        self._collator_cls = collator_cls
        self._collator_fn = None
        self._data = None
        self.learning_rate = lr

        self.validation_step_outputs = []

    def _get_embeddings(self, batch: BatchType):
        """
        Get the embeddings.

        :param batch: The batch.
        :return: The embeddings.
        """
        return {
            "image_emb": (
                self.image_encoder_config.get_image_representation_from_preprocessed(
                    model=self.image_encoder,
                    processor_output=batch,
                )
            ),
            "text_emb": (
                self.text_encoder_config.get_text_representation_from_tokenized(
                    model=self.text_encoder,
                    tokenizer_output=batch,
                )
            ),
        }

    def _shared_eval_step(self, batch: BatchType):
        """
        Perform a shared evaluation step.

        :param batch: The batch.
        :return: The loss and logits.
        """
        embeddings = self._get_embeddings(batch)
        fused_repr = self.fusion(embeddings["image_emb"], embeddings["text_emb"])
        logits = self.classifier(fused_repr)
        loss = self.loss(logits, batch["answer_label"])
        return {
            "loss": loss,
            "logits": logits,
            "labels": batch["answer_label"],
        }

    def configure_optimizers(self):
        """
        Configure the optimizers.

        The optimizer might leverage the learning rate finder:
        https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html

        :return: The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
