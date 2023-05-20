"""The ViLT classification module."""
from typing import Any

import lightning.pytorch as pl
import torch

from collators import ClassificationCollator
from lightning_modules.mixins import VQAClassificationMixin
from models.backbones import BackboneConfig
from models.classifiers import default_classifier_factory
from transforms.image import default_image_batch_transforms_factory
from transforms.noop import default_noop_transforms_factory
from transforms.text import default_text_batch_transforms_factory
from utils.datasets import DatasetsLoadingFunctionType
from utils.datasets.answer_space import AnswerSpace
from utils.types import BatchType


class ViLTClassificationModule(VQAClassificationMixin, pl.LightningModule):
    """The ViLT classification module."""

    def __init__(
        self,
        vilt_backbone_config: type[BackboneConfig],
        answer_space: AnswerSpace,
        datasets_loading_function: DatasetsLoadingFunctionType,
        collator_cls: type[ClassificationCollator],
        batch_size: int = 64,
    ):
        """
        Initialize the module.

        The ViLT classification module.
        It uses the ViLT backbone to extract the multimodal embedding and then
        uses a classifier to classify the multimodal embedding.

        :param vilt_backbone_config: The ViLT backbone config.
        :param answer_space: The answer space.
        :param datasets_loading_function: The datasets loading function.
        :param collator_cls: The collator class.
        :param batch_size: The batch size.
        """
        super().__init__()
        self.vilt_backbone_config = vilt_backbone_config
        self.image_encoder_config = vilt_backbone_config
        self.text_encoder_config = vilt_backbone_config
        self.vilt = self.vilt_backbone_config.get_model()
        self.tokenizer = self.vilt_backbone_config.get_tokenizer()
        self.image_processor = self.vilt_backbone_config.get_image_processor()

        self.answer_space = answer_space
        self.classes_num = len(self.answer_space)

        self.classifier = default_classifier_factory(
            input_dim=self.vilt_backbone_config.get_multimodal_representation_size(),
            classes_num=self.classes_num,
        )

        self.datasets_loading_function = datasets_loading_function
        self.batch_size = batch_size
        self._collator_cls = collator_cls
        self._data = None
        self._collator_fn = None

        self.batch_image_transforms = default_image_batch_transforms_factory()
        self.single_image_transforms = default_noop_transforms_factory()
        self.single_text_transforms = default_noop_transforms_factory()
        self.batch_text_transforms = default_text_batch_transforms_factory()

        self.loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self) -> Any:
        """Configure the optimizers."""
        return torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)

    def _get_multimodal_embedding(self, batch: BatchType):
        """
        Get the multimodal embedding.

        :param batch: The batch.
        :return: The embedding.
        """
        return {
            "multimodal_emb": (
                self.vilt_backbone_config.get_multimodal_representation_from_preprocessed(
                    self.vilt,
                    batch,
                )
            )
        }

    def _shared_eval_step(self, batch: BatchType):
        """
        Perform a shared evaluation step.

        :param batch: The batch.
        :return: The loss and logits.
        """
        embedding = self._get_multimodal_embedding(batch)
        logits = self.classifier(embedding["multimodal_emb"])
        loss = self.loss(logits, batch["answer_label"])
        return {
            "loss": loss,
            "logits": logits,
        }
