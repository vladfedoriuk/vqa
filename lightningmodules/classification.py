"""The module contains the base class for classification modules."""
import lightning.pytorch as pl
import torch.optim
import torchmetrics
from torch import nn
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from models.backbones import BackboneConfig


class MultiModalClassificationModule(pl.LightningModule):
    """Base class for classification modules."""

    def __init__(
        self,
        fusion: nn.Module,
        classifier: nn.Module,
        image_encoder: nn.Module,
        image_processor: BaseImageProcessor,
        tokenizer: PreTrainedTokenizer,
        text_encoder: nn.Module,
        image_encoder_config: type[BackboneConfig],
        text_encoder_config: type[BackboneConfig],
        classes_num: int,
    ):
        """
        Initialize the module.

        :param fusion: The fusion.
        :param classifier: The classifier.
        :param image_encoder: The image encoder.
        :param image_processor: The image processor.
        :param tokenizer: The tokenizer.
        :param text_encoder: The text encoder.
        :param image_encoder_config: The image encoder config.
        :param text_encoder_config: The text encoder config.
        :param classes_num: The number of classes.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["image_encoder", "text_encoder"])

        self.fusion = fusion
        self.classifier = classifier
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.classes_num = classes_num

        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.image_encoder_config = image_encoder_config
        self.text_encoder_config = text_encoder_config

        self.loss = torch.nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.classes_num)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.classes_num)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.classes_num)

    def _get_embeddings(self, batch):
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

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        embeddings = self._get_embeddings(batch)
        fused_repr = self.fusion(embeddings["image_emb"], embeddings["text_emb"])
        logits = self.classifier(fused_repr)
        loss = self.loss(logits, batch["answer_label"])
        self.log("train_loss", loss)
        self.train_accuracy(logits, batch["answer_label"])
        self.log("train_acc", self.train_accuracy)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizers.

        :return: The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=1e-4)

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss and logits.
        """
        embeddings = self._get_embeddings(batch)
        fused_repr = self.fusion(embeddings["image_emb"], embeddings["text_emb"])
        logits = self.classifier(fused_repr)
        loss = self.loss(logits, batch["answer_label"])
        # TODO: Log as a dict
        # TODO: Confusion matrix
        self.log("val_loss", loss)
        self.val_accuracy(logits, batch["answer_label"])
        self.log("val_acc", self.val_accuracy)
        return {
            "loss": loss,
            "logits": logits,
        }

    def test_step(self, batch, batch_idx):
        """
        Perform a test step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss and logits.
        """
        embeddings = self._get_embeddings(batch)
        fused_repr = self.fusion(embeddings["image_emb"], embeddings["text_emb"])
        logits = self.classifier(fused_repr)
        loss = self.loss(logits, batch["answer_label"])
        self.log("test_loss", loss)
        self.test_accuracy(logits, batch["answer_label"])
        self.log("test_acc", self.test_accuracy)
        return {
            "loss": loss,
            "logits": logits,
        }
