"""The module contains the base class for classification modules."""
import lightning.pytorch as pl
import torch.optim
import torchmetrics
from torch import nn


class MultiModalClassificationModule(pl.LightningModule):
    """Base class for classification modules."""

    def __init__(
        self, classifier: nn.Module, image_encoder: nn.Module, text_encoder: nn.Module
    ):
        """
        Initialize the module.

        .. note::
            The classifier must have an attribute ``answers_num``
            that contains the number of answers.
            The classifier must have a method ``forward``
            that takes two arguments: ``image_emb`` and ``text_emb``.

        :param classifier: The classifier.
        :param image_encoder: The image encoder.
        :param text_encoder: The text encoder.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["classifier", "image_encoder", "text_encoder"]
        )

        self.classifier = classifier
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.loss = torch.nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.classifier.answers_num
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.classifier.answers_num
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.classifier.answers_num
        )

    def _get_embeddings(self, batch):
        """
        Get the embeddings.

        :param batch: The batch.
        :return: The embeddings.
        """
        image_emb = self.image_encoder(pixel_values=batch["pixel_values"])
        text_emb = self.text_encoder(
            input_ids=torch.tensor(batch["input_ids"]),
            token_type_ids=torch.tensor(batch["token_type_ids"]),
            attention_mask=torch.tensor(batch["attention_mask"]),
        )
        return {"image_emb": image_emb, "text_emb": text_emb}

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        embeddings = self._get_embeddings(batch)
        logits = self.classifier(embeddings["image_emb"], embeddings["text_emb"])
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
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss and logits.
        """
        embeddings = self._get_embeddings(batch)
        logits = self.classifier(embeddings["image_emb"], embeddings["text_emb"])
        loss = self.loss(logits, batch["answer_label"])
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
        logits = self.classifier(embeddings["image_emb"], embeddings["text_emb"])
        loss = self.loss(logits, batch["answer_label"])
        self.log("test_loss", loss)
        self.test_accuracy(logits, batch["answer_label"])
        self.log("test_acc", self.test_accuracy)
        return {
            "loss": loss,
            "logits": logits,
        }
