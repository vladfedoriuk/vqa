"""Simple concatenation fusion model."""
import torch
from torch import nn


class SimpleCatFusionModel(nn.Module):
    """Simple concatenation fusion model."""

    def __init__(self, answers_num: int):
        """
        Initialize the model.

        :param answers_num: The number of answers.
        """
        super().__init__()
        self.answers_num = answers_num
        self.vision_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.3),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, self.answers_num),
        )

    def forward(self, image_emb, text_emb):
        """
        Perform forward pass.

        :param image_emb: The image embedding.
                            Shape: (batch_size, 768)
        :param text_emb: The text embedding.
                            Shape: (batch_size, 768)
        :return: The logits. Shape: (batch_size, answers_num)
        """
        return self.classifier(
            self.fusion(
                torch.cat(
                    [self.vision_projection(image_emb), self.text_projection(text_emb)],
                    dim=-1,
                )
            )
        )
