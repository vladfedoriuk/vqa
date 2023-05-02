"""
Simple concatenation fusion model.

The model is a simple concatenation fusion model.

See:

- https://github.com/tezansahu/VQA-With-Multimodal-Transformers/
- https://github.com/SatyamGaba/visual_question_answering
"""
import torch
from torch import nn

from models.fusions import AvailableFusionModels, BaseFusionModel, registry


@registry.register(AvailableFusionModels.CAT)
class ConcatenationFusionModel(BaseFusionModel):
    """Simple concatenation fusion model."""

    def __init__(
        self,
        image_representation_size: int,
        text_representation_size: int,
        final_representation_size: int,
    ):
        """
        Initialize the model.

        :param image_representation_size: The image representation size.
        :param text_representation_size: The text representation size.
        :param final_representation_size: The final representation size.
        """
        super().__init__(
            image_representation_size=image_representation_size,
            text_representation_size=text_representation_size,
            final_representation_size=final_representation_size,
        )
        self.vision_projection = nn.Sequential(
            nn.Linear(self.image_representation_size, 768),
            nn.ReLU(),
            nn.LayerNorm(self.final_representation_size),
            nn.Dropout(0.3),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_representation_size, 768),
            nn.ReLU(),
            nn.LayerNorm(self.final_representation_size),
            nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.final_representation_size * 2, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.3),
        )
        self._initialize_weights()

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        """
        Perform forward pass.

        :param image_emb: The image embedding.
        :param text_emb: The text embedding.
        :return: The logits. Shape: (batch_size, answers_num)
        """
        return self.fusion(
            torch.cat(
                [self.vision_projection(image_emb), self.text_projection(text_emb)],
                dim=-1,
            )
        )
