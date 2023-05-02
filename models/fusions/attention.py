"""
Attention fusion model.

The model is an attention-based fusion model.

See:

- https://github.com/facebookresearch/multimodal
"""
import torch
from torchmultimodal.modules.fusions.attention_fusion import AttentionFusionModule

from models.fusions import AvailableFusionModels, BaseFusionModel, registry


@registry.register(AvailableFusionModels.ATTENTION)
class AttentionFusionModel(BaseFusionModel):
    """Attention fusion model."""

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
        self.attention_fusion = AttentionFusionModule(
            channel_to_encoder_dim={
                "image": self.image_representation_size,
                "text": self.text_representation_size,
            },
            encoding_projection_dim=self.final_representation_size,
        )
        self._initialize_weights()

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        """
        Perform forward pass.

        :param image_emb: The image embedding.
        :param text_emb: The text embedding.
        :return: The logits. Shape: (batch_size, answers_num)
        """
        return self.attention_fusion(
            {"image": image_emb, "text": text_emb},
        )
