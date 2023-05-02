"""
DeepSet fusion model.

The model is a DeepSet-based fusion model.
It either uses pooling / attention of self-attention (a transformer) to aggregate the features.

See:

- https://github.com/facebookresearch/multimodal
- https://arxiv.org/pdf/2003.01607.pdf
"""
import torch
from torch import nn
from torchmultimodal.modules.fusions.deepset_fusion import (
    DeepsetFusionModule,
    deepset_transformer,
)
from torchmultimodal.modules.layers.mlp import MLP

from models.fusions import AvailableFusionModels, BaseFusionModel, registry


@registry.register(AvailableFusionModels.DEEP_SET)
class DeepSetFusionModel(BaseFusionModel):
    """DeepSet fusion model."""

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
        self.deep_set_fusion = DeepsetFusionModule(
            channel_to_encoder_dim={
                "image": self.image_representation_size,
                "text": self.text_representation_size,
            },
            mlp=MLP(
                in_dim=self.final_representation_size,
                out_dim=self.final_representation_size,
                normalization=nn.BatchNorm1d,
            ),
            pooling_function=torch.sum,
            modality_normalize=True,
            apply_attention=True,
        )
        self._initialize_weights()

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        """
        Perform forward pass.

        :param image_emb: The image embedding.
        :param text_emb: The text embedding.
        :return: The logits. Shape: (batch_size, answers_num)
        """
        return self.deep_set_fusion(
            {
                "image": image_emb,
                "text": text_emb,
            },
        )


@registry.register(AvailableFusionModels.DEEP_SET_TRANSFORMER)
class DeepSetTransformerFusionModel(BaseFusionModel):
    """DeepSet transformer fusion model."""

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
        self.deep_set_fusion = deepset_transformer(
            channel_to_encoder_dim={
                "image": self.image_representation_size,
                "text": self.text_representation_size,
            },
            mlp=MLP(
                in_dim=self.final_representation_size,
                out_dim=self.final_representation_size,
                normalization=nn.BatchNorm1d,
            ),
            apply_attention=False,
            attention_dim=None,
            modality_normalize=True,
            use_auto_mapping=False,
            num_transformer_layers=12,
        )
        self._initialize_weights()

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        """
        Perform forward pass.

        :param image_emb: The image embedding.
        :param text_emb: The text embedding.
        :return: The logits. Shape: (batch_size, answers_num)
        """
        return self.deep_set_fusion(
            {
                "image": image_emb,
                "text": text_emb,
            },
        )
