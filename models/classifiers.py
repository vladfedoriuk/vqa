"""The available classifiers."""
from torch import nn
from torchmultimodal.modules.layers.mlp import MLP

from utils.torch import initialize_linear_weights


def default_classifier_factory(
    input_dim: int,
    classes_num: int,
) -> nn.Module:
    """
    Create a default classifier.

    :param classes_num: The number of classes.
    :param input_dim: The input dimensionality.
    :return: The classifier.
    """
    classifier = MLP(
        in_dim=input_dim,
        hidden_dims=[512, 256],
        out_dim=classes_num,
        dropout=0.3,
        activation=nn.ReLU,
        normalization=nn.BatchNorm1d,
    )

    # Initialize the weights
    initialize_linear_weights(classifier)
    return classifier
