"""No-op transform."""
import PIL.Image
import torch


def noop(image: PIL.Image.Image | torch.Tensor):
    """
    No-op transform.

    :param image: The image to transform.
    :return: The same image.
    """
    return image
