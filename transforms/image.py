"""Transforms for image data."""
import kornia.augmentation as K
import PIL.Image
import torch
from torch import nn
from torchvision import transforms as T

from transforms.noop import Noop
from utils.batch import batch_to_device
from utils.torch import DeviceAwareModule
from utils.types import BatchType


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/run_clip.py
class SingleVQAImageAugmentationModule(nn.Module):
    """
    Perform image augmentation for VQA experiments.

    This module is intended to be used in the single image mode.

    These augmentations are quite carefully chosen not to change the
    semantics of the image.

    For example, some questions might ask about the color of the object
    in the image. If we were to change the color of the object, the
    answer would change, but the question would not.

    Another example is that some questions might ask about a relative
    position of two objects in the image. If we were to flip the image
    horizontally, the answer would change, but the question would not.

    The same goes about the questions regarding the objects laying on
    the other objects. If we were to rotate the image, the answer would
    change, but the question would not.

    Intense cropping is also not a good idea, because it might remove
    the object of interest from the image.

    Thus, the augmentations are chosen to be as non-destructive as
    possible.

    Given the rationale above, the augmentations are:

    - Small random rotation.
    - Small random translation.
    - Slight color jitter.
    - Contrast and brightness adjustments.
    - Gaussian blur.
    - Center crop.
    - Resize to the original size.

    Available augmentations are:
    https://pytorch.org/vision/stable/transforms.html
    """

    def __init__(self):
        """Initialize the module."""
        super().__init__()
        self._augmentation = T.RandomOrder(
            [
                T.RandomRotation(degrees=5),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
                T.RandomApply([T.RandomCrop(size=224)], p=0.5),
                T.RandomApply([T.RandomResizedCrop(size=224)], p=0.5),
                T.Resize(size=224),
            ]
        )

    def forward(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Perform image augmentation for VQA experiments.

        :param image: The image.
        :return: The augmented image.
        """
        return self._augmentation(image)


def augment_image_for_vqa(image: PIL.Image.Image):
    """
    Perform image augmentation for VQA experiments.

    These augmentations are quite carefully chosen not to change the
    semantics of the image.

    For example, some questions might ask about the color of the object
    in the image. If we were to change the color of the object, the
    answer would change, but the question would not.

    Another example is that some questions might ask about a relative
    position of two objects in the image. If we were to flip the image
    horizontally, the answer would change, but the question would not.

    The same goes about the questions regarding the objects laying on
    the other objects. If we were to rotate the image, the answer would
    change, but the question would not.

    Intense cropping is also not a good idea, because it might remove
    the object of interest from the image.

    Thus, the augmentations are chosen to be as non-destructive as
    possible.

    Given the rationale above, the augmentations are:

    - Small random rotation.
    - Small random translation.
    - Slight color jitter.
    - Contrast and brightness adjustments.
    - Gaussian blur.
    - Center crop.
    - Resize to the original size.

    Available augmentations are:
    https://pytorch.org/vision/stable/transforms.html

    :param image: The image.
    :return: The augmented image.
    """
    return T.RandomOrder(
        [
            T.RandomRotation(degrees=5),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
            T.RandomApply([T.RandomCrop(size=224)], p=0.5),
            T.RandomApply([T.RandomResizedCrop(size=224)], p=0.5),
            T.Resize(size=224),
        ]
    )(image)


class BatchVQAImageAugmentationModule(DeviceAwareModule):
    """
    Perform image augmentation for VQA experiments.

    The augmentations here are intended to be batch-level augmentations.
    They are applied to the whole batch of images at once. It is
    suggested to run them on GPU as underlying operations are GPU
    accelerated. They use :py:mod:`kornia` library.

    The augmentation accepts the batch of images in the format
    ``[B, C, H, W]`` and returns the augmented batch of images in the
    same format. (The images are tensors, not PIL images.)

    These augmentations are quite carefully chosen not to change the
    semantics of the image.

    For example, some questions might ask about the color of the object
    in the image. If we were to change the color of the object, the
    answer would change, but the question would not.

    Another example is that some questions might ask about a relative
    position of two objects in the image. If we were to flip the image
    horizontally, the answer would change, but the question would not.

    The same goes about the questions regarding the objects laying on
    the other objects. If we were to rotate the image, the answer would
    change, but the question would not.

    Intense cropping is also not a good idea, because it might remove
    the object of interest from the image.

    Thus, the augmentations are chosen to be as non-destructive as
    possible.

    Given the rationale above, the augmentations are:

    - Small random rotation.
    - Small random translation.
    - Slight color jitter.
    - Contrast and brightness adjustments.
    - Gaussian blur.
    - Center crop.
    - Resize to the original size.
    """

    def __init__(self):
        """Initialize the augmentation module."""
        super().__init__()
        self.augmentation = nn.Sequential(
            K.RandomRotation(degrees=5),
            K.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            K.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            K.RandomCrop(size=(224, 224), p=0.5),
            K.RandomResizedCrop(size=(224, 224), p=0.5),
            K.Resize(size=(224, 224)),
        )

    def _handle_images_structure(self, images: list[torch.Tensor] | torch.Tensor):
        """
        Convert list of images to a batch of images if needed.

        Move the batch to the same device as the hidden weight.

        :param images: The images.
        :return: The batch of images.
        """
        if isinstance(images, list):
            images = batch_to_device(torch.stack(images, dim=0), device=self.device)
        else:
            images = batch_to_device(images, device=self.device)
        return images

    @torch.no_grad()
    def forward(self, batch: BatchType):
        """
        Perform image augmentation for VQA experiments.

        :param batch: The batch of images.
        :return: A batch with the augmented images.
        """
        images = batch["pixel_values"]
        images = self._handle_images_structure(images)
        batch["pixel_values"] = self.augmentation(images)
        return batch


def default_image_batch_transforms_factory():
    """
    Create default batch image augmentation transforms.

    :return: The image augmentation transforms.
    """
    return nn.ModuleDict(
        {
            "fit": BatchVQAImageAugmentationModule(),
            "validate": Noop(),
            "test": Noop(),
            "predict": Noop(),
        }
    )


def default_image_single_transforms_factory():
    """
    Create default single image augmentation transforms.

    :return: The image augmentation transforms.
    """
    return nn.ModuleDict(
        {
            "fit": SingleVQAImageAugmentationModule(),
            "validate": Noop(),
            "test": Noop(),
            "predict": Noop(),
        }
    )
