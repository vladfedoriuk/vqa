"""Transforms for VQA V2."""
import PIL.Image


def image_augmentation_for_vqa_v2(image: PIL.Image.Image):
    """
    Perform image augmentation for VQA V2.

    Applies the following transformations:

    - Random rotation
    - Random horizontal flip
    - Random vertical flip
    - Random color jitter

    :param image: The image to augment.
    :return: The augmented image.
    """
    import torchvision.transforms as T

    return T.Compose(
        [
            T.RandomRotation(degrees=30),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ]
    )(image)
