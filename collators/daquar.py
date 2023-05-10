"""The Daquar collator."""
import dataclasses

from collators import ClassificationCollator, VQACollatorMixin, registry
from utils.datasets import AvailableDatasets
from utils.datasets.daquar import DaquarAnswerSpace, load_images_for_batch
from utils.torch import expand_first_dim_dict_of_tensors, squeeze_dict_of_tensors


@registry.register(AvailableDatasets.DAQUAR)
@dataclasses.dataclass(frozen=True)
class DaquarCollator(VQACollatorMixin, ClassificationCollator):
    """The Daquar collator."""

    #: The answer space.
    answer_space: DaquarAnswerSpace = dataclasses.field(default_factory=DaquarAnswerSpace)

    @staticmethod
    def batch_len(batch):
        """
        Get the batch length.

        :param batch: The batch.
        :return: The batch length.
        """
        return len(batch["image_id"])

    def get_image_features(self, batch):
        """
        Get the image features.

        :param batch: The batch.
        :return: The image features.
        """
        # TODO: do we really need squeeze_dict_of_tensors and expand_first_dim_dict_of_tensors in collators?
        images = load_images_for_batch(batch)
        features = self.image_encoder_config.get_processed_image(
            self.image_processor,
            image=[self.single_image_transforms(image) for image in images],
        )
        features = squeeze_dict_of_tensors(self.batch_image_transforms(features))
        features[self.IMAGE_BATCH_PROPERTY] = images
        if self.batch_len(batch) == 1:
            return expand_first_dim_dict_of_tensors(features)
        return features

    def get_text_features(self, batch):
        """
        Get the text features.

        :param batch: The batch.
        :return: The text features.
        """
        features = super().get_text_features(batch)
        if self.batch_len(batch) == 1:
            return expand_first_dim_dict_of_tensors(features)
        return features
