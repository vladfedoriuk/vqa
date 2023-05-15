"""The VQA V2 collator."""
import dataclasses

from collators import ClassificationCollator, VQAClassificationCollatorMixin, registry
from utils.datasets import AvailableDatasets
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace


@registry.register(AvailableDatasets.VQA_V2)
@registry.register(AvailableDatasets.VQA_V2_SAMPLE)
@dataclasses.dataclass
class VqaV2ClassificationCollator(VQAClassificationCollatorMixin, ClassificationCollator):
    """The VQA V2 collator."""

    #: The answer space.
    answer_space: VqaV2SampleAnswerSpace = dataclasses.field(default_factory=VqaV2SampleAnswerSpace)

    ANSWER_BATCH_PROPERTY = "multiple_choice_answer"
