"""The VQA V2 collator."""
import dataclasses

from collators import ClassificationCollator, VQACollatorMixin, registry
from utils.datasets import AvailableDatasets
from utils.datasets.vqa_v2 import VqaV2SampleAnswerSpace


@registry.register(AvailableDatasets.VQA_V2)
@registry.register(AvailableDatasets.VQA_V2_SAMPLE)
@dataclasses.dataclass(frozen=True)
class VqaV2Collator(VQACollatorMixin, ClassificationCollator):
    """The VQA V2 collator."""

    #: The answer space.
    answer_space: VqaV2SampleAnswerSpace = dataclasses.field(default_factory=VqaV2SampleAnswerSpace)

    ANSWER_BATCH_PROPERTY = "multiple_choice_answer"
