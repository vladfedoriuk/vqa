"""VQA V2 dataset configuration."""
from pathlib import Path
from typing import Final

VQA_V2_PROCESSED_PATH: Final[Path] = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "vqa_v2"
)

VQA_V2_SAMPLE_ANSWERS_SPACE_PATH: Final[Path] = (
    VQA_V2_PROCESSED_PATH / "vqa_v2_sample_answers_space.json"
)
