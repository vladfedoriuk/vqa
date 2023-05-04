"""VQA V2 dataset configuration."""
from pathlib import Path
from typing import Final

VQA_V2_PROCESSED_PATH: Final[Path] = Path(__file__).parent.parent.parent / "data" / "processed" / "vqa_v2"

VQA_V2_SAMPLE_ANSWERS_SPACE_PATH: Final[Path] = VQA_V2_PROCESSED_PATH / "vqa_v2_sample_answers_space.json"

VQA_V2_RAW_PATH: Final[Path] = Path(__file__).parent.parent.parent / "data" / "raw" / "vqa_v2"

VQA_V2_RAW_ANSWERS_SPACE_PATH: Final[Path] = VQA_V2_RAW_PATH / "vqa_v2_answers_space_raw.json"

VQA_V2_ANSWERS_SPACE_EXAMPLE_PATH = VQA_V2_RAW_PATH / "examples" / "vqa_v2_answers_space_raw_example.json"

VQA_V2_ANSWERS_SPACE_PATH = VQA_V2_PROCESSED_PATH / "vqa_v2_answers_space.json"
