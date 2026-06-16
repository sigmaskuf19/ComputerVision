from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PipelineConfig:
    enhancement_methods: List[str] = field(default_factory=lambda: [
        "clahe", "gamma", "hist_eq", "denoise", "none"
    ])
    default_enhancement: str = "clahe"
    upscale_factor: float = 2.0
    canny_low_ratio: float = 0.33
    canny_high_ratio: float = 1.30
    morph_kernel_size: int = 3
    morph_close_iterations: int = 1
    adaptive_blocksize: int = 11
    adaptive_c: int = 7
    proc_width: int = 800

    detection_strategies: List[str] = field(default_factory=lambda: [
        "contour", "hough", "edge"
    ])
    default_detection: str = "contour"

    quality_blur_threshold: float = 100.0
    quality_contrast_threshold: float = 30.0
    quality_brightness_low: float = 30.0
    quality_brightness_high: float = 225.0
    min_confidence_for_pass: float = 30.0

    output_dir: str = "output"
    save_intermediates: bool = True


CONFIG = PipelineConfig()

STAGE_NAMES = [
    "original",
    "enhanced",
    "edges",
    "cleaned_mask",
    "detected_document",
    "thresholded",
    "final_decision"
]
