import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from config import CONFIG
from enhancement import apply_enhancement, ENHANCEMENT_METHODS
from detection import (
    detect_edges, detect_document, apply_perspective_correction,
    DETECTION_STRATEGIES
)
from cleaning import binarize_document, apply_cleaning_pipeline
from decision import (
    analyze_image_quality, compute_overall_quality,
    make_decision
)
from visualization import (
    save_stage_image, create_detailed_report,
    visualize_corner_detection
)


@dataclass
class PipelineResult:
    success: bool
    stage_images: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    quality_score: float
    decision_grade: str
    decision_reason: str
    detection_strategy_used: str
    enhancement_method_used: str
    error: Optional[str] = None

    @property
    def decision(self) -> Tuple[str, str]:
        return self.decision_grade, self.decision_reason


def run_pipeline(image_path: str,
                 enhancement: str = "clahe",
                 detection_strategy: str = "contour",
                 upscale: float = 2.0,
                 output_dir: str = "output",
                 save_images: bool = True) -> PipelineResult:
    stage_images: Dict[str, np.ndarray] = {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return PipelineResult(
                success=False, stage_images={}, metrics={},
                quality_score=0, decision_grade="FAIL",
                decision_reason=f"Cannot load image: {image_path}",
                detection_strategy_used="none",
                enhancement_method_used=enhancement,
                error=f"Cannot load image: {image_path}"
            )
        stage_images["original"] = img.copy()

        enhanced = apply_enhancement(img, method=enhancement, upscale=upscale)
        stage_images["enhanced"] = enhanced.copy()

        eh, ew = enhanced.shape[:2]
        proc_w = CONFIG.proc_width
        scale = proc_w / ew if ew > proc_w else 1.0
        if scale != 1.0:
            proc = cv2.resize(enhanced, (proc_w, int(eh * scale)))
        else:
            proc = enhanced.copy()

        edges_img = detect_edges(
            proc,
            low_ratio=CONFIG.canny_low_ratio,
            high_ratio=CONFIG.canny_high_ratio,
            morph_kernel_size=CONFIG.morph_kernel_size,
        )
        stage_images["edges"] = edges_img.copy()

        if detection_strategy == "auto":
            strategy = "all"
        else:
            strategy = detection_strategy

        ordered_corners, raw_contour, used_strategy = detect_document(
            proc, edges_img,
            strategy=strategy,
            proc_scale=scale,
        )

        if ordered_corners is None:
            return PipelineResult(
                success=False, stage_images=stage_images,
                metrics={}, quality_score=0,
                decision_grade="FAIL",
                decision_reason="No document detected in image",
                detection_strategy_used=used_strategy,
                enhancement_method_used=enhancement,
            )

        corrected = apply_perspective_correction(enhanced, ordered_corners)
        stage_images["detected_document"] = corrected.copy()

        binary = binarize_document(
            corrected,
            method="adaptive",
            blocksize=CONFIG.adaptive_blocksize,
            c=CONFIG.adaptive_c,
        )
        stage_images["thresholded"] = binary.copy()

        cleaned, clean_steps = apply_cleaning_pipeline(binary)
        stage_images["cleaned_mask"] = cleaned.copy()

        metrics = analyze_image_quality(corrected, binary)
        quality_score = compute_overall_quality(metrics)
        grade, reason = make_decision(quality_score)

        result = PipelineResult(
            success=True,
            stage_images=stage_images,
            metrics=metrics,
            quality_score=quality_score,
            decision_grade=grade,
            decision_reason=reason,
            detection_strategy_used=used_strategy,
            enhancement_method_used=enhancement,
        )

        if save_images:
            save_stage_image(enhanced, "enhanced", output_dir)
            save_stage_image(edges_img, "edges", output_dir)
            save_stage_image(corrected, "detected_document", output_dir)
            save_stage_image(binary, "thresholded", output_dir)
            save_stage_image(cleaned, "cleaned_mask", output_dir)
            visualize_corner_detection(proc, ordered_corners * scale,
                                       output_dir)
            create_detailed_report(
                stage_images, metrics, result.decision, output_dir
            )

        return result

    except Exception as e:
        return PipelineResult(
            success=False, stage_images=stage_images,
            metrics={}, quality_score=0,
            decision_grade="FAIL",
            decision_reason=f"Pipeline error: {str(e)}",
            detection_strategy_used=detection_strategy,
            enhancement_method_used=enhancement,
            error=str(e),
        )


def run_comparison(image_path: str,
                   output_dir: str = "output") -> Dict[str, PipelineResult]:
    results = {}
    det_names = list(DETECTION_STRATEGIES.keys()) + ["auto"]

    for enh_name in ENHANCEMENT_METHODS:
        for det_name in det_names:
            combo = f"{enh_name}+{det_name}"
            print(f"  Running: {combo}")
            result = run_pipeline(
                image_path,
                enhancement=enh_name,
                detection_strategy=det_name,
                output_dir=os.path.join(output_dir, combo),
            )
            results[combo] = result

    return results
