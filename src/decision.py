import cv2
import numpy as np
from typing import Dict, Tuple, Optional


def compute_blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_contrast_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return gray.std()


def compute_brightness_score(img: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(gray.min()), float(gray.max())


def compute_text_coverage(binary_img: np.ndarray) -> float:
    black_pixels = np.sum(binary_img == 0)
    total_pixels = binary_img.size
    return (black_pixels / total_pixels) * 100


def compute_sharpness_metric(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    return float(magnitude.mean())


def analyze_image_quality(img: np.ndarray,
                          binary: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {}
    metrics["blur_score"] = compute_blur_score(img)
    metrics["contrast_score"] = compute_contrast_score(img)
    min_b, max_b = compute_brightness_score(img)
    metrics["brightness_min"] = min_b
    metrics["brightness_max"] = max_b
    metrics["sharpness"] = compute_sharpness_metric(img)

    if binary is not None:
        metrics["text_coverage_pct"] = compute_text_coverage(binary)

    return metrics


def compute_overall_quality(metrics: Dict[str, float],
                            config: 'PipelineConfig' = None) -> float:
    if config is None:
        from config import CONFIG as config

    score = 100.0
    blur = metrics.get("blur_score", 200)
    if blur < config.quality_blur_threshold:
        score -= 20
    if blur < config.quality_blur_threshold * 0.5:
        score -= 15

    contrast = metrics.get("contrast_score", 60)
    if contrast < config.quality_contrast_threshold:
        score -= 15

    min_b = metrics.get("brightness_min", 0)
    max_b = metrics.get("brightness_max", 255)
    if min_b < config.quality_brightness_low:
        score -= 10
    if max_b > config.quality_brightness_high:
        pass
    if max_b - min_b < 100:
        score -= 10

    coverage = metrics.get("text_coverage_pct", 0)
    if coverage < 1.0:
        score -= 10
    elif coverage > 40:
        score -= 5

    return max(0.0, min(100.0, score))


def make_decision(quality_score: float,
                  ocr_confidence: Optional[float] = None,
                  config: 'PipelineConfig' = None) -> Tuple[str, str]:
    if config is None:
        from config import CONFIG as config

    reasons = []
    if quality_score >= 80:
        grade = "PASS"
        reasons.append("High quality document scan")
    elif quality_score >= 50:
        grade = "PASS (with warnings)"
        if quality_score < 80:
            reasons.append("Moderate quality")
    else:
        grade = "FAIL"
        reasons.append("Low quality document scan")

    if ocr_confidence is not None:
        if ocr_confidence >= config.min_confidence_for_pass:
            reasons.append(f"OCR confidence: {ocr_confidence:.1f}%")
        else:
            reasons.append(f"Low OCR confidence: {ocr_confidence:.1f}%")
            if grade.startswith("PASS"):
                grade = "PASS (with warnings)"

    reason_str = "; ".join(reasons)
    return grade, reason_str
