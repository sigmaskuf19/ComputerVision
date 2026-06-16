import cv2
import numpy as np
from typing import Tuple


def apply_morphological_cleaning(mask: np.ndarray,
                                 kernel_size: int = 3,
                                 close_iterations: int = 1,
                                 open_iterations: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))

    if close_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=close_iterations)
    if open_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=open_iterations)

    return mask


def apply_cleaning_pipeline(mask: np.ndarray) -> Tuple[np.ndarray, dict]:
    steps = {}

    closed = apply_morphological_cleaning(mask, close_iterations=1,
                                          open_iterations=0)
    steps["after_close"] = closed

    opened = apply_morphological_cleaning(closed, close_iterations=0,
                                          open_iterations=1)
    steps["after_open"] = opened

    cleaned = apply_morphological_cleaning(opened, close_iterations=1,
                                           open_iterations=0)
    steps["after_final_close"] = cleaned

    return cleaned, steps


def binarize_document(doc_img: np.ndarray,
                      method: str = "adaptive",
                      blocksize: int = 11,
                      c: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(doc_img, cv2.COLOR_BGR2GRAY) if len(doc_img.shape) == 3 else doc_img

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, blocksize, c)
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "binary":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = gray.copy()

    return binary
