import cv2
import numpy as np
from typing import Tuple


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    result = clahe.apply(gray)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) if len(img.shape) == 3 else result


def apply_gamma_correction(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return cv2.equalizeHist(img)


def apply_denoising(img: np.ndarray, h: float = 10,
                    template_window: int = 7,
                    search_window: int = 21) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h,
                                           template_window, search_window)


def apply_upscale(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


ENHANCEMENT_METHODS = {
    "clahe": apply_clahe,
    "gamma": apply_gamma_correction,
    "hist_eq": apply_histogram_equalization,
    "denoise": apply_denoising,
    "none": lambda img: img,
}


def apply_enhancement(img: np.ndarray, method: str = "clahe",
                      upscale: float = 2.0) -> np.ndarray:
    img = apply_upscale(img, upscale)
    if method in ENHANCEMENT_METHODS:
        img = ENHANCEMENT_METHODS[method](img)
    return img
