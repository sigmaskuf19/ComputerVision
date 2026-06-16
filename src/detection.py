import cv2
import numpy as np
from typing import Optional, Tuple


def detect_edges(img: np.ndarray,
                 low_ratio: float = 0.33,
                 high_ratio: float = 1.30,
                 morph_kernel_size: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = np.median(blurred)
    low = int(max(0, low_ratio * median))
    high = int(min(255, high_ratio * median))
    edges = cv2.Canny(blurred, low, high)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)]
    ])


def detect_by_contour(edges: np.ndarray,
                      min_area_ratio: float = 0.01) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = edges.shape
    min_area = w * h * min_area_ratio

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def detect_by_hough(edges: np.ndarray) -> Optional[np.ndarray]:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                            minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return None

    h, w = edges.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(canvas, (x1, y1), (x2, y2), 255, 2)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def detect_by_edge_analysis(img: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edge_maps = []
    for low, high in [(30, 90), (50, 150), (100, 200)]:
        edges = cv2.Canny(blurred, low, high)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edge_maps.append(edges)

    combined = cv2.bitwise_or(edge_maps[0], edge_maps[1])
    combined = cv2.bitwise_or(combined, edge_maps[2])

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    min_area = w * h * 0.02

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


DETECTION_STRATEGIES = {
    "contour": detect_by_contour,
    "hough": detect_by_hough,
    "edge": detect_by_edge_analysis,
}


def detect_document(img: np.ndarray,
                    edges: np.ndarray,
                    strategy: str = "contour",
                    proc_scale: float = 1.0) -> Tuple[Optional[np.ndarray],
                                                      Optional[np.ndarray],
                                                      str]:
    if strategy == "all":
        strategies_to_try = ["contour", "edge", "hough"]
    else:
        strategies_to_try = [strategy]

    for strat in strategies_to_try:
        if strat not in DETECTION_STRATEGIES:
            continue
        fn = DETECTION_STRATEGIES[strat]
        try:
            if strat == "contour":
                doc_contour = fn(edges)
            elif strat == "hough":
                doc_contour = fn(edges)
            else:
                doc_contour = fn(img)

            if doc_contour is not None:
                pts = doc_contour.reshape(4, 2).astype(np.float32)
                ordered = order_corners(pts)
                if proc_scale != 1.0:
                    ordered = ordered / proc_scale
                return ordered, doc_contour, strat
        except Exception:
            continue

    return None, None, "none"


def apply_perspective_correction(img: np.ndarray,
                                 corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    ow = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    oh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    dst = np.float32([[0, 0], [ow - 1, 0],
                      [ow - 1, oh - 1], [0, oh - 1]])
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, H, (ow, oh))
