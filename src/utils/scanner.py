import cv2
import numpy as np
from .geometry import order_corners


def scan_document(img, debug=False):
    orig_h, orig_w = img.shape[:2]
    PROC_W = 800
    scale = PROC_W / orig_w
    proc = cv2.resize(img, (PROC_W, int(orig_h * scale)))
    ph, pw = proc.shape[:2]

    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median = np.median(blurred)
    low = int(max(0, 0.33 * median))
    high = int(min(255, 1.30 * median))
    edges = cv2.Canny(blurred, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    if debug:
        from .visualization import show
        show(edges, f"Edges (Canny {low}/{high})")

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = pw * ph * 0.01
    doc_cnt = None

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is None:
        print("scan_document: no document contour found.")
        return None, None

    pts = doc_cnt.reshape(4, 2).astype(np.float32)
    c_proc = order_corners(pts)
    c_orig = c_proc / scale

    if debug:
        v = proc.copy()
        for pt in c_proc.astype(int):
            cv2.circle(v, tuple(pt), 10, (0, 255, 0), -1)
        cv2.polylines(v, [c_proc.astype(np.int32)], True, (0, 255, 100), 2)
        show(v, "Detected corners")

    tl, tr, br, bl = c_orig
    ow = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    oh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    dst = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])
    H = cv2.getPerspectiveTransform(c_orig, dst)
    s_color = cv2.warpPerspective(img, H, (ow, oh))
    s_gray = cv2.cvtColor(s_color, cv2.COLOR_BGR2GRAY)
    s_clean = cv2.adaptiveThreshold(
        s_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7
    )
    return s_color, s_clean
