import numpy as np


def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)],
                     pts[np.argmin(d)],
                     pts[np.argmax(s)],
                     pts[np.argmax(d)]])


def dist(a, b):
    return np.linalg.norm(b - a)
