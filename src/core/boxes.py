import numpy as np
from .types import BBox


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [x1,y1,x2,y2]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    M = np.zeros((len(a), len(b)), dtype=float)
    for i in range(len(a)):
        for j in range(len(b)):
            M[i, j] = iou_xyxy(a[i], b[j])
    return M