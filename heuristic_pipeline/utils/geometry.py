"""utils/geometry.py — Reusable geometric helpers."""
import numpy as np


def centroid(box: np.ndarray) -> np.ndarray:
    """box: [x1, y1, x2, y2] -> (cx, cy)"""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def velocity(positions: list[np.ndarray]) -> np.ndarray:
    """Mean frame-to-frame velocity from a list of 2-D positions."""
    if len(positions) < 2:
        return np.zeros(2)
    diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    return np.mean(diffs, axis=0)
