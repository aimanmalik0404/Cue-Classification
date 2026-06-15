"""
core/keypoint_utils.py
Utility functions that operate on a single person's keypoints array.
Shape expected: (17, 3) → (x, y, confidence)
"""

import numpy as np
from ..config import KP, MIN_KP_CONF


# ─── Basic helpers ────────────────────────────────────────────────────────────

def kp_valid(kp_array: np.ndarray, idx: int) -> bool:
    """Return True if keypoint at `idx` has confidence above threshold."""
    return kp_array[idx, 2] >= MIN_KP_CONF


def kp_xy(kp_array: np.ndarray, idx: int):
    """Return (x, y) for keypoint `idx`. No confidence check."""
    return float(kp_array[idx, 0]), float(kp_array[idx, 1])


def midpoint_xy(kp_array: np.ndarray, idx_a: int, idx_b: int):
    """Return midpoint (x, y) between two keypoints."""
    ax, ay = kp_xy(kp_array, idx_a)
    bx, by = kp_xy(kp_array, idx_b)
    return (ax + bx) / 2, (ay + by) / 2


def valid_pair(kp_array: np.ndarray, idx_a: int, idx_b: int) -> bool:
    return kp_valid(kp_array, idx_a) and kp_valid(kp_array, idx_b)


# ─── Core geometry ────────────────────────────────────────────────────────────

def body_angle_from_vertical(kp_array: np.ndarray):
    """
    Compute angle (degrees) between the torso spine vector and vertical axis.
    0° = perfectly upright, 90° = perfectly horizontal (lying flat).
    Returns None if key joints are missing.
    """
    ls, rs = KP["LEFT_SHOULDER"], KP["RIGHT_SHOULDER"]
    lh, rh = KP["LEFT_HIP"],      KP["RIGHT_HIP"]

    if not (valid_pair(kp_array, ls, rs) and valid_pair(kp_array, lh, rh)):
        return None

    sx, sy = midpoint_xy(kp_array, ls, rs)
    hx, hy = midpoint_xy(kp_array, lh, rh)

    dx = sx - hx
    dy = sy - hy  # y increases downward in image coords

    angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
    return angle


def lying_score(kp_array: np.ndarray):
    """
    Core lying metric:
        lying_score = shoulder_x_span / (vertical_span + 1)

    A standing person has lying_score ~ 0.2–0.4
    A lying person has lying_score ~ 1.5–5.0+

    Also returns vertical_span and shoulder_x_span for secondary checks.
    Returns (score, vertical_span, shoulder_x_span) or (None, None, None).
    """
    ls, rs = KP["LEFT_SHOULDER"], KP["RIGHT_SHOULDER"]
    lh, rh = KP["LEFT_HIP"],      KP["RIGHT_HIP"]
    la, ra = KP["LEFT_ANKLE"],    KP["RIGHT_ANKLE"]

    # Need at minimum shoulders + hips
    if not (valid_pair(kp_array, ls, rs) and valid_pair(kp_array, lh, rh)):
        return None, None, None

    # Shoulder midpoint Y
    _, sh_y = midpoint_xy(kp_array, ls, rs)
    # Hip midpoint Y
    _, hip_y = midpoint_xy(kp_array, lh, rh)

    # Use ankles if available, else estimate from hip
    if valid_pair(kp_array, la, ra):
        _, ankle_y = midpoint_xy(kp_array, la, ra)
    else:
        # Estimate: ankle is ~50% further below hip than hip is below shoulder
        ankle_y = hip_y + (hip_y - sh_y) * 1.2

    vertical_span    = abs(ankle_y - sh_y)
    shoulder_x_span  = abs(kp_xy(kp_array, ls)[0] - kp_xy(kp_array, rs)[0])
    score            = shoulder_x_span / (vertical_span + 1)

    return score, vertical_span, shoulder_x_span


def all_joints_same_y(kp_array: np.ndarray, frame_height: int, tolerance: float = 0.15) -> bool:
    """
    Check if all major joints (shoulder, hip, knee, ankle) are at roughly
    the same Y level — indicating person is horizontal.
    tolerance: fraction of frame_height allowed as Y spread.
    """
    joints = [
        KP["LEFT_SHOULDER"],  KP["RIGHT_SHOULDER"],
        KP["LEFT_HIP"],       KP["RIGHT_HIP"],
        KP["LEFT_KNEE"],      KP["RIGHT_KNEE"],
        KP["LEFT_ANKLE"],     KP["RIGHT_ANKLE"],
    ]
    valid_ys = [
        kp_array[j, 1] for j in joints if kp_valid(kp_array, j)
    ]
    if len(valid_ys) < 4:
        return False

    y_spread = max(valid_ys) - min(valid_ys)
    return y_spread < tolerance * frame_height


def bbox_aspect_ratio(kp_array: np.ndarray):
    """
    Bounding box of all valid keypoints.
    Returns width/height aspect ratio (> 1 = wide = possibly lying).
    """
    valid_mask = kp_array[:, 2] >= MIN_KP_CONF
    if valid_mask.sum() < 3:
        return None

    xs = kp_array[valid_mask, 0]
    ys = kp_array[valid_mask, 1]
    w  = xs.max() - xs.min()
    h  = ys.max() - ys.min()
    if h < 1:
        return None
    return w / h


def is_arms_raised(kp_array: np.ndarray) -> bool:
    """True if either wrist is above the corresponding shoulder (balance reflex)."""
    ls, rs = KP["LEFT_SHOULDER"], KP["RIGHT_SHOULDER"]
    lw, rw = KP["LEFT_WRIST"],    KP["RIGHT_WRIST"]

    left_raised  = (kp_valid(kp_array, ls) and kp_valid(kp_array, lw)
                    and kp_array[lw, 1] < kp_array[ls, 1])
    right_raised = (kp_valid(kp_array, rs) and kp_valid(kp_array, rw)
                    and kp_array[rw, 1] < kp_array[rs, 1])
    return left_raised or right_raised


def center_of_mass_x(kp_array: np.ndarray):
    """Approximate CoM as midpoint of hips."""
    lh, rh = KP["LEFT_HIP"], KP["RIGHT_HIP"]
    if not valid_pair(kp_array, lh, rh):
        return None
    return (kp_array[lh, 0] + kp_array[rh, 0]) / 2


def support_base_x(kp_array: np.ndarray):
    """Return (min_x, max_x) of ankle support base."""
    la, ra = KP["LEFT_ANKLE"], KP["RIGHT_ANKLE"]
    xs = []
    if kp_valid(kp_array, la):
        xs.append(kp_array[la, 0])
    if kp_valid(kp_array, ra):
        xs.append(kp_array[ra, 0])
    if not xs:
        return None, None
    margin = 25  # pixels
    return min(xs) - margin, max(xs) + margin


# ─── Bending geometry ─────────────────────────────────────────────────────────

def hip_height_ratio(kp_array: np.ndarray, standing_hip_y: float,
                     standing_height: float) -> float:
    """
    How far the hip has dropped compared to its standing position,
    expressed as a fraction of the person's standing height.

    standing_hip_y:   calibrated hip Y when upright (from FrameWindow)
    standing_height:  calibrated vertical span when upright (from FrameWindow)

    Returns:
        0.0  = hip is exactly where it is when standing
        0.10 = hip dropped 10% of body height  (slight squat or bend)
        0.35 = hip dropped 35% of body height  (falling / sitting down fast)
    Returns None if hip keypoints not visible.
    """
    lh, rh = KP["LEFT_HIP"], KP["RIGHT_HIP"]
    if not valid_pair(kp_array, lh, rh):
        return None
    _, current_hip_y = midpoint_xy(kp_array, lh, rh)
    # y increases downward — a drop means current_hip_y > standing_hip_y
    drop = current_hip_y - standing_hip_y
    return drop / (standing_height + 1e-6)


def torso_compression_ratio(kp_array: np.ndarray,
                             standing_torso_length: float) -> float:
    """
    Current torso length (shoulder_mid → hip_mid) as a fraction of the
    calibrated standing torso length.

    1.0  = fully extended (upright or rigid fall)
    0.5  = torso is half as long as usual (bent forward, shoulder near hip)
    0.3  = deeply bent or crouched

    Returns None if key joints not visible.
    """
    ls, rs = KP["LEFT_SHOULDER"], KP["RIGHT_SHOULDER"]
    lh, rh = KP["LEFT_HIP"],      KP["RIGHT_HIP"]
    if not (valid_pair(kp_array, ls, rs) and valid_pair(kp_array, lh, rh)):
        return None

    sx, sy = midpoint_xy(kp_array, ls, rs)
    hx, hy = midpoint_xy(kp_array, lh, rh)

    # Euclidean torso length (not just vertical — handles lateral bending too)
    current_torso = float(np.hypot(sx - hx, sy - hy))
    return current_torso / (standing_torso_length + 1e-6)


def current_torso_length(kp_array: np.ndarray) -> float:
    """
    Raw Euclidean distance from shoulder midpoint to hip midpoint.
    Returns None if joints not visible.
    """
    ls, rs = KP["LEFT_SHOULDER"], KP["RIGHT_SHOULDER"]
    lh, rh = KP["LEFT_HIP"],      KP["RIGHT_HIP"]
    if not (valid_pair(kp_array, ls, rs) and valid_pair(kp_array, lh, rh)):
        return None
    sx, sy = midpoint_xy(kp_array, ls, rs)
    hx, hy = midpoint_xy(kp_array, lh, rh)
    return float(np.hypot(sx - hx, sy - hy))


def current_hip_y(kp_array: np.ndarray) -> float:
    """
    Y coordinate of hip midpoint.  Returns None if not visible.
    """
    lh, rh = KP["LEFT_HIP"], KP["RIGHT_HIP"]
    if not valid_pair(kp_array, lh, rh):
        return None
    _, hy = midpoint_xy(kp_array, lh, rh)
    return hy