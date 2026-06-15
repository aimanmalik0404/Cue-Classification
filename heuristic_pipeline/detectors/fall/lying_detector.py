"""
detectors/lying_detector.py

Two independent paths to LYING_OR_COLLAPSED:

PATH A — Geometry (original):
  Body orientation is horizontal (lying_score, vertical_span, joint_y, aspect_ratio).
  Works when the full or partial body is visible on the floor/ground.

PATH B — Surface rest (bar/cafe addition):
  Head is resting on ANY surface (table, bar counter, chair back, etc.).
  Does NOT require the head to be near the floor.
  Triggered by:
    1. nose_y has dropped to shoulder level or below  (head bowed forward)
    2. nose_y variance stays within the "resting range" — low enough that the
       head is not actively moving, but allows the drunk-wobble movement that
       a conscious resting person naturally has.
    3. Sustained for HEAD_REST_MIN_FRAMES consecutive frames

Either path alone is sufficient to label the person LYING_OR_COLLAPSED.
"""

import numpy as np
from .core.keypoint_utils import (
    lying_score as compute_lying_score,
    all_joints_same_y,
    bbox_aspect_ratio,
    body_angle_from_vertical,
    kp_valid,
    midpoint_xy,
    current_torso_length,
    current_hip_y,
)
from .core.frame_window import FrameWindow
from .config import (
    LYING_SCORE_THRESHOLD,
    LYING_VERTICAL_RATIO,
    LYING_JOINT_Y_TOLERANCE,
    ASPECT_RATIO_LYING,
    KP,
)

# ── Bar/cafe constants (can be moved to config.py) ────────────────────────────

# How many pixels below shoulder_mid_y the nose must reach.
# nose_y >= shoulder_mid_y - TOLERANCE  means head is AT or BELOW shoulder level.
HEAD_BELOW_SHOULDER_TOLERANCE_PX = 10   # px

# ── Head movement thresholds (drunk-aware) ────────────────────────────────────
#
# We use TWO variance measures on nose_y to classify head movement:
#
#   raw_var:        overall spread of nose Y over the window
#   detrended_var:  variance AFTER removing linear drift (residual oscillation)
#
# Categories (approximate, at 30fps / 15-frame window):
#
#   State                    raw_var    detrended_var
#   ─────────────────────── ─────────  ──────────────
#   Dead still (unconscious)   < 2         < 2
#   Slight drunk sway          2 – 10      2 – 10
#   Moderate drunk bobbing    10 – 80     10 – 80      ← WANT to catch
#   Heavy drunk wobbling      80 – 200    80 – 200     ← WANT to catch
#   Slow deliberate head turn 50 – 200     0 – 10      ← must REJECT (drift, not rest)
#   Active nodding            200 – 500  200 – 500     ← must REJECT
#   Looking around            500+        300+         ← must REJECT
#
# Rule:
#   head_is_still = (
#       raw_var        < HEAD_RAW_VAR_MAX          # not wildly flailing
#       AND detrended_var < HEAD_DETRENDED_VAR_MAX  # oscillating near a point
#   )
#
# The detrended check is what catches "slow turn" (detrended ≈ 0) as NOT resting,
# even though its raw_var looks similar to heavy drunk bobbing.
# Wait — that's wrong direction: slow turn has LOW detrended var.
# So we need an additional guard: if detrended_var is suspiciously LOW but
# raw_var is medium, it's a slow turn → reject via HEAD_DRIFT_MAX.
#
#   net_drift = |nose_y[latest] - nose_y[oldest]|
#   if net_drift > HEAD_DRIFT_MAX → head is moving away, not resting
#
# Raw variance ceiling — head flailing above this is definitely not resting
HEAD_RAW_VAR_MAX = 200.0         # px²

# Detrended variance ceiling — filters out vigorous active nodding
HEAD_DETRENDED_VAR_MAX = 150.0   # px²

# Direction change minimum — the key drunk-motion discriminator:
#   Drunk resting: head oscillates back and forth → many direction reversals (≥ 4)
#   Slow head turn: head drifts steadily one way → almost zero reversals (0–2)
#   This catches slow turns that look like resting on raw/detrended variance alone.
HEAD_DIR_CHANGES_MIN  = 3        # fewer than this = sustained drift, not rest

# How many consecutive frames the surface-rest conditions must hold.
HEAD_REST_MIN_FRAMES = 10


class LyingDetector:
    """
    Stateless per-call detector — all state lives in the FrameWindow.
    Call detect(window, keypoints, frame_height) each frame.
    """

    def detect(
        self,
        window: FrameWindow,
        keypoints: np.ndarray,
        frame_height: int,
    ) -> tuple[bool, dict]:
        """
        Returns (is_lying, evidence_dict).
        """
        # ── Shared raw metrics ────────────────────────────────────────────────
        score, v_span, sh_x_span = compute_lying_score(keypoints)
        angle                    = body_angle_from_vertical(keypoints)
        ar                       = bbox_aspect_ratio(keypoints)

        # ── Nose position ─────────────────────────────────────────────────────
        nose_idx = KP["NOSE"]
        nose_y   = float(keypoints[nose_idx, 1]) if kp_valid(keypoints, nose_idx) else None

        # Shoulder midpoint Y
        ls, rs = KP["LEFT_SHOULDER"], KP["RIGHT_SHOULDER"]
        if kp_valid(keypoints, ls) and kp_valid(keypoints, rs):
            _, shoulder_mid_y = midpoint_xy(keypoints, ls, rs)
        else:
            shoulder_mid_y = None

        evidence = {
            # Geometry path
            "lying_score":           score,
            "vertical_span":         v_span,
            "shoulder_x_span":       sh_x_span,
            "angle":                 angle,
            "aspect_ratio":          ar,
            "signal_score_pass":     False,
            "signal_vspan_pass":     False,
            "signal_joints_pass":    False,
            "signal_ar_pass":        False,
            # Surface-rest path
            "nose_y":                nose_y,
            "shoulder_mid_y":        shoulder_mid_y,
            "head_below_shoulder":   False,
            "head_raw_var":          None,
            "head_detrended_var":    None,
            "head_dir_changes":      None,
            "head_is_still":         False,
            "head_on_surface":       False,
            # Combined
            "path":                  "none",   # "geometry" | "surface_rest" | "both"
            "sustained":             False,
        }

        # ═══════════════════════════════════════════════════════════════════════
        # PATH B — Surface rest  (evaluated first, independent of geometry)
        # ═══════════════════════════════════════════════════════════════════════
        head_on_surface = False

        if nose_y is not None and shoulder_mid_y is not None:
            # Condition 1: nose has dropped to/below shoulder level
            head_below = nose_y >= (shoulder_mid_y - HEAD_BELOW_SHOULDER_TOLERANCE_PX)
            evidence["head_below_shoulder"] = head_below

            # Condition 2: head movement is in the "resting range"
            # Allows drunk wobble; rejects active nodding, head turns, looking around.
            # Note: nose_y is pushed into window history via the metrics push below,
            # so the variance here reflects the window BEFORE this frame's nose_y.
            # That slight lag is acceptable — sustained check covers it.
            raw_var, det_var, dir_changes = window.head_stillness_variance()
            evidence["head_raw_var"]       = raw_var
            evidence["head_detrended_var"] = det_var
            evidence["head_dir_changes"]   = dir_changes

            # head_is_still = head is in "resting range":
            #   raw_var < ceiling          → not flailing or actively nodding
            #   detrended_var < ceiling    → residual oscillation within bounds
            #   dir_changes >= minimum     → oscillating (drunk rest), not drifting
            #                                (a slow head turn has dir_changes ≈ 0)
            head_still = (
                raw_var     is not None
                and raw_var     < HEAD_RAW_VAR_MAX
                and det_var     < HEAD_DETRENDED_VAR_MAX
                and dir_changes >= HEAD_DIR_CHANGES_MIN
            )
            evidence["head_is_still"] = head_still

            head_on_surface = head_below and head_still

        evidence["head_on_surface"] = head_on_surface

        # ═══════════════════════════════════════════════════════════════════════
        # PATH A — Geometry  (original body-orientation signals)
        # ═══════════════════════════════════════════════════════════════════════
        geo_positive = False

        if score is not None:
            signal_score  = score >= LYING_SCORE_THRESHOLD
            standing_h    = window.effective_standing_height(frame_height)
            signal_vspan  = (v_span is not None and
                             v_span < LYING_VERTICAL_RATIO * standing_h)
            signal_joints = all_joints_same_y(keypoints, frame_height,
                                              tolerance=LYING_JOINT_Y_TOLERANCE)
            signal_ar     = (ar is not None and ar >= ASPECT_RATIO_LYING)

            evidence["signal_score_pass"]  = signal_score
            evidence["signal_vspan_pass"]  = signal_vspan
            evidence["signal_joints_pass"] = signal_joints
            evidence["signal_ar_pass"]     = signal_ar

            secondary_count = sum([signal_vspan, signal_joints, signal_ar])
            geo_positive    = signal_score and secondary_count >= 1

        # ── Push all metrics to window ────────────────────────────────────────
        window.push(keypoints, {
            "angle":           angle,
            "lying_score":     score,
            "vertical_span":   v_span,
            "shoulder_x_span": sh_x_span,
            "aspect_ratio":    ar,
            "nose_y":          nose_y,
            "head_on_surface": head_on_surface,
            # Bending calibration — recorded by FrameWindow when upright
            "hip_y":           current_hip_y(keypoints),
            "torso_length":    current_torso_length(keypoints),
        })

        # ── Temporal: sustained across recent frames ──────────────────────────
        # sustained_lying() now accepts EITHER geo_lying OR head_on_surface per frame
        sustained = window.sustained_lying()   # uses FPS-calibrated frame count
        evidence["sustained"] = sustained

        # ── Final decision ────────────────────────────────────────────────────
        # Either path triggers the label.  Both paths require temporal stability.
        if geo_positive and head_on_surface:
            evidence["path"] = "both"
        elif geo_positive:
            evidence["path"] = "geometry"
        elif head_on_surface:
            evidence["path"] = "surface_rest"

        is_lying = (geo_positive or head_on_surface) and sustained

        return is_lying, evidence
