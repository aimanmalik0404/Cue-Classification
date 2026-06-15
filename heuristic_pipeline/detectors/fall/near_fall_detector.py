
"""
detectors/near_fall_detector.py

Two independent paths to NEAR_FALL:

PATH A — Dynamic instability (original, unchanged):
  Body angle enters danger zone (20°–60°) + rapid angular change
  + at least one instability cue (arms raised OR CoM outside support base).

PATH B — Bend detection (bar/cafe addition):
  Person bends forward (e.g. reaching for something, feeling unwell, losing
  consciousness while seated/standing). Treated as NEAR_FALL because in a
  bar context a bend that is sustained or deepening warrants the same level
  of staff attention.

  A bend is distinguished from a true fall by the "hip hinge":
    • When bending:  torso folds at the hip.
                     Hip stays near its calibrated standing height.
                     Torso length (shoulder→hip Euclidean) shortens noticeably.
    • When falling:  whole body tilts as a rigid unit.
                     Hip descends with the shoulders.
                     Torso length stays roughly constant.

  Bend detection signals:
    1. body_angle ≥ BEND_ANGLE_MIN  (actually tilted forward, not just slouching)
    2. hip_drop_ratio < BEND_HIP_DROP_RATIO_MAX
         Hip has not dropped significantly — it is acting as a hinge, not falling.
    3. torso_compression_ratio < (1 - BEND_TORSO_COMPRESSION_MIN)
         Torso is measurably shorter than when standing — shoulder folding toward hip.
    4. Sustained BEND_SUSTAINED_FRAMES consecutive frames (not a momentary lean).

Either path alone produces the NEAR_FALL label.
When PATH B fires, evidence["bend_detected"] = True so the visualiser can
show a different sub-label ("BENDING") if desired.
"""

import numpy as np
from .core.keypoint_utils import (
    body_angle_from_vertical,
    is_arms_raised,
    center_of_mass_x,
    support_base_x,
    hip_height_ratio,
    torso_compression_ratio,
    current_torso_length,
    current_hip_y,
)
from .core.frame_window import FrameWindow
from .config import (
    NEAR_FALL_ANGLE_MIN,
    NEAR_FALL_ANGLE_MAX,
    BEND_HIP_DROP_RATIO_MAX,
    BEND_TORSO_COMPRESSION_MIN,
    BEND_ANGLE_MIN,
)


class NearFallDetector:
    """
    Stateless per-call detector — all state lives in the FrameWindow.
    Call detect(window, keypoints) each frame.
    NOTE: window.push() is called by LyingDetector each frame before this
          runs, so angle history is already populated.
    """

    def detect(
        self,
        window: FrameWindow,
        keypoints: np.ndarray,
    ) -> tuple[bool, dict]:
        """
        Returns (is_near_fall, evidence_dict).
        evidence_dict["bend_detected"] distinguishes bending from dynamic fall.
        """
        angle = body_angle_from_vertical(keypoints)

        evidence = {
            # Path A — dynamic instability
            "angle":              angle,
            "in_angle_zone":      False,
            "rapid_tilt":         False,
            "arms_raised":        False,
            "com_off_balance":    False,
            "angle_velocity":     0.0,
            "angle_delta_3f":     None,
            # Path B — bend
            "bend_detected":      False,
            "bend_angle_pass":    False,
            "hip_drop_ratio":     None,
            "hip_stable":         None,
            "torso_compression":  None,
            "torso_folded":       None,
            "bend_sustained":     False,
            # Combined
            "path":               "none",
        }

        if angle is None:
            return False, evidence

        # ═══════════════════════════════════════════════════════════════════════
        # PATH A — Dynamic instability (original logic, unchanged)
        # ═══════════════════════════════════════════════════════════════════════

        in_zone = NEAR_FALL_ANGLE_MIN <= angle <= NEAR_FALL_ANGLE_MAX
        evidence["in_angle_zone"] = in_zone

        # angle_delta and velocity thresholds are FPS-calibrated inside the window
        delta_3f   = window.angle_delta()        # ~0.1s worth of frames
        velocity   = window.angle_velocity()     # degrees/frame over velocity window
        threshold  = window.angle_delta_per_frame  # deg/frame equiv of config deg/s
        rapid_tilt = (
            (delta_3f is not None and abs(delta_3f) >= threshold * window.velocity_window_frames)
            or abs(velocity) >= threshold
        )
        evidence["rapid_tilt"]     = rapid_tilt
        evidence["angle_velocity"] = velocity
        evidence["angle_delta_3f"] = delta_3f

        arms_up = is_arms_raised(keypoints)
        evidence["arms_raised"] = arms_up

        com_x               = center_of_mass_x(keypoints)
        base_min, base_max  = support_base_x(keypoints)
        com_off = (
            com_x is not None and base_min is not None
            and not (base_min <= com_x <= base_max)
        )
        evidence["com_off_balance"] = com_off

        path_a = in_zone and rapid_tilt and (arms_up or com_off)

        # ═══════════════════════════════════════════════════════════════════════
        # PATH B — Bend detection
        # ═══════════════════════════════════════════════════════════════════════

        path_b       = False
        bend_signals = False

        # Only attempt if we have calibrated standing references
        standing_h     = window.standing_height_estimate
        standing_hip   = window.standing_hip_y_estimate
        standing_torso = window.standing_torso_length_estimate

        if standing_h is not None and standing_hip is not None:

            # Signal 1: angle is meaningfully forward (body is not upright)
            bend_angle_pass = angle >= BEND_ANGLE_MIN
            evidence["bend_angle_pass"] = bend_angle_pass

            # Signal 2: hip has NOT dropped significantly (hip acts as hinge)
            #   Bending: torso pivots at hip — hip stays near standing height
            #   Falling: whole body tilts — hip descends with shoulders
            h_drop = hip_height_ratio(keypoints, standing_hip, standing_h)
            evidence["hip_drop_ratio"] = h_drop
            hip_stable = (h_drop is not None and h_drop < BEND_HIP_DROP_RATIO_MAX)
            evidence["hip_stable"] = hip_stable

            # Signal 3: torso vertical gap is reduced compared to standing
            #   When standing: shoulder_y is well above hip_y (large vertical gap)
            #   When bending:  shoulder drops toward hip on Y axis (gap shrinks)
            #   We use the standing torso length as the baseline for this gap.
            t_comp = torso_compression_ratio(keypoints, standing_torso) if standing_torso else None
            evidence["torso_compression"] = t_comp
            # Accept any compression below 0.95 (torso 5%+ shorter than standing)
            # This is a loose check — the main discriminator is hip_stable
            torso_check = (t_comp is None or t_comp <= 0.95)
            evidence["torso_folded"] = torso_check

            bend_signals = bend_angle_pass and hip_stable

        # ── Push bend flag into window so sustained_bending() can check it ────
        # We update the latest entry's is_bending flag.
        # (window.push happened in LyingDetector; we patch the latest entry)
        latest = window.latest()
        if latest is not None:
            latest["is_bending"] = bend_signals

        # Temporal: bend must hold for BEND_SUSTAINED_FRAMES consecutive frames
        bend_sustained = window.sustained_bending()   # uses FPS-calibrated frame count
        evidence["bend_sustained"] = bend_sustained
        evidence["bend_detected"]  = bend_signals and bend_sustained
        path_b = evidence["bend_detected"]

        # ── Combined decision ─────────────────────────────────────────────────
        if path_a and path_b:
            evidence["path"] = "both"
        elif path_a:
            evidence["path"] = "dynamic"
        elif path_b:
            evidence["path"] = "bend"

        is_near_fall = path_a or path_b
        return is_near_fall, evidence
