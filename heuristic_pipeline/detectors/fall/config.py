"""
config.py — Central configuration for the fall detection pipeline.
All thresholds, paths, and model settings live here.
"""

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = "yolo11n-pose.pt"          # YOLOv11 nano-pose (auto-downloaded)
DEVICE     = "cpu"                      # "cpu" | "cuda" | "mps"
CONF_THRESHOLD = 0.5                    # YOLO detection confidence

# ─── Keypoint confidence ──────────────────────────────────────────────────────
MIN_KP_CONF = 0.3                       # Ignore keypoints below this confidence

# ─── Temporal window ──────────────────────────────────────────────────────────
# All time-based thresholds are defined in SECONDS here.
# FrameWindow converts them to frames at runtime using the actual video FPS.
# This means a 30fps and a 15fps video both react at the same real-world speed.

WINDOW_SECONDS              = 0.5      # Rolling buffer duration (0.5s = 15f@30fps)

# Lying / collapsed
LYING_SUSTAINED_SECONDS     = 0.4      # Must hold lying state this long
HEAD_REST_SECONDS           = 0.4      # Head-on-surface must hold this long

# Near-fall / bend
NEAR_FALL_VELOCITY_SECONDS  = 0.17     # Window for angular velocity averaging
BEND_SUSTAINED_SECONDS      = 0.17     # Bend must hold this long

# Near-fall angle delta: degrees of tilt per second (not per frame)
NEAR_FALL_ANGLE_DELTA_DEG_S = 90       # 90 deg/s rapid tilt (= 3deg/frame @30fps)

# ─── Lying / Collapsed thresholds ────────────────────────────────────────────
LYING_SCORE_THRESHOLD   = 1.5          # shoulder_x_span / (vertical_span + 1)
LYING_VERTICAL_RATIO    = 0.35         # vertical_span < ratio * estimated_standing_height
LYING_JOINT_Y_TOLERANCE = 0.15         # All key joints within 15% of body height on Y
ASPECT_RATIO_LYING      = 1.5          # bbox width/height ratio for lying

# ─── Near-fall thresholds ─────────────────────────────────────────────────────
NEAR_FALL_ANGLE_MIN         = 20       # Degrees — start of at-risk zone
NEAR_FALL_ANGLE_MAX         = 60       # Degrees — before full lying
# NEAR_FALL_ANGLE_DELTA and NEAR_FALL_SCORE_WINDOW are now derived from
# NEAR_FALL_ANGLE_DELTA_DEG_S and NEAR_FALL_VELOCITY_SECONDS at runtime.

# ─── Bending detection thresholds ────────────────────────────────────────────
#
# A bend is detected via the "hip hinge ratio":
#   When bending, torso folds at the hip — the shoulder drops toward the hip
#   but the hip itself stays near its normal (standing) height.
#   When falling, the whole body tilts as a rigid unit — hip also descends.
#
# hip_drop_ratio = (hip_y_now - hip_y_standing) / standing_height
#   Bending:  hip barely moves → ratio ≈ 0.0–0.08  (hip stays high)
#   Falling:  hip descends with body → ratio ≈ 0.15–0.40
#
# torso_compression_ratio = torso_length_now / torso_length_standing
#   Bending:  shoulder folds toward hip → torso appears shorter → ratio < 0.7
#   Falling:  torso stays extended → ratio ≈ 0.9–1.0
#
BEND_HIP_DROP_RATIO_MAX         = 0.10  # Hip has not dropped more than 10% of standing height
BEND_TORSO_COMPRESSION_MIN      = 0.30  # Torso compressed to at least 30% shorter than standing
BEND_ANGLE_MIN                  = 25    # Body angle must be at least 25° (actually bent)
BEND_SUSTAINED_FRAMES           = 5     # Must be sustained this many frames to confirm

# ─── Estimated standing height ───────────────────────────────────────────────
# Used when we don't have prior standing reference per-person.
# 0.7 * frame_height is a safe heuristic default.
DEFAULT_STANDING_HEIGHT_RATIO = 0.7

# ─── Output ───────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
OUTPUT_FPS   = None                    # None = match input FPS
DRAW_SKELETON = True
DRAW_BBOX     = True
DRAW_KEYPOINTS = True

# ─── COCO Keypoint indices ────────────────────────────────────────────────────
KP = {
    "NOSE":          0,
    "LEFT_EYE":      1,
    "RIGHT_EYE":     2,
    "LEFT_EAR":      3,
    "RIGHT_EAR":     4,
    "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER":6,
    "LEFT_ELBOW":    7,
    "RIGHT_ELBOW":   8,
    "LEFT_WRIST":    9,
    "RIGHT_WRIST":   10,
    "LEFT_HIP":      11,
    "RIGHT_HIP":     12,
    "LEFT_KNEE":     13,
    "RIGHT_KNEE":    14,
    "LEFT_ANKLE":    15,
    "RIGHT_ANKLE":   16,
}

# ─── COCO Skeleton connections (for drawing) ──────────────────────────────────
SKELETON = [
    (0,  1),  (0,  2),   # nose → eyes
    (1,  3),  (2,  4),   # eyes → ears
    (5,  6),             # shoulder–shoulder
    (5,  7),  (7,  9),   # left arm
    (6,  8),  (8, 10),   # right arm
    (5, 11),  (6, 12),   # torso sides
    (11,12),             # hip–hip
    (11,13),  (13,15),   # left leg
    (12,14),  (14,16),   # right leg
]

# ─── Label colours (BGR) ─────────────────────────────────────────────────────
LABEL_COLORS = {
    "NORMAL":            (0,   200,  0),
    "BENDING":           (0,   220, 180),   # teal — attention but not alarm
    "NEAR_FALL":         (0,   165, 255),   # orange — alert
    "LYING_OR_COLLAPSED":(0,   0,   255),   # red — emergency
}