# Confidence Notes — Nex Behavioral Cue Pipeline v0.1

Per Kenneth's request: per-cue emit thresholds and signal descriptions
so MS-11 can tune operational alerts.

---

## Global rule

Drop any cue below **0.30** at source (configured in `cue_publisher.emit_threshold`).
Per-cue overrides below take precedence.

---

## Physical Instability Cues

### `near_fall`
- **Emit threshold**: 0.50
- **Range**: 0.45 – 0.95
- **Signal**: Body angle into danger zone (20–60°) + rate of angular change + instability cues (arms raised, CoM outside support base).
- **Score formula**: `0.45 + angle_depth×0.30 + instability_count/3×0.20`
  - `angle_depth = (angle − 20) / 40` clamped [0, 1]
  - `instability_count ∈ {0, 1, 2, 3}` (rapid_tilt, arms_raised, com_off_balance)
- **Bending sub-path** (`path=bend`): maps to `near_fall`; score = `0.45 + (angle − 25)/55×0.45`

### `lying_or_collapsed`
- **Emit threshold**: 0.50
- **Range**: 0.45 – 0.95
- **Geometry path** (`path=geometry`):
  - `0.45 + (signals_passed/4)×0.30 + (lying_score/3)×0.20`
  - signals: score > 1.5, vertical span < 35% of standing height, all joints same Y, aspect ratio > 1.5
- **Surface-rest path** (`path=surface_rest`):
  - `max(0.45, 1.0 − head_raw_var/400)` where `head_raw_var` is nose-Y variance (px²)
  - Lower variance = more confident the head is resting (not actively moving)

### `head_drop`
- **Emit threshold**: 0.45
- **Range**: 0.45 – 1.0
- **Score formula**:
  - `0.45` base: shoulder-width collapses (person faces away / side-on — head hidden)
  - `+0.35`: nose Y drops below shoulder level (head bowed forward)
  - `+0.20`: nose Y drops further over the window (temporal downward change)
- **Max score 1.0** is legitimate (all three signals present simultaneously)

### `slumped_posture`
- **Emit threshold**: 0.45
- **Range**: 0.45 – 0.90
- **Signal**: Fraction of 4-s window frames where shoulder/hip width ratio < 0.75 AND spine deviation > 20°
- **Scaling**: `[min_hit_ratio=0.40 → 1.0]` maps to `[0.45 → 0.90]`
  - Score of 0.90 means slumped every single frame of the window

### `leaning_on_surface_or_person`
- **Emit threshold**: 0.45
- **Range**: 0.45 – 0.90
- **Signal**: Lateral torso offset / torso height > 0.25 AND torso angle > 20°, gated by low movement (< 1.5% diagonal/frame). Same hit-ratio scaling as slumped_posture.
- **Movement gate**: suppresses false positives from walking and dancing

### `stumble_recover` *(PoseC3D)*
- **Emit threshold**: 0.35
- **Range**: 0.0 – 1.0 (sigmoid)
- PoseC3D sigmoid output — raw model probability. MS-11 should apply its own calibration layer.

---

## Aggressive Interaction Cues

### `fast_approach`
- **Emit threshold**: 0.40
- **Range**: 0.0 – 1.0
- **Signal**: Pair velocity (normalized to frame diagonal/s) × direction alignment × proximity score.
- **Velocity threshold**: 10% of frame diagonal per second (auto-scales to any resolution)
- **track_id** = approacher (the person moving toward the target)

### `crowd_convergence`
- **Emit threshold**: 0.40
- **Range**: 0.0 – 1.0
- **Score formula**: `0.45×(1 − dist_ratio) + 0.35×alignment_frac + 0.20×size_score`
  - `dist_ratio`: current avg pairwise distance / baseline avg pairwise distance (lower = more convergence)
  - `alignment_frac`: fraction of velocity vectors pointing toward cluster center
  - `size_score = min(n_persons/6, 1.0)`
- **track_id** = `null` (crowd-level cue)

### PoseC3D cues: `push_shove`, `swing_attempt`, `grapple_clinch`, `aggressive_posture`, `person_falls_after_contact`
- **Emit threshold**: 0.35 (0.40 for `person_falls_after_contact`)
- **Range**: 0.0 – 1.0 (sigmoid)
- Raw sigmoid from PoseC3D multi-label head. MS-11 normalization recommended.

---

## Context Hints (not cue codes)

These appear in `context_hints[]` and should lower MS-11 alert severity, not trigger alerts:

| Hint | Condition |
|---|---|
| `lighting_poor` | Composite visibility score < 0.40 (brightness, contrast, blur all poor) |
| `partial_body_visible` | Bounding-box IoU overlap between tracked persons > 15% |

---

*Nex Team — 15 Jun 2026*
