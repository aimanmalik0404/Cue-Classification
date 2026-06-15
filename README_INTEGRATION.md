# Nex Behavioral Cues Pipeline — v0.1
## Integration Package for Kenneth / MS-11

---

## What's in this package

```
nex_behavioral_cues_v0.1/
├── README_INTEGRATION.md          ← this file
├── cue_manifest_v1.json           ← machine-readable cue code → source mapping
├── CONFIDENCE_NOTES.md            ← per-cue thresholds + signal descriptions
├── heuristic_pipeline/            ← full runnable heuristic pipeline
│   ├── main.py                    ← entry point (video file or frame-by-frame)
│   ├── configs/config.yaml        ← all tunables including cue_publisher settings
│   ├── utils/cue_publisher.py     ← behavior_cues formatter + Redis publisher
│   ├── utils/alert_manager.py     ← internal alert bus (calls cue_publisher)
│   ├── detectors/                 ← one sub-package per behavioral cue
│   │   ├── fast_approach/
│   │   ├── crowd_convergence/
│   │   ├── head_drop/
│   │   ├── fall/                  ← near_fall + lying_or_collapsed
│   │   ├── slumped_posture/
│   │   ├── leaning/
│   │   └── scene_quality/         ← fires context_hints, not cue codes
│   ├── tracking/tracker.py        ← YOLOv11-pose wrapper + ByteTrack
│   └── visualization/             ← annotated video output (optional)
└── sample_outputs/
    ├── aggressive_interaction_samples.jsonl
    └── physical_instability_samples.jsonl
```

---

## Quickstart — standalone video file

```bash
# 1. Create virtualenv + install deps
python -m venv .venv && source .venv/bin/activate
pip install -r heuristic_pipeline/requirements.txt

# 2. Edit config — set your camera/org IDs
#    heuristic_pipeline/configs/config.yaml  →  cue_publisher section

# 3. Run on a video file
cd heuristic_pipeline
python main.py --video /path/to/clip.mp4 --config configs/config.yaml

# Output:
#   outputs/behavior_cues/<clip>_behavior_cues.jsonl   ← Kenneth's schema, one msg/line
#   outputs/alerts/<clip>_alerts_<ts>.jsonl            ← internal debug log
#   outputs/videos/<clip>_annotated.mp4                ← visualized output
```

---

## Redis live publishing

Set `redis_url` in `configs/config.yaml`:

```yaml
cue_publisher:
  camera_id: 4              # ← your actual camera ID
  organization_id: 12       # ← your actual org ID
  redis_url: "redis://localhost:6379"   # ← point at your Redis instance
```

The pipeline will then publish to:
```
org:{organization_id}:behavior_cues:{family}:cam:{camera_id}
```

**No other code changes needed.** Once Kenneth ships `shared/cue_publisher.py`, swap out `utils/cue_publisher.py` with his helper — the call signature in `alert_manager.py` is already compatible.

---

## Integrating with your existing pose stream (MS-1 keypoints)

If you already have YOLOv11-pose keypoints flowing and don't want to re-run inference:

```python
from heuristic_pipeline.pipeline import HeuristicPipeline
import yaml

# 1. Load config
with open("heuristic_pipeline/configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Initialize and calibrate pipeline with your stream's FPS + resolution
pipeline = HeuristicPipeline(cfg, video_stem="cam4_live")
pipeline.calibrate(fps=25.0, width=1920, height=1080)

# 3. Per-frame: pass your Track objects directly into the pipeline
#    Track objects need: .track_id, .box, .keypoints, .centroid, .velocity, .speed
#    See tracking/tracker.py → TrackState for the expected shape
for frame_idx, (frame, tracks) in enumerate(your_stream):
    timestamp = frame_idx / 25.0
    
    # Process heuristics, fire alerts, and publish to Redis automatically
    pipeline.process_tracks(frame, tracks, frame_idx, timestamp)
```

---

## Output format — one message per alert

Each line in the `.jsonl` file is one `behavior_cues` message:

```json
{
  "type": "behavior_cues",
  "family": "physical_instability",
  "camera_id": 4,
  "organization_id": 12,
  "track_id": 3,
  "timestamp": 12.967,
  "pts_timestamp": 12.967,
  "frame_id": "cam4_f324",
  "cues": [
    { "code": "near_fall", "confidence": 0.864 }
  ],
  "context_hints": [],
  "metadata": {
    "module": "heuristic_pipeline",
    "module_version": "0.1.0",
    "window_seconds": 4.0,
    "keypoint_model": "yolo11n-pose",
    "detector_signals": {
      "fall_detection": { "type": "near_fall", "path": "dynamic" }
    }
  }
}
```

`pts_timestamp` is set equal to `timestamp` until MS-1 PTS is wired through.

---

## Cue codes covered — this package

### `physical_instability`
| Code | Source | Status |
|---|---|---|
| `near_fall` | Heuristic | ✅ |
| `lying_or_collapsed` | Heuristic | ✅ |
| `head_drop` | Heuristic | ✅ |
| `slumped_posture` | Heuristic | ✅ |
| `leaning_on_surface_or_person` | Heuristic | ✅ |
| `stumble_recover` | PoseC3D *(separate package)* | ✅ |
| `not_visible_enough` | → `context_hint: lighting_poor` | ✅ |
| `stagger_walk` | — | 🔶 R&D |
| `swaying` | — | 🔶 R&D |

### `aggressive_interaction`
| Code | Source | Status |
|---|---|---|
| `fast_approach` | Heuristic | ✅ |
| `crowd_convergence` | Heuristic | ✅ |
| `aggressive_posture` | PoseC3D *(separate package)* | ✅ |
| `push_shove` | PoseC3D *(separate package)* | ✅ |
| `swing_attempt` | PoseC3D *(separate package)* | ✅ |
| `grapple_clinch` | PoseC3D *(separate package)* | ✅ |
| `person_falls_after_contact` | PoseC3D *(separate package)* | ✅ |
| `occluded_unclear` | → `context_hint: partial_body_visible` | ✅ |

PoseC3D outputs integrate via the same `AlertManager.fire()` + `CuePublisher.publish()` pathway — just use the Kenneth cue code as the `cue` key when calling `fire()`.

---

## Replacing `cue_publisher.py` with Kenneth's helper

When `shared/cue_publisher.py` is available:

1. Delete `utils/cue_publisher.py`
2. In `utils/alert_manager.py`, replace:
   ```python
   from utils.cue_publisher import CuePublisher
   ```
   with:
   ```python
   from shared.cue_publisher import build_behavior_cues_message, publish_behavior_cues
   ```
3. In `AlertManager.fire()`, replace `self.cue_publisher.publish(alert)` with your call to Kenneth's helper — the `alert` dict contains all required fields.

---

## Known limitations (for Ben's demo)

- **Wide / distant shots**: YOLO11n-pose loses keypoint precision beyond ~8 m at 1080p
- **Dark scenes** (< 30 lux): confidence drops, `lighting_poor` context_hint fires automatically
- **Heavy occlusion** (>15% IoU overlap between bounding boxes): `partial_body_visible` hint fires
- **CPU speed**: ~6 fps on CPU at 4K; 25+ fps expected on GPU with `device: cuda:0` in config
- `stagger_walk` and `swaying` are not yet implemented (R&D, next sprint)

---

*Nex Team — 15 Jun 2026 — v0.1*
