# Nex Heuristic Cue Service — Deployment Guide

**Version:** 0.1.0  
**Last updated:** June 2026  
**Handoff recipient:** Kenneth / OffTapWatch Engineering

---

## Overview

The **Nex Heuristic Cue Service** is a persistent Python daemon that subscribes to MS-1 `ai_detection` frames from Redis, runs all heuristic cue detectors on the keypoint stream, and publishes `behavior_cues` messages back to Redis — using the same schema as the JSONL replay path already validated in staging.

This service replaces the need to run YOLO on your edge. All pose tracking is done by your MS-1; the Nex service is a pure logic/inference step.

---

## Architecture

```
MS-1 (Camera + Pose Tracking)
        |
        | Redis pubsub
        v
org:{org_id}:base_detection:{cam_id}
        |
        v
nex-heuristic-cues (this service)
        |  Runs: HeuristicPipeline (windowing, detection, scoring)
        |
        v
org:{org_id}:behavior_cues:{family}:{cam_id}
        |
        v
MS-11 Alert Aggregator → OffTapWatch
```

---

## Quick Start

### 1. Build the image

```bash
docker build -f Dockerfile.heuristic -t nex/heuristic-cue-service:latest .
```

### 2. Run with Docker Compose (staging)

```bash
docker-compose -f docker-compose.heuristic.yml up
```

Adjust the environment variables in `docker-compose.heuristic.yml` to match your `ORGANIZATION_ID`, `CAMERA_ID`, and `REDIS_URL` before deploying.

### 3. Run standalone (no Compose)

```bash
docker run -d \
  -e REDIS_URL=redis://your-redis-host:6379/0 \
  -e ORGANIZATION_ID=1 \
  -e CAMERA_ID=4 \
  -e SUBSCRIBE_CHANNEL=org:1:base_detection:4 \
  nex/heuristic-cue-service:latest
```

---

## Environment Variables

| Variable             | Required | Default                         | Description |
|----------------------|----------|---------------------------------|-------------|
| `REDIS_URL`          | ✅ Yes   | —                               | Redis connection URL |
| `ORGANIZATION_ID`    | ✅ Yes   | —                               | Your org ID (must match MS-1 channel) |
| `CAMERA_ID`          | ✅ Yes   | —                               | Camera ID (must match MS-1 channel) |
| `SUBSCRIBE_CHANNEL`  | ✅ Yes   | `org:{org}:base_detection:{cam}`| Full Redis subscribe channel. For edge: `org:1:device:{device_id}:base_detection:4` |
| `DEVICE_ID`          | No       | `0`                             | Edge device ID (informational) |
| `EMIT_THRESHOLD`     | No       | `0.30`                          | Global minimum confidence to publish |
| `FPS_HINT`           | No       | `25.0`                          | Camera FPS for window calibration |
| `WIDTH_HINT`         | No       | `1920`                          | Frame width in pixels |
| `HEIGHT_HINT`        | No       | `1080`                          | Frame height in pixels |
| `LOG_LEVEL`          | No       | `INFO`                          | `DEBUG`, `INFO`, or `WARNING` |
| `CONFIG_PATH`        | No       | `configs/config.yaml`           | Path to the detector config file |

---

## Redis Channel Format

### Subscribe (Input from MS-1)
| Deployment  | Channel |
|-------------|---------|
| Cloud/Staging | `org:{organization_id}:base_detection:{camera_id}` |
| Edge (with device) | `org:{organization_id}:device:{device_id}:base_detection:{camera_id}` |

### Publish (Output to MS-11)
```
org:{organization_id}:behavior_cues:{family}:{camera_id}
```
Where `family` is either `physical_instability` or `aggressive_interaction`.

---

## Expected Latency

| Scenario | Per-frame Processing Time |
|----------|--------------------------|
| CPU (2 people in frame) | ~3–8 ms |
| CPU (5 people in frame) | ~8–18 ms |
| GPU (any) | ~1–3 ms |

> Windowed detectors (`near_fall`, `slumped_posture`, `leaning`, etc.) only emit a cue at the **end of each 4-second window**, not every frame. Per-frame latency is for track state updates.

---

## Multi-person & Pair Cue Handling

### `fast_approach`
Evaluated as a **pair cue**. For every combination of (approacher, target) in the current window, the service checks:
- Speed of the approacher (from `velocity` in the MS-1 payload, or derived from centroid history)
- Decrease in inter-person distance over the window
- Alignment of velocity vector toward the target

Published with `track_id = approacher_id`. The `metadata` field includes both `approacher_id` and `target_id`.

### `crowd_convergence`
Evaluated across **all persons** in the window using cluster-based convergence scoring.
Published with `track_id = null` (crowd-level, no single subject).

### PoseC3D pair cues (push_shove, grapple_clinch, swing_attempt)
These are handled by the **separate PoseC3D cue service** (coming separately from your teammate). Both services subscribe to the same `base_detection` channel independently.

---

## Per-cue Confidence Thresholds

Apply per-cue emit thresholds from the `CONFIDENCE_NOTES.md` document. These can also be overridden in `configs/config.yaml` under `cue_publisher.per_cue_thresholds`:

```yaml
cue_publisher:
  emit_threshold: 0.30     # global floor
  per_cue_thresholds:
    near_fall:          0.50
    lying_or_collapsed: 0.50
    head_drop:          0.45
    slumped_posture:    0.45
    leaning_on_surface_or_person: 0.45
    fast_approach:      0.40
    crowd_convergence:  0.40
```

---

## Audit Log

Every published `behavior_cue` message is also written to a local `.jsonl` file at:
```
outputs/behavior_cues/live_cam{camera_id}_behavior_cues.jsonl
```
This mirrors the offline JSONL path already validated in staging.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No cues published | `EMIT_THRESHOLD` too high | Lower it to `0.30` |
| `redis.exceptions.ConnectionError` | Wrong `REDIS_URL` | Check URL and network access |
| No cues for `fast_approach` | `FPS_HINT` mismatch | Set `FPS_HINT` to match your actual camera FPS |
| Short clips show no cues | Window not complete | Detectors need ~4s of data. Use longer clips for testing. |

---

*Nex Team — June 2026*
