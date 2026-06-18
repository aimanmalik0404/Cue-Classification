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
MS-1 (Cameras + Pose Tracking)
         |
         | Redis wildcard psubscribe
         v
org:{org_id}:device:{device_id}:base_detection:*
         |
         v
nex-cue-heuristic (Global Orchestrator)
         |  - Fetches camera list from MS-3 API
         |  - Listens for add/remove camera commands
         |  - Routes to internal per-camera HeuristicPipelines
         v
org:{org_id}:device:{device_id}:behavior_cues:{family}:{cam_id}
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

### 2. Run with Docker Compose (Staging / Dev Mode)

For local development without Kenneth's MS-3 API, you can hardcode the cameras you want the orchestrator to track:

```bash
# In docker-compose.heuristic.yml, set:
# STAGING_CAMERA_IDS: "4,5"
docker-compose -f docker-compose.heuristic.yml up
```

### 3. Run Production Edge (with MS-3)

```bash
docker run -d \
  -e REDIS_HOST=localhost \
  -e REDIS_PORT=6379 \
  -e ORGANIZATION_ID=1 \
  -e DEVICE_ID=edge-node-01 \
  -e MS3_URL=http://ms3:8000 \
  nex/heuristic-cue-service:latest
```

---

## Environment Variables

| Variable             | Required | Default                         | Description |
|----------------------|----------|---------------------------------|-------------|
| `REDIS_HOST`         | No       | `localhost`                     | Redis hostname |
| `REDIS_PORT`         | No       | `6379`                          | Redis port |
| `REDIS_URL`          | No       | —                               | Full Redis URL (overrides HOST/PORT if set) |
| `ORGANIZATION_ID`    | ✅ Yes   | —                               | Your org ID |
| `DEVICE_ID`          | Edge     | —                               | Edge device ID (enables MS-3 lookup and device-scoped channels) |
| `MS3_URL`            | Edge     | —                               | Base URL for MS-3 to fetch assigned cameras |
| `STAGING_CAMERA_IDS` | Staging  | —                               | Comma-separated list of cameras (e.g., `4,5`). Bypasses MS-3 lookup. |
| `EMIT_THRESHOLD`     | No       | `0.30`                          | Global minimum confidence to publish |
| `FPS_HINT`           | No       | `25.0`                          | Camera FPS for window calibration |
| `WIDTH_HINT`         | No       | `1920`                          | Frame width in pixels |
| `HEIGHT_HINT`        | No       | `1080`                          | Frame height in pixels |
| `LOG_LEVEL`          | No       | `INFO`                          | `DEBUG`, `INFO`, or `WARNING` |

---

## Redis Channel Format

### Subscribe (Input from MS-1)
The service automatically multiplexes frames from a wildcard channel.

| Deployment Mode | Channel Subscribed |
|-----------------|--------------------|
| Edge (Production)| `org:{organization_id}:device:{device_id}:base_detection:*` |
| Cloud/Staging    | `org:{organization_id}:base_detection:*` |

### Command Channel (Edge only)
The service listens for `add_camera` and `remove_camera` actions on:
```
device:{device_id}:commands
```

### Publish (Output to MS-11)
| Deployment Mode | Channel Published |
|-----------------|-------------------|
| Edge (Production)| `org:{organization_id}:device:{device_id}:behavior_cues:{family}:{camera_id}` |
| Cloud/Staging    | `org:{organization_id}:behavior_cues:{family}:{camera_id}` |

*(Where `family` is either `physical_instability` or `aggressive_interaction`)*

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
