"""heuristic_pipeline/service.py — Live Redis Cue Service

Subscribes to MS-1 ai_detection messages on:
    org:{organization_id}:base_detection:{camera_id}
  (Edge variant with device_id prefix also supported via SUBSCRIBE_CHANNEL env var)

For each frame message, parses persons[] into PersonTrack objects and runs the
HeuristicPipeline, which publishes behavior_cues back to Redis on:
    org:{organization_id}:behavior_cues:{family}:{camera_id}

Environment variables (required):
    REDIS_URL          e.g. redis://redis:6379/0
    ORGANIZATION_ID    e.g. 1
    CAMERA_ID          e.g. 4
    SUBSCRIBE_CHANNEL  e.g. org:1:base_detection:4

Environment variables (optional):
    DEVICE_ID          edge device id (informational only — used in subscribe channel if provided)
    EMIT_THRESHOLD     global min confidence to publish (default 0.30)
    CONFIG_PATH        path to config.yaml (default: configs/config.yaml)
    LOG_LEVEL          DEBUG | INFO | WARNING (default: INFO)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import redis

# ── Path bootstrap so imports work from the project root ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from pipeline import HeuristicPipeline
from tracking.tracker import PersonTrack

# ─────────────────────────────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("CueService")


def _build_config(cfg_path: str) -> dict:
    """Load config.yaml and overlay environment variable overrides."""
    cfg = load_config(cfg_path)

    # Inject publisher config from environment
    pub = cfg.setdefault("cue_publisher", {})
    pub["organization_id"] = int(os.environ.get("ORGANIZATION_ID", pub.get("organization_id", 1)))
    pub["camera_id"]       = int(os.environ.get("CAMERA_ID",       pub.get("camera_id", 0)))
    pub["redis_url"]       = os.environ.get("REDIS_URL",           pub.get("redis_url", "redis://localhost:6379/0"))
    pub["emit_threshold"]  = float(os.environ.get("EMIT_THRESHOLD", pub.get("emit_threshold", 0.30)))

    return cfg


def _handle_frame(
    msg_data: dict,
    pipeline: HeuristicPipeline,
    history_store: dict,
    history_maxlen: int,
    frame_number: int,
) -> None:
    """
    Parse one ai_detection message and push through the HeuristicPipeline.

    Args:
        msg_data:       Parsed JSON dict from the Redis message payload.
        pipeline:       Initialized HeuristicPipeline instance.
        history_store:  Persistent track history dict (mutated in-place).
        history_maxlen: Centroid/keypoints history buffer length.
        frame_number:   Running frame counter used for windowed detectors.
    """
    persons = msg_data.get("persons", [])
    if not persons:
        return

    timestamp  = float(msg_data.get("pts_timestamp") or msg_data.get("timestamp") or time.time())
    frame_shape = msg_data.get("frame_shape", [1080, 1920, 3])  # [H, W, C]
    height, width = int(frame_shape[0]), int(frame_shape[1])

    # Build tracks — no YOLO, no image, straight from MS-1 JSON
    tracks: list[PersonTrack] = [
        PersonTrack.from_ms1_json(p, history_store, history_maxlen)
        for p in persons
    ]

    # Pass a None frame — detectors that need pixel data (scene_quality) will
    # skip gracefully; all keypoint-based detectors work purely on tracks.
    pipeline.process_tracks(
        frame=None,
        tracks=tracks,
        frame_idx=frame_number,
        timestamp=timestamp,
    )


def main() -> None:
    # ── Configuration ─────────────────────────────────────────────────────────
    cfg_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    cfg = _build_config(cfg_path)

    organization_id = cfg["cue_publisher"]["organization_id"]
    camera_id       = cfg["cue_publisher"]["camera_id"]
    redis_url       = cfg["cue_publisher"]["redis_url"]
    subscribe_ch    = os.environ.get(
        "SUBSCRIBE_CHANNEL",
        f"org:{organization_id}:base_detection:{camera_id}",
    )

    fps_hint = float(os.environ.get("FPS_HINT", "25.0"))
    width_hint  = int(os.environ.get("WIDTH_HINT",  "1920"))
    height_hint = int(os.environ.get("HEIGHT_HINT", "1080"))

    # ── Pipeline init ─────────────────────────────────────────────────────────
    # Use a fixed stem for the JSONL audit log in live mode
    pipeline = HeuristicPipeline(cfg, video_stem=f"live_cam{camera_id}")
    pipeline.calibrate(fps=fps_hint, width=width_hint, height=height_hint)

    history_store: dict = {}
    history_maxlen = max(60, round(
        float(cfg.get("pipeline", {}).get("history_buffer_seconds", 5.0)) * fps_hint
    ))

    # ── Redis connection ──────────────────────────────────────────────────────
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
    logger.info(f"Connected to Redis at {redis_url}")
    logger.info(f"Subscribing to channel: {subscribe_ch}")

    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(subscribe_ch)

    frame_number = 0
    logger.info("Service ready — waiting for MS-1 frames...")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            msg_data = json.loads(message["data"])
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON payload: {e}")
            continue

        if msg_data.get("type") != "ai_detection":
            continue

        t0 = time.perf_counter()
        _handle_frame(msg_data, pipeline, history_store, history_maxlen, frame_number)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if frame_number % 100 == 0:
            alert_summary, pub_summary = pipeline.get_summary()
            logger.info(
                f"frame={frame_number}  persons={len(msg_data.get('persons', []))}  "
                f"latency={elapsed_ms:.1f}ms  "
                f"alerts={alert_summary['total']}  published={pub_summary['total_published']}"
            )

        frame_number += 1


if __name__ == "__main__":
    main()
