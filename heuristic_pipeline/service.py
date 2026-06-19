"""heuristic_pipeline/service.py — Global Multi-Camera Orchestrator

One container handles ALL cameras assigned to an edge device.

Bootstrap (production):
    1. Read ORGANIZATION_ID, DEVICE_ID, REDIS_HOST/PORT, MS3_URL from env.
    2. Call MS-3: GET {MS3_URL}/api/edge/devices/{device_id}/cameras
    3. Initialize one HeuristicPipeline per camera.
    4. psubscribe to org:{org}:device:{device_id}:base_detection:*
    5. Subscribe to device:{device_id}:commands for live add_camera/remove_camera.

Bootstrap (staging / local dev, no MS-3):
    Set STAGING_CAMERA_IDS=4,5  →  skips MS-3 lookup, uses staging channels.

Environment variables:
    REDIS_HOST          Redis hostname (default: localhost)
    REDIS_PORT          Redis port (default: 6379)
    REDIS_URL           Full URL, overrides REDIS_HOST/PORT if set
    ORGANIZATION_ID     Required
    DEVICE_ID           Required for production edge (optional for staging)
    MS3_URL             MS-3 base URL e.g. http://ms3:8000
    STAGING_CAMERA_IDS  Comma-separated IDs, bypasses MS-3 e.g. "4,5"
    EMIT_THRESHOLD      Global min confidence (default 0.30)
    FPS_HINT            Camera FPS for window calibration (default 25.0)
    WIDTH_HINT          Frame width hint (default 1920)
    HEIGHT_HINT         Frame height hint (default 1080)
    CONFIG_PATH         Path to config.yaml (default: configs/config.yaml)
    LOG_LEVEL           DEBUG | INFO | WARNING (default: INFO)
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from typing import Optional

import redis

# ── Path bootstrap ────────────────────────────────────────────────────────────
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


# ── Per-camera state container ────────────────────────────────────────────────

@dataclass
class CameraState:
    camera_id: int
    pipeline: HeuristicPipeline
    history_store: dict = field(default_factory=dict)
    frame_number: int = 0
    history_maxlen: int = 125


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_base_config(cfg_path: str) -> dict:
    """Load config.yaml and apply global env var overrides (no camera_id yet)."""
    cfg = load_config(cfg_path)
    pub = cfg.setdefault("cue_publisher", {})
    pub["organization_id"] = int(os.environ.get("ORGANIZATION_ID", pub.get("organization_id", 1)))
    pub["device_id"]       = os.environ.get("DEVICE_ID", pub.get("device_id", ""))
    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        host = os.environ.get("REDIS_HOST", "localhost")
        port = os.environ.get("REDIS_PORT", "6379")
        redis_url = f"redis://{host}:{port}/0"
    pub["redis_url"]      = redis_url
    pub["emit_threshold"] = float(os.environ.get("EMIT_THRESHOLD", pub.get("emit_threshold", 0.30)))
    return cfg


def _build_camera_config(base_cfg: dict, camera_id: int) -> dict:
    """Deep-copy base config and inject the camera-specific camera_id."""
    cfg = copy.deepcopy(base_cfg)
    cfg["cue_publisher"]["camera_id"] = camera_id
    return cfg


def _init_camera(camera_id: int, base_cfg: dict, fps: float, width: int, height: int) -> CameraState:
    """Instantiate a fresh pipeline + state for a single camera."""
    cfg = _build_camera_config(base_cfg, camera_id)
    pipeline = HeuristicPipeline(cfg, video_stem=f"live_cam{camera_id}")
    pipeline.calibrate(fps=fps, width=width, height=height)
    history_maxlen = max(60, round(
        float(cfg.get("pipeline", {}).get("history_buffer_seconds", 5.0)) * fps
    ))
    logger.info(f"[camera={camera_id}] Pipeline initialized (history={history_maxlen}f)")
    return CameraState(
        camera_id=camera_id,
        pipeline=pipeline,
        history_maxlen=history_maxlen,
    )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _service_jwt(secret: str, organization_id: Optional[int]) -> str:
    """Mint an HS256 service JWT matching shared.jwt_utils.create_service_token.

    Stdlib-only; MS-3 accepts a service token signed with JWT_SECRET_KEY for
    inter-service calls such as device camera discovery.
    """
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": "heuristic-cue-service",
        "iss": "microservice-auth",
        "type": "service",
        "iat": now,
        "exp": now + 3600,
    }
    if organization_id:
        payload["organization_id"] = int(organization_id)
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + _b64url(sig)


def _ms3_headers(organization_id: Optional[int]) -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    secret = os.environ.get("SECRET_KEY", "").strip()
    if secret:
        headers["Authorization"] = f"Bearer {_service_jwt(secret, organization_id)}"
    else:
        logger.warning("SECRET_KEY not set — calling MS-3 without auth (will 401 on staging)")
    if organization_id:
        headers["X-Organization-ID"] = str(organization_id)
    return headers


def _fetch_cameras_from_ms3(
    ms3_url: str, device_id: str, organization_id: Optional[int] = None
) -> list[int]:
    """Call MS-3 to get the list of camera IDs assigned to this device.

    The endpoint returns camera objects with an ``id`` field (auth required).
    """
    url = f"{ms3_url.rstrip('/')}/api/edge/devices/{device_id}/cameras"
    logger.info(f"Fetching camera list from MS-3: {url}")
    try:
        req = urllib.request.Request(url, headers=_ms3_headers(organization_id))
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if isinstance(data, list):
            ids: list[int] = []
            for c in data:
                cam = c.get("id", c.get("camera_id")) if isinstance(c, dict) else c
                if cam is not None:
                    ids.append(int(cam))
            return ids
        logger.warning(f"Unexpected MS-3 response format: {data}")
        return []
    except urllib.error.URLError as e:
        logger.error(f"MS-3 unreachable ({e}). Cannot bootstrap cameras.")
        raise


def _handle_frame(msg_data: dict, state: CameraState) -> None:
    """Route one ai_detection frame through the correct camera pipeline."""
    persons = msg_data.get("persons", [])
    if not persons:
        return

    timestamp = float(msg_data.get("pts_timestamp") or msg_data.get("timestamp") or time.time())

    tracks = [
        PersonTrack.from_ms1_json(p, state.history_store, state.history_maxlen)
        for p in persons
    ]

    state.pipeline.process_tracks(
        frame=None,
        tracks=tracks,
        frame_idx=state.frame_number,
        timestamp=timestamp,
    )
    state.frame_number += 1


def _handle_command(cmd_data: dict, cameras: dict[int, CameraState],
                    base_cfg: dict, fps: float, width: int, height: int) -> None:
    """Process add_camera / remove_camera commands from Kenneth's command channel."""
    action    = cmd_data.get("action", "")
    camera_id = cmd_data.get("camera_id")

    if camera_id is None:
        logger.warning(f"Command missing camera_id: {cmd_data}")
        return

    camera_id = int(camera_id)

    if action == "add_camera":
        if camera_id in cameras:
            logger.info(f"[camera={camera_id}] Already tracking, ignoring add_camera.")
            return
        cameras[camera_id] = _init_camera(camera_id, base_cfg, fps, width, height)
        logger.info(f"[camera={camera_id}] Added dynamically via command.")

    elif action == "remove_camera":
        if camera_id not in cameras:
            logger.warning(f"[camera={camera_id}] remove_camera but not tracked.")
            return
        del cameras[camera_id]
        logger.info(f"[camera={camera_id}] Removed. State torn down — memory freed.")

    else:
        logger.warning(f"Unknown command action: {action!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    base_cfg = _build_base_config(cfg_path)

    org_id    = base_cfg["cue_publisher"]["organization_id"]
    device_id = base_cfg["cue_publisher"]["device_id"]
    redis_url = base_cfg["cue_publisher"]["redis_url"]

    fps    = float(os.environ.get("FPS_HINT",    "25.0"))
    width  = int(os.environ.get("WIDTH_HINT",  "1920"))
    height = int(os.environ.get("HEIGHT_HINT", "1080"))

    # ── Bootstrap camera list ─────────────────────────────────────────────────
    staging_ids_env = os.environ.get("STAGING_CAMERA_IDS", "")
    if staging_ids_env:
        camera_ids = [int(c.strip()) for c in staging_ids_env.split(",") if c.strip()]
        logger.info(f"Staging mode — using STAGING_CAMERA_IDS: {camera_ids}")
    elif device_id and os.environ.get("MS3_URL"):
        camera_ids = _fetch_cameras_from_ms3(os.environ["MS3_URL"], device_id, org_id)
        logger.info(f"MS-3 assigned cameras: {camera_ids}")
    else:
        logger.warning("No MS3_URL or STAGING_CAMERA_IDS set — starting with no cameras. "
                       "Send add_camera commands to register cameras dynamically.")
        camera_ids = []

    # ── Initialize one pipeline per camera ───────────────────────────────────
    cameras: dict[int, CameraState] = {}
    for cid in camera_ids:
        cameras[cid] = _init_camera(cid, base_cfg, fps, width, height)

    # ── Redis connection ──────────────────────────────────────────────────────
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()
    logger.info(f"Connected to Redis at {redis_url}")

    pubsub = r.pubsub(ignore_subscribe_messages=True)

    # Production edge pattern
    if device_id:
        edge_pattern   = f"org:{org_id}:device:{device_id}:base_detection:*"
        cmd_channel    = f"device:{device_id}:commands"
        pubsub.psubscribe(edge_pattern)
        pubsub.subscribe(cmd_channel)
        logger.info(f"Subscribed (edge): {edge_pattern}  commands: {cmd_channel}")

    # Always also subscribe to staging pattern (works for local dev without device prefix)
    staging_pattern = f"org:{org_id}:base_detection:*"
    pubsub.psubscribe(staging_pattern)
    logger.info(f"Subscribed (staging): {staging_pattern}")

    logger.info(f"Orchestrator ready — managing {len(cameras)} camera(s): {list(cameras.keys())}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    log_ticks: dict[int, int] = {}  # camera_id -> frames since last log

    while True:
        try:
            message = pubsub.get_message(timeout=1.0)
            if message is None:
                continue

            msg_type = message.get("type", "")

            # ── Camera data frames (psubscribe → pmessage) ────────────────────
            if msg_type == "pmessage":
                try:
                    msg_data = json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    continue

                if msg_data.get("type") != "ai_detection":
                    continue

                camera_id = int(msg_data.get("camera_id", -1))
                if camera_id < 0:
                    continue

                # Auto-create pipeline for previously unseen camera (graceful fallback)
                if camera_id not in cameras:
                    logger.info(f"[camera={camera_id}] First seen — auto-initializing pipeline.")
                    cameras[camera_id] = _init_camera(camera_id, base_cfg, fps, width, height)

                t0 = time.perf_counter()
                _handle_frame(msg_data, cameras[camera_id])
                elapsed_ms = (time.perf_counter() - t0) * 1000

                log_ticks[camera_id] = log_ticks.get(camera_id, 0) + 1
                if log_ticks[camera_id] % 100 == 0:
                    state = cameras[camera_id]
                    a_sum, p_sum = state.pipeline.get_summary()
                    logger.info(
                        f"[camera={camera_id}] frame={state.frame_number}  "
                        f"persons={len(msg_data.get('persons', []))}  "
                        f"latency={elapsed_ms:.1f}ms  "
                        f"alerts={a_sum['total']}  published={p_sum['total_published']}"
                    )

            # ── Command channel (subscribe → message) ─────────────────────────
            elif msg_type == "message":
                try:
                    cmd_data = json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    continue
                _handle_command(cmd_data, cameras, base_cfg, fps, width, height)

        except redis.exceptions.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
