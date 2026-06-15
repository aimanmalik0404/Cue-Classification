"""utils/cue_publisher.py — Formats and publishes behavior_cues messages.

This module converts our internal alert format → Kenneth's locked v1 schema:

    {
        "type": "behavior_cues",
        "family": "physical_instability" | "aggressive_interaction",
        "camera_id": int,
        "organization_id": int,
        "track_id": int | null,
        "timestamp": float,
        "pts_timestamp": float,
        "frame_id": str | null,
        "cues": [{"code": str, "confidence": float}],
        "context_hints": [str],
        "metadata": dict
    }

Two transport modes:
    - JSON file  (always active — one .jsonl per run, one message per line)
    - Redis pubsub (optional — enable via config + install redis-py)

Kenneth will later replace this with build_behavior_cues_message() /
publish_behavior_cues() from shared/cue_publisher.py.  Until then this
module is the drop-in stand-in — the output format is identical.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("CuePublisher")

# ---------------------------------------------------------------------------
# Cue code mapping
# Our internal cue name  →  (kenneth_code, family)
# ---------------------------------------------------------------------------
_CUE_MAP: dict[str, tuple[str, str]] = {
    # ── physical_instability ──────────────────────────────────────────────
    "head_drop":        ("head_drop",                    "physical_instability"),
    "slumped_posture":  ("slumped_posture",               "physical_instability"),
    "leaning":          ("leaning_on_surface_or_person",  "physical_instability"),
    # fall_detection has sub-types resolved at fire-time via metadata["type"]
    "fall_detection":   (None,                            "physical_instability"),

    # ── aggressive_interaction ────────────────────────────────────────────
    "fast_approach":    ("fast_approach",                 "aggressive_interaction"),
    "crowd_convergence":("crowd_convergence",             "aggressive_interaction"),

    # ── PoseC3D cues (fired externally with the code already set) ─────────
    "push_shove":               ("push_shove",               "aggressive_interaction"),
    "swing_attempt":            ("swing_attempt",             "aggressive_interaction"),
    "grapple_clinch":           ("grapple_clinch",            "aggressive_interaction"),
    "person_falls_after_contact":("person_falls_after_contact","aggressive_interaction"),
    "stumble_recover":          ("stumble_recover",           "physical_instability"),
    "aggressive_posture":       ("aggressive_posture",        "aggressive_interaction"),
}

# fall_detection sub-type → kenneth code + family
_FALL_SUBTYPE_MAP: dict[str, tuple[str, str]] = {
    "near_fall":          ("near_fall",        "physical_instability"),
    "lying_or_collapsed": ("lying_or_collapsed","physical_instability"),
    "bending":            ("near_fall",         "physical_instability"),  # bending → near_fall cue
}

# scene_quality signals → context_hint codes
_QUALITY_HINTS: dict[str, str] = {
    "POOR": "lighting_poor",
}

# Minimum confidence to publish (drops at source per Kenneth's request)
# Can be overridden per cue in config.
_DEFAULT_EMIT_THRESHOLD = 0.30


def _resolve_cue_code(internal_cue: str, metadata: dict) -> Optional[tuple[str, str]]:
    """Return (kenneth_code, family) or None if unknown / suppressed."""
    if internal_cue == "fall_detection":
        sub = metadata.get("type", "")
        return _FALL_SUBTYPE_MAP.get(sub)

    if internal_cue == "scene_quality":
        # Scene quality fires as context_hints, not cue codes.
        return None

    return _CUE_MAP.get(internal_cue)


def _context_hints_from_quality(quality_metadata: Optional[dict]) -> list[str]:
    """Convert SceneQualityDetector metadata → context_hint strings."""
    if not quality_metadata:
        return []
    hints = []
    cat = quality_metadata.get("category", "")
    if cat in _QUALITY_HINTS:
        hints.append(_QUALITY_HINTS[cat])
    occ = quality_metadata.get("occlusion", 0.0)
    if occ > 0.15:
        hints.append("partial_body_visible")
    return hints


# ---------------------------------------------------------------------------

class CuePublisher:
    """
    Formats internal alerts into behavior_cues messages and writes them.

    Usage (inside AlertManager.fire):
        publisher.publish(alert, quality_metadata)

    The publisher accumulates all messages in memory (self.messages) and
    also writes each one to a .jsonl file immediately on fire.

    Redis publishing is available when redis_url is set in config and
    the 'redis' package is installed.
    """

    def __init__(self, cfg: dict, output_dir: str, video_stem: str):
        pub_cfg = cfg.get("cue_publisher", {})

        self.camera_id       = int(pub_cfg.get("camera_id", 0))
        self.organization_id = int(pub_cfg.get("organization_id", 0))
        self.module_version  = pub_cfg.get("module_version", "0.1.0")
        self.emit_threshold  = float(pub_cfg.get("emit_threshold", _DEFAULT_EMIT_THRESHOLD))
        self.per_cue_thresh  = pub_cfg.get("per_cue_thresholds", {})
        self.window_seconds  = cfg.get("pipeline", {}).get("window_seconds", 4.0)

        # JSON file output
        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._out_dir / f"{video_stem}_behavior_cues.jsonl"

        # Redis (optional)
        self._redis = None
        redis_url = pub_cfg.get("redis_url")
        if redis_url:
            try:
                import redis  # type: ignore
                self._redis = redis.from_url(redis_url)
                logger.info(f"CuePublisher: Redis connected → {redis_url}")
            except ImportError:
                logger.warning("CuePublisher: 'redis' package not installed — Redis publish disabled.")
            except Exception as e:
                logger.warning(f"CuePublisher: Redis connection failed ({e}) — file-only mode.")

        self.messages: list[dict] = []
        self._quality_metadata: Optional[dict] = None  # refreshed each frame

    # ------------------------------------------------------------------
    def update_quality(self, quality_metadata: Optional[dict]):
        """Call each frame with the latest SceneQualityDetector metadata."""
        self._quality_metadata = quality_metadata

    # ------------------------------------------------------------------
    def publish(self, alert: dict) -> Optional[dict]:
        """
        Convert one internal alert dict → behavior_cues message and emit.

        Returns the published message dict, or None if filtered (confidence
        below threshold or unknown cue code).
        """
        internal_cue = alert.get("cue", "")
        confidence   = float(alert.get("confidence", 0.0))
        metadata     = alert.get("metadata", {})

        # Resolve Kenneth code + family
        resolved = _resolve_cue_code(internal_cue, metadata)
        if resolved is None:
            return None  # scene_quality or unknown — not published as a cue

        kenneth_code, family = resolved

        # Per-cue emit threshold
        threshold = self.per_cue_thresh.get(kenneth_code, self.emit_threshold)
        if confidence < threshold:
            logger.debug(f"CuePublisher: dropped {kenneth_code} conf={confidence:.3f} < {threshold}")
            return None

        # Context hints from latest scene quality reading
        hints = _context_hints_from_quality(self._quality_metadata)

        # Build cue entry
        cue_entry = {"code": kenneth_code, "confidence": round(confidence, 3)}

        # track_id — null for crowd-level cues, person-level for others
        if internal_cue == "crowd_convergence":
            track_id = None
        elif internal_cue == "fast_approach":
            # approacher is the subject of the cue
            track_id = metadata.get("approacher_id", metadata.get("track_id"))
        else:
            track_id = metadata.get("track_id")

        # Detector signal detail for metadata
        signal_detail = {k: v for k, v in metadata.items()
                         if k not in ("track_id", "is_window_prediction", "best", "events")}

        ts = float(alert.get("timestamp", time.time()))

        msg = {
            "type":            "behavior_cues",
            "family":          family,
            "camera_id":       self.camera_id,
            "organization_id": self.organization_id,
            "track_id":        track_id,
            "timestamp":       round(ts, 3),
            "pts_timestamp":   round(ts, 3),   # same as timestamp until MS-1 PTS is wired
            "frame_id":        f"cam{self.camera_id}_f{alert.get('frame_idx', 0)}",
            "cues":            [cue_entry],
            "context_hints":   hints,
            "metadata": {
                "module":          "heuristic_pipeline",
                "module_version":  self.module_version,
                "window_seconds":  self.window_seconds,
                "keypoint_model":  "yolo11n-pose",
                "detector_signals": {internal_cue: signal_detail},
            },
        }

        self._write(msg)
        self.messages.append(msg)
        return msg

    # ------------------------------------------------------------------
    def _write(self, msg: dict):
        """Write message to .jsonl file and optionally to Redis."""
        line = json.dumps(msg)

        # File
        with open(self._jsonl_path, "a") as f:
            f.write(line + "\n")

        # Redis
        if self._redis is not None:
            channel = (
                f"org:{self.organization_id}:behavior_cues"
                f":{msg['family']}:cam:{self.camera_id}"
            )
            try:
                self._redis.publish(channel, line)
            except Exception as e:
                logger.warning(f"CuePublisher: Redis publish failed: {e}")

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return count of published messages per kenneth code."""
        from collections import Counter
        counts = Counter(
            cue["code"]
            for msg in self.messages
            for cue in msg.get("cues", [])
        )
        return {"total_published": len(self.messages), "by_cue": dict(counts)}
