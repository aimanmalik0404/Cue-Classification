"""utils/config_hotreload.py — Candidate config load + health-check (C5).

A reload command points at a candidate config.yaml. Before swapping it in for
a live camera, we:
  1. load_candidate_config()      — load YAML + apply the SAME env overrides
                                     the running service applies at boot.
  2. validate_candidate_config()  — structural + numeric assertions, then a
                                     throwaway HeuristicPipeline instantiation,
                                     calibrate, and one empty process_tracks().

Any failure raises — the caller leaves the active config untouched.
"""
from __future__ import annotations

import logging
from typing import Callable

from utils.config_loader import load_config

logger = logging.getLogger("CueService.config_hotreload")

# Top-level config blocks the running pipeline (pipeline.py) requires, plus
# EVERY detector block it instantiates.
_REQUIRED_TOP_KEYS = ("pipeline", "cue_publisher", "output")
_REQUIRED_DETECTOR_BLOCKS = (
    "fast_approach",
    "crowd_convergence",
    "scene_quality",
    "head_drop",
    "fall_detection",
    "slumped_posture",
    "leaning",
)

# (block, key) pairs that MUST be numeric thresholds when the block is enabled.
_REQUIRED_NUMERIC_THRESHOLDS = {
    "pipeline": ("window_seconds", "predict_interval_seconds", "history_buffer_seconds"),
    "fast_approach": ("velocity_threshold_pct", "distance_decrease_pct"),
    "crowd_convergence": ("distance_decrease_ratio", "min_vector_alignment"),
    "scene_quality": ("blur_threshold",),
    "head_drop": ("score_threshold",),
    "fall_detection": ("lying_score_threshold",),
    "slumped_posture": ("spine_angle_deg", "min_hit_ratio"),
    "leaning": ("lean_ratio_threshold", "min_hit_ratio"),
}


def load_candidate_config(local_yaml_path: str, apply_env_overrides: Callable[[dict], dict]) -> dict:
    """Load a candidate config.yaml and apply the live service's env overrides.

    ``apply_env_overrides`` is service.apply_env_overrides — passed in to avoid
    a circular import and to guarantee the candidate gets the identical
    organization_id / device_id / redis_url / emit_threshold treatment.
    """
    cfg = load_config(local_yaml_path)
    cfg = apply_env_overrides(cfg)
    return cfg


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_candidate_config(cfg: dict) -> bool:
    """HEALTH-CHECK a candidate config. Raises on any failure, else returns True."""
    if not isinstance(cfg, dict):
        raise ValueError("Candidate config is not a mapping")

    # 1. Required top-level blocks present.
    for key in _REQUIRED_TOP_KEYS:
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(f"Candidate config missing required block: {key!r}")

    # 2. Every detector block referenced by pipeline.py must be present.
    for block in _REQUIRED_DETECTOR_BLOCKS:
        if block not in cfg or not isinstance(cfg[block], dict):
            raise ValueError(f"Candidate config missing detector block: {block!r}")

    # 3. Required numeric thresholds present and numeric (only for enabled blocks).
    for block, keys in _REQUIRED_NUMERIC_THRESHOLDS.items():
        section = cfg.get(block, {})
        if block != "pipeline" and not section.get("enabled", True):
            continue
        for k in keys:
            if k not in section:
                raise ValueError(f"Candidate config missing threshold {block}.{k}")
            if not _is_number(section[k]):
                raise ValueError(
                    f"Candidate config threshold {block}.{k} is not numeric: {section[k]!r}"
                )

    # 4. output.output_dir is required by HeuristicPipeline construction.
    if "output_dir" not in cfg["output"]:
        raise ValueError("Candidate config missing output.output_dir")

    # 5. Live smoke test: build a throwaway pipeline, calibrate, run empty frame.
    #    Imported lazily to avoid a circular import (service imports this module).
    from pipeline import HeuristicPipeline
    from service import _build_camera_config

    try:
        probe_cfg = _build_camera_config(cfg, 0)
        probe = HeuristicPipeline(probe_cfg, video_stem="healthcheck")
        probe.calibrate(fps=25.0, width=1920, height=1080)
        probe.process_tracks(frame=None, tracks=[], frame_idx=0, timestamp=0.0)
    except Exception as e:
        raise ValueError(f"Candidate config failed pipeline health-check: {e}") from e

    logger.info("Candidate config passed health-check.")
    return True
