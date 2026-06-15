"""utils/windowing.py — Shared window / prediction-interval helpers."""
from __future__ import annotations


def resolve_window_seconds(cfg: dict, cue_key: str) -> tuple[float, float]:
    """
    Return (window_seconds, predict_interval_seconds) for a cue.

    Per-cue values override the global ``pipeline`` defaults.
    """
    pipe = cfg.get("pipeline", {})
    cue = cfg.get(cue_key, {}) if cue_key else {}

    window_sec = cue.get("window_seconds", pipe.get("window_seconds", 4.0))
    interval_sec = cue.get(
        "predict_interval_seconds",
        pipe.get("predict_interval_seconds", window_sec),
    )
    return float(window_sec), float(interval_sec)


def should_predict_frame(
    frame_idx: int,
    window_frames: int,
    predict_interval_frames: int,
) -> bool:
    """True when ``frame_idx`` is a prediction boundary (1-based frame count)."""
    frame_count = frame_idx + 1
    if frame_count < window_frames:
        return False
    return frame_count % predict_interval_frames == 0


def should_flush_on_eof(
    frame_idx: int,
    window_frames: int,
    predict_interval_frames: int,
) -> bool:
    """True when video ended mid-interval and a partial window should flush."""
    frame_count = frame_idx + 1
    if frame_count < window_frames:
        return False
    return frame_count % predict_interval_frames != 0
