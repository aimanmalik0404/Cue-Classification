"""detectors/base_detector.py — Abstract base class for all cue detectors."""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union
import numpy as np

from utils.windowing import resolve_window_seconds, should_predict_frame


@dataclass
class DetectionResult:
    triggered: bool
    confidence: float
    metadata: dict = field(default_factory=dict)


ResultType = Union[DetectionResult, list[DetectionResult]]


class BaseDetector(ABC):
    """
    Every cue detector inherits this class and implements ``detect()``.

    ``calibrate(fps, width, height)`` MUST be called once after opening the video.
    It converts all config values from camera-agnostic units
    (% of frame diagonal, seconds) into actual pixel/frame values.

    Windowed detectors set ``CUE_KEY`` and ``USE_WINDOW = True``. They accumulate
    state every frame but only emit predictions (and alerts) at window boundaries.
    """

    CUE_KEY: str = ""
    USE_WINDOW: bool = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = True
        self.fps: float = 25.0
        self.frame_diagonal: float = 1.0
        self.frame_width: int = 1
        self.frame_height: int = 1

        self.window_frames: int = 100
        self.predict_interval_frames: int = 100
        self._window_seconds: float = 4.0
        self._predict_interval_seconds: float = 4.0

        # Last prediction kept for visualization between boundaries
        self._held_results: list[DetectionResult] = []

    def calibrate(self, fps: float, width: int, height: int):
        """Convert camera-agnostic config units to pixel/frame values."""
        self.fps = fps
        self.frame_width = width
        self.frame_height = height
        self.frame_diagonal = math.sqrt(width ** 2 + height ** 2)
        self._calibrate_window()
        self._on_calibrate()

    def _calibrate_window(self):
        if self.CUE_KEY:
            ws, ps = resolve_window_seconds(self.cfg, self.CUE_KEY)
        else:
            pipe = self.cfg.get("pipeline", {})
            ws = float(pipe.get("window_seconds", 4.0))
            ps = float(pipe.get("predict_interval_seconds", ws))

        self._window_seconds = ws
        self._predict_interval_seconds = ps
        self.window_frames = self.sec_to_frames(ws)
        self.predict_interval_frames = self.sec_to_frames(ps)

    def _on_calibrate(self):
        """Override in subclasses to recompute derived thresholds after calibration."""
        pass

    # helpers ──────────────────────────────────────────────────────────────────
    def pct_to_px(self, pct: float) -> float:
        """Convert a fraction-of-diagonal percentage to pixels."""
        return pct * self.frame_diagonal

    def sec_to_frames(self, seconds: float) -> int:
        """Convert seconds to a frame count."""
        return max(1, round(seconds * self.fps))

    def pct_per_sec_to_px_per_frame(self, pct_per_sec: float) -> float:
        """
        Convert a speed expressed as '% of diagonal per second'
        into 'pixels per frame'.
        """
        return (pct_per_sec * self.frame_diagonal) / self.fps

    # window helpers ───────────────────────────────────────────────────────────
    def should_predict(self, frame_idx: int) -> bool:
        if not self.USE_WINDOW:
            return True
        return should_predict_frame(
            frame_idx, self.window_frames, self.predict_interval_frames
        )

    def _empty_results(self) -> list[DetectionResult]:
        return [DetectionResult(triggered=False, confidence=0.0, metadata={})]

    def _normalize_results(self, result: ResultType) -> list[DetectionResult]:
        if isinstance(result, list):
            return result
        return [result]

    def _tag_window_prediction(self, results: list[DetectionResult]) -> list[DetectionResult]:
        for r in results:
            r.metadata["is_window_prediction"] = True
        return results

    def _held_for_display(self) -> list[DetectionResult]:
        """Return last window prediction for overlays; never fires new alerts."""
        if not self._held_results:
            return self._empty_results()
        display = []
        for r in self._held_results:
            display.append(DetectionResult(
                triggered=r.triggered,
                confidence=r.confidence,
                metadata={**r.metadata, "is_window_prediction": False},
            ))
        return display

    def run_windowed_detect(
        self,
        frame: np.ndarray,
        tracks: list,
        frame_idx: int,
        timestamp: float,
        *,
        update_fn,
        predict_fn,
        flush: bool = False,
    ) -> ResultType:
        """
        Standard window orchestration used by most detectors.

        ``update_fn`` runs every frame to accumulate state.
        ``predict_fn`` runs only at window boundaries (or EOF flush).
        """
        update_fn(frame, tracks, frame_idx, timestamp)

        if not self.USE_WINDOW:
            results = self._normalize_results(predict_fn(frame, tracks, frame_idx, timestamp))
            self._held_results = results
            return results

        if not (self.should_predict(frame_idx) or flush):
            return self._held_for_display()

        results = self._tag_window_prediction(
            self._normalize_results(predict_fn(frame, tracks, frame_idx, timestamp))
        )
        self._held_results = results
        self.on_window_reset(frame_idx)
        return results

    def on_window_reset(self, frame_idx: int):
        """Override to clear per-window accumulators after a prediction."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray, tracks: list, frame_idx: int,
               timestamp: float) -> ResultType:
        ...

    @abstractmethod
    def draw(self, frame: np.ndarray, result: DetectionResult,
             tracks: list) -> np.ndarray:
        ...
