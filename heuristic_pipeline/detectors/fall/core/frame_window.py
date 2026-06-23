"""
core/frame_window.py
Rolling frame-window buffer for a single tracked person.

Temporal durations are defined in seconds (config.yaml fall_detection section)
and converted to frame counts at init time using the actual video FPS.
"""

from collections import deque
import numpy as np

_DEFAULT_FPS = 30.0

_DEFAULT_FALL_CFG = {
    "window_seconds": 0.5,
    "lying_sustained_seconds": 0.4,
    "head_rest_seconds": 0.4,
    "near_fall_velocity_seconds": 0.33,
    "bend_sustained_seconds": 0.5,
    "near_fall_angle_delta_deg_s": 270,
}


class FrameWindow:
    """
    Maintains a fixed-size sliding window of recent frames for one person.
    Each entry stores the raw keypoint array + pre-computed metrics.
    """

    def __init__(self, fps: float = _DEFAULT_FPS, fall_cfg: dict | None = None):
        self.fps = max(fps, 1.0)
        cfg = {**_DEFAULT_FALL_CFG, **(fall_cfg or {})}

        self.window_size = max(4, round(float(cfg["window_seconds"]) * self.fps))
        self.lying_sustained_frames = max(2, round(float(cfg["lying_sustained_seconds"]) * self.fps))
        self.head_rest_frames = max(2, round(float(cfg["head_rest_seconds"]) * self.fps))
        self.velocity_window_frames = max(2, round(float(cfg["near_fall_velocity_seconds"]) * self.fps))
        self.bend_sustained_frames = max(2, round(float(cfg["bend_sustained_seconds"]) * self.fps))
        self.angle_delta_per_frame = float(cfg["near_fall_angle_delta_deg_s"]) / self.fps

        self._buffer: deque = deque(maxlen=self.window_size)
        self._standing_height_samples: deque = deque(maxlen=30)
        self._standing_hip_y_samples: deque = deque(maxlen=30)
        self._standing_torso_samples: deque = deque(maxlen=30)
        self.standing_height_estimate: float | None = None
        self.standing_hip_y_estimate: float | None = None
        self.standing_torso_length_estimate: float | None = None
        self._nose_y_history: deque = deque(maxlen=self.window_size)

    def recalibrate(self, fps: float, fall_cfg: dict | None = None):
        """Rebuild window sizes when FPS or fall config changes."""
        old = list(self._buffer)
        self.__init__(fps, fall_cfg)
        for entry in old[-self.window_size:]:
            self._buffer.append(entry)

    def push(self, keypoints: np.ndarray, metrics: dict):
        entry = {"keypoints": keypoints, **metrics}
        self._buffer.append(entry)

        nose_y = metrics.get("nose_y")
        if nose_y is not None:
            self._nose_y_history.append(nose_y)

        angle = metrics.get("angle")
        vspan = metrics.get("vertical_span")
        if angle is not None and vspan is not None and angle < 20 and vspan > 50:
            self._standing_height_samples.append(vspan)
            self.standing_height_estimate = float(np.median(self._standing_height_samples))

            hip_y_val = metrics.get("hip_y")
            torso_val = metrics.get("torso_length")
            if hip_y_val is not None:
                self._standing_hip_y_samples.append(hip_y_val)
                self.standing_hip_y_estimate = float(np.median(self._standing_hip_y_samples))
            if torso_val is not None:
                self._standing_torso_samples.append(torso_val)
                self.standing_torso_length_estimate = float(np.median(self._standing_torso_samples))

    def __len__(self):
        return len(self._buffer)

    def latest(self) -> dict | None:
        return self._buffer[-1] if self._buffer else None

    def get_metric_series(self, key: str) -> list:
        return [e.get(key) for e in self._buffer]

    def angle_delta(self, n_frames: int | None = None) -> float | None:
        if n_frames is None:
            n_frames = max(2, round(0.1 * self.fps))
        if len(self._buffer) < n_frames + 1:
            return None
        angles = [
            e["angle"] for e in list(self._buffer)[-(n_frames + 1):]
            if e.get("angle") is not None
        ]
        if len(angles) < 2:
            return None
        return angles[-1] - angles[0]

    def angle_velocity(self) -> float:
        angles = [
            e["angle"] for e in list(self._buffer)[-self.velocity_window_frames:]
            if e.get("angle") is not None
        ]
        if len(angles) < 2:
            return 0.0
        deltas = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        return float(np.mean(deltas))

    def sustained_lying(self) -> bool:
        n = self.lying_sustained_frames
        entries = list(self._buffer)[-n:]
        if len(entries) < n:
            return False
        for e in entries:
            geo = (e.get("lying_score") is not None and e["lying_score"] > 1.5)
            surf = e.get("head_on_surface", False)
            if not (geo or surf):
                return False
        return True

    def sustained_bending(self) -> bool:
        n = self.bend_sustained_frames
        entries = list(self._buffer)[-n:]
        if len(entries) < n:
            return False
        return all(e.get("is_bending", False) for e in entries)

    def head_stillness_variance(self) -> tuple:
        n = self.head_rest_frames
        recent = list(self._nose_y_history)[-n:]
        if len(recent) < max(4, n // 2):
            return None, None, None

        arr = np.array(recent, dtype=float)
        raw_var = float(np.var(arr))

        x = np.arange(len(arr), dtype=float)
        coeffs = np.polyfit(x, arr, 1)
        trend = np.polyval(coeffs, x)
        det_var = float(np.var(arr - trend))

        diffs = np.diff(arr)
        signs = np.sign(diffs)
        nonzero = signs[signs != 0]
        dir_changes = int(np.sum(np.diff(nonzero) != 0)) if len(nonzero) >= 2 else 0

        return raw_var, det_var, dir_changes

    def lying_score_trend(self) -> float:
        scores = [e["lying_score"] for e in self._buffer if e.get("lying_score") is not None]
        return float(np.mean(scores)) if scores else 0.0

    def effective_standing_height(self, frame_height: int, default_ratio: float = 0.7) -> float:
        if self.standing_height_estimate:
            return self.standing_height_estimate
        return frame_height * default_ratio
