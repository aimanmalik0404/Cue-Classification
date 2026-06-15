"""detectors/scene_quality/detector.py

VISIBILITY + LIGHTING + OCCLUSION METRICS
==========================================
Produces scene quality scores. Runs every frame for the live quality panel.
"""
from __future__ import annotations
import numpy as np
import cv2
from collections import deque
from itertools import combinations

from detectors.base_detector import BaseDetector, DetectionResult
from utils.geometry import box_iou


class SceneQualityDetector(BaseDetector):
    CUE_KEY = "scene_quality"
    USE_WINDOW = False

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        sq = cfg["scene_quality"]
        self.brightness_low    = sq["brightness_low"]
        self.brightness_high   = sq["brightness_high"]
        self.contrast_low      = sq["contrast_low"]
        self.blur_thresh       = sq["blur_threshold"]
        self.occlusion_iou     = sq["occlusion_iou_threshold"]
        self._smooth_sec       = sq["quality_smoothing_seconds"]
        self.color_good        = tuple(cfg["visualization"]["quality_good_color"])
        self.color_fair        = tuple(cfg["visualization"]["quality_fair_color"])
        self.color_poor        = tuple(cfg["visualization"]["quality_poor_color"])

        if sq.get("use_window", False) is False:
            self.USE_WINDOW = False

        self.smooth_frames = 10
        self._history: deque = deque(maxlen=self.smooth_frames)

    def _on_calibrate(self):
        self.smooth_frames = self.sec_to_frames(self._smooth_sec)
        self._history = deque(maxlen=self.smooth_frames)

    def detect(self, frame, tracks, frame_idx, timestamp) -> DetectionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        brightness = float(gray.mean())
        contrast   = float(gray.std())
        blur       = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        boxes     = [t.box for t in tracks]
        occ_ratio = 0.0
        if len(boxes) >= 2:
            overlapping = sum(
                1 for a, b in combinations(boxes, 2)
                if box_iou(a, b) >= self.occlusion_iou
            )
            total_pairs = len(boxes) * (len(boxes) - 1) / 2
            occ_ratio   = overlapping / total_pairs

        b_score  = self._brightness_score(brightness)
        c_score  = min(contrast / 80.0, 1.0)
        bl_score = min(blur / (self.blur_thresh * 3), 1.0)
        occ_score= 1.0 - occ_ratio
        visibility = b_score * 0.35 + c_score * 0.25 + bl_score * 0.25 + occ_score * 0.15

        raw = {
            "brightness": round(brightness, 2),
            "contrast":   round(contrast, 2),
            "blur":       round(blur, 2),
            "occlusion":  round(occ_ratio, 3),
            "visibility": round(visibility, 3),
        }
        self._history.append(raw)

        smoothed = {k: round(float(np.mean([h[k] for h in self._history])), 3)
                    for k in raw}
        smoothed["category"] = self._category(smoothed["visibility"])

        return DetectionResult(
            triggered=False,
            confidence=smoothed["visibility"],
            metadata=smoothed,
        )

    def draw(self, frame, result: DetectionResult, tracks) -> np.ndarray:
        vis = frame.copy()
        m   = result.metadata
        if not m:
            return vis

        color = {"GOOD": self.color_good, "FAIR": self.color_fair,
                 "POOR": self.color_poor}.get(m.get("category", "FAIR"), self.color_fair)

        h, w = vis.shape[:2]
        panel_x, panel_y = w - 260, 10
        panel_w, panel_h = 250, 145

        overlay = vis.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        cv2.putText(vis, "SCENE QUALITY", (panel_x + 8, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        lines = [
            f"Brightness : {m.get('brightness', 0):.1f}",
            f"Contrast   : {m.get('contrast', 0):.1f}",
            f"Blur       : {m.get('blur', 0):.1f}",
            f"Occlusion  : {m.get('occlusion', 0)*100:.1f}%",
            f"Visibility : {m.get('visibility', 0)*100:.0f}%  [{m.get('category', '')}]",
        ]
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (panel_x + 8, panel_y + 40 + i * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        bar_y     = panel_y + panel_h - 15
        bar_full_w= panel_w - 16
        bar_fill  = int(bar_full_w * m.get("visibility", 0))
        cv2.rectangle(vis, (panel_x + 8, bar_y), (panel_x + 8 + bar_full_w, bar_y + 8),
                      (60, 60, 60), -1)
        cv2.rectangle(vis, (panel_x + 8, bar_y), (panel_x + 8 + bar_fill, bar_y + 8),
                      color, -1)
        return vis

    def _brightness_score(self, b: float) -> float:
        if b < self.brightness_low:
            return b / self.brightness_low
        if b > self.brightness_high:
            return max(0.0, 1.0 - (b - self.brightness_high) / (255 - self.brightness_high))
        return 1.0

    def _category(self, vis: float) -> str:
        if vis >= 0.65:
            return "GOOD"
        if vis >= 0.40:
            return "FAIR"
        return "POOR"
