from __future__ import annotations
import numpy as np
import cv2
from detectors.base_detector import BaseDetector, DetectionResult
from .lying_detector import LyingDetector
from .near_fall_detector import NearFallDetector
from .core.frame_window import FrameWindow


class FallDetector(BaseDetector):
    CUE_KEY = "fall_detection"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.enabled = cfg.get("fall_detection", {}).get("enabled", True)
        self._fall_cfg = cfg.get("fall_detection", {})

        self.lying_detector = LyingDetector()
        self.near_fall_detector = NearFallDetector()
        self.windows: dict[int, FrameWindow] = {}
        self._pending_alerts = []

    def _on_calibrate(self):
        self._fall_cfg = self.cfg.get("fall_detection", {})
        for w in self.windows.values():
            w.recalibrate(self.fps, self._fall_cfg)

    def on_window_reset(self, frame_idx: int):
        self._pending_alerts.clear()

    def detect(self, frame, tracks, frame_idx, timestamp):
        if not self.enabled:
            return []
        return self.run_windowed_detect(
            frame, tracks, frame_idx, timestamp,
            update_fn=self._update,
            predict_fn=self._predict,
        )

    def _get_window(self, tid: int) -> FrameWindow:
        if tid not in self.windows:
            self.windows[tid] = FrameWindow(self.fps, self._fall_cfg)
        return self.windows[tid]

    def _update(self, frame, tracks, frame_idx, timestamp):
        active_ids = {t.track_id for t in tracks}
        for tid in list(self.windows.keys()):
            if tid not in active_ids:
                del self.windows[tid]

        height = self.frame_height
        for t in tracks:
            window = self._get_window(t.track_id)
            is_lying, lying_ev = self.lying_detector.detect(window, t.keypoints, height)
            is_near_fall, nf_ev = self.near_fall_detector.detect(window, t.keypoints)

            if is_lying or is_near_fall:
                cue_type = "lying_or_collapsed" if is_lying else (
                    "bending" if nf_ev.get("bend_detected") else "near_fall"
                )
                path = lying_ev.get("path") if is_lying else nf_ev.get("path")
                confidence = (
                    self._lying_confidence(lying_ev)
                    if is_lying else
                    self._near_fall_confidence(nf_ev)
                )

                new_alert = {
                    "track_id": t.track_id,
                    "type": cue_type,
                    "path": path,
                    "confidence": confidence,
                    # Carry the tracker's bbox so the cue's detector_signals always has a
                    # renderable box. Without this, near_fall/lying clips rendered
                    # "Bounding Boxes: 0" (the cross-region DetectionCache lookup misses
                    # on Nex pose-tracker ids). Format is [x1,y1,x2,y2] frame-pixel coords.
                    "box": t.box.tolist() if getattr(t, "box", None) is not None else None,
                }

                existing = next((alert for alert in self._pending_alerts if alert["track_id"] == t.track_id), None)
                if not existing or confidence > existing["confidence"]:
                    if existing:
                        self._pending_alerts.remove(existing)
                    self._pending_alerts.append(new_alert)

    def _predict(self, frame, tracks, frame_idx, timestamp) -> list[DetectionResult]:
        results = []
        for alert in self._pending_alerts:
            results.append(DetectionResult(
                triggered=True,
                confidence=alert["confidence"],
                metadata={
                    "track_id": alert["track_id"],
                    "type": alert["type"],
                    "path": alert["path"],
                    **({"box": alert["box"]} if alert.get("box") is not None else {}),
                },
            ))
            
        self._pending_alerts.clear()
        return results

    # ------------------------------------------------------------------
    # Confidence helpers — compute real scores from evidence dicts
    # ------------------------------------------------------------------

    @staticmethod
    def _lying_confidence(ev: dict) -> float:
        """Derive a [0.45, 0.95] confidence from LyingDetector evidence."""
        path = ev.get("path", "geometry")
        if path == "surface_rest":
            # Head stillness: lower variance = more confident
            raw_var = ev.get("head_raw_var") or 200.0
            conf = max(0.45, 1.0 - raw_var / 400.0)
        else:
            # Geometry path: lying_score magnitude + number of secondary signals
            n_signals = sum([
                bool(ev.get("signal_score_pass")),
                bool(ev.get("signal_vspan_pass")),
                bool(ev.get("signal_joints_pass")),
                bool(ev.get("signal_ar_pass")),
            ])
            lying_s = float(ev.get("lying_score") or 0)
            # lying_score threshold is ~1.5; cap at 3.0 for full bonus
            score_bonus = min(lying_s / 3.0, 1.0)
            conf = 0.45 + (n_signals / 4) * 0.30 + score_bonus * 0.20
        return round(min(0.95, conf), 3)

    @staticmethod
    def _near_fall_confidence(ev: dict) -> float:
        """Derive a [0.45, 0.95] confidence from NearFallDetector evidence."""
        angle = float(ev.get("angle") or 20)
        if ev.get("bend_detected"):
            # Bend: deeper angle = higher confidence (threshold is ~25 deg)
            conf = 0.45 + min((angle - 25) / 55.0, 1.0) * 0.45
        else:
            # Dynamic near-fall: angle depth + number of instability signals
            angle_score = min(1.0, max(0.0, (angle - 20) / 40.0))
            instability = sum([
                bool(ev.get("rapid_tilt")),
                bool(ev.get("arms_raised")),
                bool(ev.get("com_off_balance")),
            ])
            conf = 0.45 + angle_score * 0.30 + (instability / 3) * 0.20
        return round(min(0.95, max(0.45, conf)), 3)

    def draw(self, frame: np.ndarray, result: DetectionResult, tracks: list) -> np.ndarray:
        if "track_id" not in result.metadata:
            return frame
        tid = result.metadata["track_id"]
        track = next((t for t in tracks if t.track_id == tid), None)
        if not track:
            return frame

        ctype = result.metadata["type"]
        box = track.box.astype(int)

        if ctype == "lying_or_collapsed":
            color = (0, 0, 255)
            label = "LYING / COLLAPSED"
        elif ctype == "near_fall":
            color = (0, 165, 255)
            label = "NEAR FALL"
        elif ctype == "bending":
            color = (0, 220, 180)
            label = "BENDING"
        else:
            return frame

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
