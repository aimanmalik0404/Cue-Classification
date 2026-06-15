from __future__ import annotations
import numpy as np
import cv2
import math
from collections import defaultdict
from detectors.base_detector import BaseDetector, DetectionResult

L_SHOULDER = 5; R_SHOULDER = 6
L_HIP = 11; R_HIP = 12
CONF_THR = 0.5


class LeaningDetector(BaseDetector):
    CUE_KEY = "leaning"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.enabled = cfg.get("leaning", {}).get("enabled", True)
        self._window_hits: dict[int, list[bool]] = defaultdict(list)

    def _on_calibrate(self):
        c = self.cfg.get("leaning", {})
        self.lean_ratio_thr = c.get("lean_ratio_threshold", 0.25)
        self.lean_angle_thr = c.get("lean_angle_deg", 20)
        self.movement_thr = self.pct_to_px(c.get("movement_threshold_pct", 0.015))
        self.min_hit_ratio = c.get("min_hit_ratio", 0.40)

    def on_window_reset(self, frame_idx: int):
        self._window_hits.clear()

    def detect(self, frame, tracks, frame_idx, timestamp):
        if not self.enabled:
            return []
        return self.run_windowed_detect(
            frame, tracks, frame_idx, timestamp,
            update_fn=self._update,
            predict_fn=self._predict,
        )

    def _update(self, frame, tracks, frame_idx, timestamp):
        for t in tracks:
            is_moving = t.speed > self.movement_thr
            self._window_hits[t.track_id].append(
                self._detect_leaning(t.keypoints, is_moving)
            )

    def _predict(self, frame, tracks, frame_idx, timestamp) -> list[DetectionResult]:
        results = []
        active_ids = {t.track_id for t in tracks}

        for tid in list(self._window_hits.keys()):
            if tid not in active_ids:
                del self._window_hits[tid]

        min_hits = max(1, round(self.min_hit_ratio * self.window_frames))

        for tid, hits in self._window_hits.items():
            if sum(hits) >= min_hits:
                hit_ratio = sum(hits) / max(len(hits), 1)
                # Scale: min_hit_ratio -> 0.45, 1.0 -> 0.90
                span = max(1.0 - self.min_hit_ratio, 1e-6)
                conf = 0.45 + (hit_ratio - self.min_hit_ratio) / span * 0.45
                conf = round(min(0.90, max(0.45, conf)), 3)
                results.append(DetectionResult(
                    triggered=True,
                    confidence=conf,
                    metadata={"track_id": tid, "hit_ratio": round(hit_ratio, 3)},
                ))

        return results

    def _get_kp(self, kp, idx):
        if kp[idx, 2] >= CONF_THR:
            return kp[idx, :2]
        return None

    def _detect_leaning(self, kp: np.ndarray, is_moving: bool) -> bool:
        if is_moving:
            return False

        l_sh = self._get_kp(kp, L_SHOULDER)
        r_sh = self._get_kp(kp, R_SHOULDER)
        l_hp = self._get_kp(kp, L_HIP)
        r_hp = self._get_kp(kp, R_HIP)

        def mid(p1, p2):
            return (p1 + p2) / 2.0 if p1 is not None and p2 is not None else p1 if p1 is not None else p2

        sh_mid = mid(l_sh, r_sh)
        hp_mid = mid(l_hp, r_hp)

        if sh_mid is None or hp_mid is None:
            return False

        torso_height = abs(sh_mid[1] - hp_mid[1])
        if torso_height < 1:
            return False

        lateral_offset = abs(sh_mid[0] - hp_mid[0])
        lean_ratio = lateral_offset / torso_height

        dx = sh_mid[0] - hp_mid[0]
        dy = sh_mid[1] - hp_mid[1]
        torso_angle = math.degrees(math.atan2(abs(dx), abs(dy)))

        return lean_ratio > self.lean_ratio_thr and torso_angle > self.lean_angle_thr

    def draw(self, frame: np.ndarray, result: DetectionResult, tracks: list) -> np.ndarray:
        if "track_id" not in result.metadata:
            return frame
        tid = result.metadata["track_id"]
        track = next((t for t in tracks if t.track_id == tid), None)
        if not track:
            return frame

        box = track.box.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 140, 255), 2)
        cv2.putText(frame, "LEANING", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        return frame
