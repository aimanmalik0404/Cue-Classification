from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from detectors.base_detector import BaseDetector, DetectionResult

NOSE       = 0
L_SHOULDER = 5;  R_SHOULDER = 6
L_HIP      = 11; R_HIP      = 12

CONF_THR = 0.30


class TemporalFilter:
    def __init__(self, confirm_windows: int, reset_windows: int):
        self.confirm_windows = confirm_windows
        self.reset_windows = reset_windows
        self.pos = 0
        self.neg = 0
        self.confirmed = False
        self.last_conf = 0.0

    def update(self, detected: bool, confidence: float):
        if detected:
            self.pos += 1
            self.neg = 0
            self.last_conf = confidence
            if self.pos >= self.confirm_windows:
                self.confirmed = True
        else:
            self.neg += 1
            self.pos = 0
            if self.neg >= self.reset_windows:
                self.confirmed = False
                self.last_conf = 0.0
        return self.confirmed, self.last_conf


class HeadDropDetector(BaseDetector):
    CUE_KEY = "head_drop"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = cfg.get("head_drop", {})
        self.enabled = c.get("enabled", True)
        self._confirm_windows = c.get("confirm_windows", 1)
        self._reset_windows = c.get("reset_windows", 2)
        self._tfilters: dict[int, TemporalFilter] = {}
        self._pending_alerts = []

    def _on_calibrate(self):
        c = self.cfg.get("head_drop", {})
        self.min_torso_px = self.pct_to_px(c.get("min_torso_pct", 0.02))
        vel_per_sec = c.get("max_walking_vel_per_sec", c.get("max_walking_vel_pct", 1.25))
        self.max_walking_vel = float(vel_per_sec) / self.fps

        self.sho_width_thr = c.get("sho_width_threshold", 0.18)
        self.sho_width_persist = c.get("sho_width_persist", 0.60)
        self.nose_sho_thr = c.get("nose_sho_threshold", -0.10)
        self.nose_sho_persist = c.get("nose_sho_persist", 0.55)
        self.nose_sho_change_thr = c.get("nose_sho_change_thr", 0.04)
        self.trunk_angle_gate = c.get("trunk_angle_gate", 1.0)
        self.score_threshold = c.get("score_threshold", 0.55)

        for tid, tf in list(self._tfilters.items()):
            self._tfilters[tid] = TemporalFilter(self._confirm_windows, self._reset_windows)

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

    def _update(self, frame, tracks, frame_idx, timestamp):
        active_ids = {t.track_id for t in tracks}

        for tid in list(self._tfilters.keys()):
            if tid not in active_ids:
                del self._tfilters[tid]

        for t in tracks:
            tid = t.track_id
            if len(t.keypoints_history) < self.window_frames:
                continue

            hist = list(t.keypoints_history)[-self.window_frames:]
            seq_raw = np.stack(hist, axis=0)
            seq_norm = np.stack([self._normalize(kp) for kp in seq_raw], axis=0)

            valid_frames = np.sum(np.sum(seq_norm[:, :, 2] > 0, axis=1) > 0)
            if valid_frames < self.window_frames // 2:
                continue

            seq_smooth = self._smooth(seq_norm)
            det, conf, sigs = self._detect_head_drop_logic(seq_smooth)

            if tid not in self._tfilters:
                self._tfilters[tid] = TemporalFilter(self._confirm_windows, self._reset_windows)
            confirmed, last_conf = self._tfilters[tid].update(det, conf)

            if confirmed:
                new_alert = {
                    "track_id": tid,
                    "confidence": last_conf,
                    **sigs
                }
                existing = next((alert for alert in self._pending_alerts if alert["track_id"] == tid), None)
                if not existing or last_conf > existing["confidence"]:
                    if existing:
                        self._pending_alerts.remove(existing)
                    self._pending_alerts.append(new_alert)

    def _predict(self, frame, tracks, frame_idx, timestamp) -> list[DetectionResult]:
        results = []
        for alert in self._pending_alerts:
            results.append(DetectionResult(
                triggered=True,
                confidence=alert["confidence"],
                metadata={k: v for k, v in alert.items() if k != "confidence"}
            ))
            
        self._pending_alerts.clear()
        return results

    def _normalize(self, kp: np.ndarray) -> np.ndarray:
        norm = kp.copy()
        hip_mid = (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2.0
        sho_mid = (kp[L_SHOULDER, :2] + kp[R_SHOULDER, :2]) / 2.0
        torso = np.linalg.norm(sho_mid - hip_mid)
        if torso < self.min_torso_px:
            norm[:, 2] = 0.0
            return norm
        for j in range(17):
            if kp[j, 2] > CONF_THR:
                norm[j, 0] = (kp[j, 0] - hip_mid[0]) / torso
                norm[j, 1] = (kp[j, 1] - hip_mid[1]) / torso
            else:
                norm[j, 0] = 0.0
                norm[j, 1] = 0.0
        return norm

    def _smooth(self, seq: np.ndarray) -> np.ndarray:
        s = seq.copy()
        for j in range(17):
            s[:, j, 0] = gaussian_filter1d(seq[:, j, 0], sigma=1.2)
            s[:, j, 1] = gaussian_filter1d(seq[:, j, 1], sigma=1.2)
        return s

    def _detect_head_drop_logic(self, seq: np.ndarray) -> tuple[bool, float, dict]:
        T = seq.shape[0]
        if T < 10:
            return False, 0.0, {"status": "collecting", "score": 0.0}

        def shoulder_mid(s): return (s[:, L_SHOULDER, :2] + s[:, R_SHOULDER, :2]) / 2.0
        def hip_mid(s): return (s[:, L_HIP, :2] + s[:, R_HIP, :2]) / 2.0

        hip = hip_mid(seq)
        vmean = float(np.mean(np.linalg.norm(np.diff(hip, axis=0), axis=1)))
        if vmean > self.max_walking_vel:
            return False, 0.0, {"status": "suppressed: vel", "vel_mean": vmean}

        sho = shoulder_mid(seq)
        vec = sho - hip
        ang = np.arctan2(vec[:, 0], -vec[:, 1])
        trunk_angle = float(np.mean(np.abs(ang)))
        if trunk_angle > self.trunk_angle_gate:
            return False, 0.0, {"status": "trunk_gate", "trunk_angle": trunk_angle}

        sho_width = np.abs(seq[:, L_SHOULDER, 0] - seq[:, R_SHOULDER, 0])
        mean_width = float(np.mean(sho_width))
        width_persist = float(np.mean(sho_width < self.sho_width_thr))
        s1_det = (mean_width < self.sho_width_thr and width_persist > self.sho_width_persist)

        nose_y = seq[:, NOSE, 1]
        sho_y = shoulder_mid(seq)[:, 1]
        nose_sho = nose_y - sho_y
        mean_nose_sho = float(np.mean(nose_sho))
        nose_persist = float(np.mean(nose_sho > self.nose_sho_thr))
        s2_det = (mean_nose_sho > self.nose_sho_thr and nose_persist > self.nose_sho_persist)

        early_end = max(2, int(T * 0.25))
        late_start = min(T - 2, int(T * 0.75))
        early_ns = float(np.mean(seq[:early_end, NOSE, 1] - shoulder_mid(seq[:early_end])[:, 1]))
        late_ns = float(np.mean(seq[late_start:, NOSE, 1] - shoulder_mid(seq[late_start:])[:, 1]))
        ns_change = late_ns - early_ns
        s3_det = ns_change > self.nose_sho_change_thr

        if not s1_det:
            return False, 0.0, {"status": "no primary signal", "score": 0.0, "mode": "blocked_no_s1"}

        score = 0.45
        if s2_det:
            score += 0.35
        if s3_det:
            score += 0.20

        detected = score >= self.score_threshold
        return detected, min(1.0, score), {
            "score": score,
            "mode": "width+nose",
            "sho_width": mean_width,
            "nose_sho": mean_nose_sho,
        }

    def draw(self, frame: np.ndarray, result: DetectionResult, tracks: list) -> np.ndarray:
        if "track_id" not in result.metadata:
            return frame
        tid = result.metadata["track_id"]
        track = next((t for t in tracks if t.track_id == tid), None)
        if not track:
            return frame

        box = track.box.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(frame, f"HEAD DROP {result.confidence:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        kp = track.keypoints
        for j in range(17):
            if kp[j, 2] > CONF_THR:
                cv2.circle(frame, (int(kp[j, 0]), int(kp[j, 1])), 4, (255, 0, 255), -1)

        return frame
