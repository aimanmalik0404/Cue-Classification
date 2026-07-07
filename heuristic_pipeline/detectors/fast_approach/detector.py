"""detectors/fast_approach/detector.py

FAST APPROACH DETECTOR
======================
Fires when Person A moves rapidly TOWARD Person B.

Predictions are emitted once per global observation window, not every frame.
"""
from __future__ import annotations
import numpy as np
import cv2
from itertools import combinations

from detectors.base_detector import BaseDetector, DetectionResult
from utils.geometry import euclidean, cosine_similarity


class FastApproachDetector(BaseDetector):
    CUE_KEY = "fast_approach"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        fa = cfg["fast_approach"]
        self._velocity_pct        = fa["velocity_threshold_pct"]
        self._dist_decrease_pct   = fa["distance_decrease_pct"]
        self._min_pair_dist_pct   = fa["min_pair_distance_pct"]
        self._max_pair_dist_pct   = fa["max_pair_distance_pct"]
        self._cooldown_sec        = fa["alert_cooldown_seconds"]
        self.min_frames_tracked   = fa["min_frames_tracked"]
        self.direction_thresh     = fa["direction_alignment_threshold"]
        self.color                = tuple(cfg["visualization"]["fast_approach_color"])

        self.velocity_thresh      = 5.0
        self.dist_decrease_thresh = 5.0
        self.min_pair_dist        = 50.0
        self.max_pair_dist        = 600.0
        self.cooldown_frames      = 30

        self._cooldown_counters: dict[tuple, int] = {}
        self._window_dist_history: dict[tuple, list[float]] = {}
        self._pending_alerts = []

    def _on_calibrate(self):
        self.velocity_thresh      = self.pct_per_sec_to_px_per_frame(self._velocity_pct)
        self.dist_decrease_thresh = self.pct_per_sec_to_px_per_frame(self._dist_decrease_pct)
        self.min_pair_dist        = self.pct_to_px(self._min_pair_dist_pct)
        self.max_pair_dist        = self.pct_to_px(self._max_pair_dist_pct)
        self.cooldown_frames      = self.sec_to_frames(self._cooldown_sec)

    def on_window_reset(self, frame_idx: int):
        self._window_dist_history.clear()
        self._pending_alerts.clear()

    def detect(self, frame, tracks, frame_idx, timestamp):
        return self.run_windowed_detect(
            frame, tracks, frame_idx, timestamp,
            update_fn=self._update,
            predict_fn=self._predict,
        )

    def _update(self, frame, tracks, frame_idx, timestamp):
        self._tick_cooldowns()
        if len(tracks) < 2:
            return

        for a, b in combinations(tracks, 2):
            pair = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
            
            if a.frames_tracked < self.min_frames_tracked or b.frames_tracked < self.min_frames_tracked:
                continue
            
            dist = euclidean(a.centroid, b.centroid)
            
            if dist < self.min_pair_dist or dist > self.max_pair_dist:
                continue
            
            dh = self._window_dist_history.setdefault(pair, [])
            dh.append(dist)
            
            if len(dh) < 3:
                continue

            dist_delta = dh[-1] - dh[0]

            for approacher, target in [(a, b), (b, a)]:
                vel   = approacher.velocity
                speed = approacher.speed

                toward_target = target.centroid - approacher.centroid
                alignment     = cosine_similarity(vel, toward_target)

                if dist_delta > -self.dist_decrease_thresh * len(dh):
                    continue

                if speed < self.velocity_thresh:
                    continue

                if alignment < self.direction_thresh:
                    continue

                key = (approacher.track_id, target.track_id)
                if self._cooldown_counters.get(key, 0) > 0:
                    continue

                confidence = self._compute_confidence(speed, abs(dist_delta), alignment, len(dh))
                
                # Cache the alert and keep the highest confidence one for this pair
                # We deduplicate symmetrically so A->B and B->A don't appear as two separate events
                existing = next((alert for alert in self._pending_alerts 
                               if (alert["approacher_id"] == approacher.track_id and alert["target_id"] == target.track_id) or
                                  (alert["approacher_id"] == target.track_id and alert["target_id"] == approacher.track_id)), None)
                if not existing or confidence > existing["confidence"]:
                    if existing:
                        self._pending_alerts.remove(existing)
                    self._pending_alerts.append({
                        "approacher_id":      approacher.track_id,
                        "target_id":          target.track_id,
                        "speed_px_per_frame": round(float(speed), 2),
                        "dist_delta":         round(float(dist_delta), 2),
                        "alignment":          round(float(alignment), 3),
                        "pair_distance":      round(float(dist), 2),
                        "confidence":         round(float(confidence), 3),
                        "approacher_centroid": approacher.centroid.tolist(),
                        "target_centroid":    target.centroid.tolist(),
                        "approacher_box":     approacher.box.tolist(),
                        "target_box":         target.box.tolist(),
                    })
                    
                # We don't set the cooldown here because we are evaluating continuously
                # Cooldowns will be applied when the alert actually fires in _predict

    def _predict(self, frame, tracks, frame_idx, timestamp) -> DetectionResult:
        if not self._pending_alerts:
            return DetectionResult(triggered=False, confidence=0.0, metadata={})
            
        alerts = list(self._pending_alerts)
        self._pending_alerts.clear()
        
        # Apply cooldowns now that we are actually firing the alerts
        for alert in alerts:
            key = (alert["approacher_id"], alert["target_id"])
            self._cooldown_counters[key] = self.cooldown_frames
            
        best = max(alerts, key=lambda x: x["confidence"])
        
        return DetectionResult(
            triggered=True,
            confidence=best["confidence"],
            metadata={"events": alerts, "best": best},
        )

    def draw(self, frame, result: DetectionResult, tracks) -> np.ndarray:
        vis = frame.copy()

        for t in tracks:
            vel = t.velocity
            if t.speed < self.velocity_thresh * 0.5:
                continue
            cx, cy = int(t.centroid[0]), int(t.centroid[1])
            ex = int(cx + vel[0] * 3)
            ey = int(cy + vel[1] * 3)
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (200, 200, 200), 1, tipLength=0.3)

        if not result.triggered:
            return vis

        for evt in result.metadata.get("events", []):
            ac  = tuple(map(int, evt["approacher_centroid"]))
            tc  = tuple(map(int, evt["target_centroid"]))
            ab  = list(map(int, evt["approacher_box"]))
            tb  = list(map(int, evt["target_box"]))

            cv2.rectangle(vis, (ab[0], ab[1]), (ab[2], ab[3]), self.color, 2)
            cv2.rectangle(vis, (tb[0], tb[1]), (tb[2], tb[3]), self.color, 2)

            t_obj = next((t for t in tracks if t.track_id == evt["approacher_id"]), None)
            if t_obj is not None:
                vel = t_obj.velocity
                ex  = int(ac[0] + vel[0] * 5)
                ey  = int(ac[1] + vel[1] * 5)
                cv2.arrowedLine(vis, ac, (ex, ey), self.color, 3, tipLength=0.3)

            cv2.line(vis, ac, tc, self.color, 1, cv2.LINE_AA)

            label = (f"FAST APPROACH  "
                     f"ID{evt['approacher_id']}→ID{evt['target_id']}  "
                     f"conf={evt['confidence']:.2f}  "
                     f"spd={evt['speed_px_per_frame']:.1f}px/f")
            cv2.putText(vis, label, (ab[0], max(ab[1] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.color, 2)

        return vis

    def _compute_confidence(self, speed, dist_delta, alignment, num_frames) -> float:
        s_score = min(speed / (self.velocity_thresh * 3), 1.0)
        d_score = min(abs(dist_delta) / (self.dist_decrease_thresh * num_frames * 3), 1.0)
        a_score = alignment
        # Cap geometry-only confidence at 0.55: approach speed/alignment ALONE is not
        # violence (people merely walking toward each other hit 0.94-1.00 here). The cap is
        # held BELOW every MS-11 aggressive_interaction rule threshold (all >=0.60) so
        # fast_approach is emitted as an informational/corroboration cue but never fires a
        # violent_interaction alert on its own — a real contact cue (push_shove/swing/
        # grapple) is still required to escalate at the MS-11 composition layer.
        # (Was 0.60, which sat ABOVE the old 0.65 publish threshold, so the cue was never
        # emitted and approach events wrongly surfaced as crowd_convergence. Nex fix 2026-07-07.)
        return min(s_score * 0.40 + d_score * 0.35 + a_score * 0.25, 0.55)

    def _tick_cooldowns(self):
        for k in list(self._cooldown_counters):
            self._cooldown_counters[k] = max(0, self._cooldown_counters[k] - 1)
