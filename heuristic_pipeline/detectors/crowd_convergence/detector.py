"""detectors/crowd_convergence/detector.py

CROWD CONVERGENCE DETECTOR
===========================
Fires when multiple people simultaneously move toward a common spatial point.

Predictions are emitted once per global observation window, not every frame.
"""
from __future__ import annotations
import numpy as np
import cv2
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN

from detectors.base_detector import BaseDetector, DetectionResult
from utils.geometry import euclidean, cosine_similarity


class CrowdConvergenceDetector(BaseDetector):
    CUE_KEY = "crowd_convergence"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        cc = cfg["crowd_convergence"]
        self._dbscan_eps_pct      = cc["dbscan_eps_pct"]
        self._cooldown_sec        = cc["alert_cooldown_seconds"]
        self._window_sec          = cc["convergence_window_seconds"]
        self.min_crowd            = cc["min_crowd_size"]
        self.dist_ratio           = cc["distance_decrease_ratio"]
        self.vector_thresh        = cc["min_vector_alignment"]
        self.min_samples          = cc["dbscan_min_samples"]
        self.color                = tuple(cfg["visualization"]["crowd_color"])

        self.eps             = 150
        self.cooldown_frames = 60
        self.trend_frames    = 20

        self._window_snapshots: list[dict] = []
        self._cooldown_counter: int = 0

    def _on_calibrate(self):
        self.eps             = self.pct_to_px(self._dbscan_eps_pct)
        self.cooldown_frames = self.sec_to_frames(self._cooldown_sec)
        self.trend_frames    = self.sec_to_frames(self._window_sec)
        self._history_len    = max(self.trend_frames, 60)
        self._pending_alerts = []

    def on_window_reset(self, frame_idx: int):
        pass # We no longer clear history on window reset since it's continuous

    def detect(self, frame, tracks, frame_idx, timestamp):
        return self.run_windowed_detect(
            frame, tracks, frame_idx, timestamp,
            update_fn=self._update,
            predict_fn=self._predict,
        )

    def _cluster_key(self, member_ids: list[int]) -> tuple[int, ...]:
        return tuple(sorted(member_ids))

    def _update(self, frame, tracks, frame_idx, timestamp):
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # maintain history size
        if len(self._window_snapshots) > self._history_len:
            self._window_snapshots.pop(0)

        if len(tracks) < self.min_crowd:
            return

        centroids = np.array([t.centroid for t in tracks])
        labels    = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(centroids)

        for label in set(labels):
            if label == -1:
                continue
            members = [tracks[i] for i, l in enumerate(labels) if l == label]
            if len(members) < self.min_crowd:
                continue

            cluster_cents  = np.array([m.centroid for m in members])
            cluster_center = cluster_cents.mean(axis=0)
            member_ids     = [m.track_id for m in members]

            n = len(members)
            dists = [euclidean(members[i].centroid, members[j].centroid)
                     for i in range(n) for j in range(i + 1, n)]
            avg_dist = float(np.mean(dists)) if dists else 0.0

            self._window_snapshots.append({
                "cluster_key": self._cluster_key(member_ids),
                "member_ids": member_ids,
                "avg_dist": avg_dist,
                "cluster_center": cluster_center.copy(),
                "members": members,
            })

        # --- Continuous Evaluation ---
        if self._cooldown_counter > 0:
            return

        by_cluster: dict[tuple, list] = defaultdict(list)
        for snap in self._window_snapshots:
            by_cluster[snap["cluster_key"]].append(snap)

        cluster_alerts = []
        for cluster_key, snaps in by_cluster.items():
            if len(snaps) < max(3, self.trend_frames // 2):
                continue

            baseline = snaps[0]["avg_dist"]
            latest   = snaps[-1]
            avg_dist = latest["avg_dist"]
            if baseline < 1e-3:
                continue

            ratio = avg_dist / baseline
            if ratio > self.dist_ratio:
                continue

            members = latest["members"]
            cluster_center = latest["cluster_center"]
            n = len(members)

            inward_count = 0
            for m in members:
                vel    = m.velocity
                toward = cluster_center - m.centroid
                if cosine_similarity(vel, toward) > 0.3:
                    inward_count += 1

            alignment_frac = inward_count / n
            if alignment_frac < self.vector_thresh:
                continue

            score = self._compute_confidence(ratio, alignment_frac, n)
            cluster_alerts.append({
                "cluster_key":      cluster_key,
                "member_ids":       latest["member_ids"],
                "num_persons":      n,
                "avg_pairwise_dist": round(avg_dist, 2),
                "dist_ratio":       round(ratio, 3),
                "alignment_frac":   round(alignment_frac, 3),
                "convergence_score": round(score, 3),
                "cluster_center":   cluster_center.tolist(),
                "member_centroids": [m.centroid.tolist() for m in members],
                "member_boxes":     [m.box.tolist() for m in members],
            })

        if cluster_alerts:
            best = max(cluster_alerts, key=lambda x: x["convergence_score"])
            self._cooldown_counter = self.cooldown_frames
            self._pending_alerts.append({
                "confidence": best["convergence_score"],
                "events": cluster_alerts,
                "best": best
            })
            self._window_snapshots.clear()

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

    def draw(self, frame, result: DetectionResult, tracks) -> np.ndarray:
        vis = frame.copy()
        if not result.triggered:
            return vis

        for evt in result.metadata.get("events", []):
            cc    = tuple(map(int, evt["cluster_center"]))
            n     = evt["num_persons"]
            score = evt["convergence_score"]

            for box, cent in zip(evt["member_boxes"], evt["member_centroids"]):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int(cent[0]), int(cent[1])
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.color, 2)
                cv2.arrowedLine(vis, (cx, cy), cc, self.color, 2, tipLength=0.2)

            cv2.circle(vis, cc, 18, self.color, -1)
            cv2.circle(vis, cc, 18, (255, 255, 255), 2)

            pts = np.array([list(map(int, c)) for c in evt["member_centroids"]])
            if len(pts) >= 3:
                hull = cv2.convexHull(pts)
                cv2.polylines(vis, [hull], isClosed=True, color=self.color, thickness=2)

            label = f"CROWD CONVERGENCE  n={n}  score={score:.2f}"
            cv2.putText(vis, label, (cc[0] - 100, max(cc[1] - 25, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)

        return vis

    def _compute_confidence(self, dist_ratio, alignment_frac, n) -> float:
        ratio_score = max(0.0, 1.0 - dist_ratio)
        align_score = alignment_frac
        size_score  = min(n / 6.0, 1.0)
        return ratio_score * 0.45 + align_score * 0.35 + size_score * 0.20
