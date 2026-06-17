"""tracking/tracker.py

Wraps the Ultralytics built-in ByteTrack integration so the rest of the
pipeline gets a clean, typed interface instead of raw ultralytics result objects.

Also exposes PersonTrack.from_ms1_json() for the live Redis cue service,
which receives pre-computed tracks from MS-1 instead of running YOLO locally.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional
import numpy as np


@dataclass
class PersonTrack:
    track_id: int
    box: np.ndarray          # [x1, y1, x2, y2]
    keypoints: np.ndarray    # (17, 3) — x, y, confidence
    centroid: np.ndarray     # (2,)
    confidence: float
    # History maintained by TrackerManager or the live service
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=120))
    keypoints_history: deque = field(default_factory=lambda: deque(maxlen=120))
    # Optional: pre-computed velocity from MS-1 (bypasses history-based calculation)
    _ms1_velocity: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def velocity(self) -> np.ndarray:
        """Pre-computed velocity from MS-1 if available, else derive from history."""
        if self._ms1_velocity is not None:
            return self._ms1_velocity
        h = list(self.centroid_history)[-5:]
        if len(h) < 2:
            return np.zeros(2)
        diffs = [np.array(h[i + 1]) - np.array(h[i]) for i in range(len(h) - 1)]
        return np.mean(diffs, axis=0)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def frames_tracked(self) -> int:
        return len(self.centroid_history)

    @staticmethod
    def from_ms1_json(person: dict, history_store: dict, history_maxlen: int = 120) -> "PersonTrack":
        """
        Build a PersonTrack from a single persons[] entry in Kenneth's ai_detection JSON.

        Args:
            person:         One element from msg["persons"].
            history_store:  A dict keyed by track_id holding persistent deques across frames.
                            Pass a module-level dict from service.py so history accumulates.
            history_maxlen: Max frames of history to keep (default 120).
        """
        tid = int(person["track_id"])
        box_raw = person["bounding_box"]  # [x1, y1, x2, y2]
        box = np.array(box_raw, dtype=float)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centroid = np.array([cx, cy])

        # Build (17, 3) keypoints array — pad with zeros if fewer than 17 given
        kps_raw = person.get("pose_keypoints", [])
        kps = np.zeros((17, 3), dtype=float)
        for i, kp in enumerate(kps_raw[:17]):
            if len(kp) >= 2:
                kps[i, 0] = kp[0]
                kps[i, 1] = kp[1]
                kps[i, 2] = kp[2] if len(kp) > 2 else 1.0

        # Native velocity/speed from MS-1 (avoids cold-start history lag)
        vel_raw = person.get("velocity")  # [vx, vy] or None
        ms1_vel = np.array(vel_raw, dtype=float) if vel_raw is not None else None

        # Maintain persistent centroid history across frames
        if tid not in history_store:
            history_store[tid] = {
                "centroids": deque(maxlen=history_maxlen),
                "keypoints": deque(maxlen=history_maxlen),
            }
        history_store[tid]["centroids"].append(centroid.copy())
        history_store[tid]["keypoints"].append(kps.copy())

        return PersonTrack(
            track_id=tid,
            box=box,
            keypoints=kps,
            centroid=centroid,
            confidence=float(person.get("confidence", 1.0)),
            centroid_history=history_store[tid]["centroids"],
            keypoints_history=history_store[tid]["keypoints"],
            _ms1_velocity=ms1_vel,
        )


class TrackerManager:
    """
    Runs YOLO + BoT-SORT/ByteTrack on each frame and returns a list of PersonTrack objects.
    Maintains centroid history per track_id across frames.

    NOTE: In the live Redis cue service (service.py), TrackerManager is NOT used.
    Instead, PersonTrack.from_ms1_json() parses each MS-1 frame directly.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        model_path = cfg["model"]["yolo_model"]
        self.device = cfg["model"]["device"]
        self.conf   = cfg["model"]["conf_threshold"]
        self.iou    = cfg["model"]["iou_threshold"]
        self.tracker_cfg = cfg["model"]["tracker"]

        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self._history_maxlen = 120
        self._histories: dict[int, deque] = {}
        self._kp_histories: dict[int, deque] = {}

    def calibrate(self, fps: float, width: int, height: int):
        """Size history buffers to cover the global observation window."""
        _ = (width, height)
        pipe = self.cfg.get("pipeline", {})
        buffer_sec = float(pipe.get("history_buffer_seconds", 5.0))
        self._history_maxlen = max(60, round(buffer_sec * fps))

        for tid in list(self._histories):
            self._histories[tid] = deque(self._histories[tid], maxlen=self._history_maxlen)
            self._kp_histories[tid] = deque(
                self._kp_histories[tid], maxlen=self._history_maxlen
            )

    def _get_history(self, tid: int) -> deque:
        if tid not in self._histories:
            self._histories[tid] = deque(maxlen=self._history_maxlen)
            self._kp_histories[tid] = deque(maxlen=self._history_maxlen)
        return self._histories[tid]

    def update(self, frame: np.ndarray) -> list[PersonTrack]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            tracker=self.tracker_cfg,
            verbose=False,
        )

        tracks: list[PersonTrack] = []
        if results is None or len(results) == 0:
            return tracks

        r = results[0]

        if r.boxes is None or r.boxes.id is None:
            return tracks

        boxes     = r.boxes.xyxy.cpu().numpy()
        ids       = r.boxes.id.cpu().numpy().astype(int)
        confs     = r.boxes.conf.cpu().numpy()
        keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        kp_conf   = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else None

        for i, (box, tid, conf) in enumerate(zip(boxes, ids, confs)):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            c  = np.array([cx, cy])
            tid = int(tid)

            if keypoints is not None and i < len(keypoints):
                kp_xy  = keypoints[i]
                kp_c   = kp_conf[i][:, None]
                kps    = np.concatenate([kp_xy, kp_c], axis=1)
            else:
                kps = np.zeros((17, 3))

            self._get_history(tid)
            self._histories[tid].append(c.copy())
            self._kp_histories[tid].append(kps.copy())

            pt = PersonTrack(
                track_id=tid,
                box=box,
                keypoints=kps,
                centroid=c,
                confidence=float(conf),
                centroid_history=self._histories[tid],
                keypoints_history=self._kp_histories[tid],
            )
            tracks.append(pt)

        return tracks
