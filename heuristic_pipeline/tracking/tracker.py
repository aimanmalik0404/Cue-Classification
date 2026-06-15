"""tracking/tracker.py

Wraps the Ultralytics built-in ByteTrack integration so the rest of the
pipeline gets a clean, typed interface instead of raw ultralytics result objects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from ultralytics import YOLO


@dataclass
class PersonTrack:
    track_id: int
    box: np.ndarray          # [x1, y1, x2, y2]
    keypoints: np.ndarray    # (17, 3) — x, y, confidence
    centroid: np.ndarray     # (2,)
    confidence: float
    # History maintained by TrackerManager
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=120))
    keypoints_history: deque = field(default_factory=lambda: deque(maxlen=120))

    @property
    def velocity(self) -> np.ndarray:
        """Mean velocity over the last few recorded centroids."""
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


class TrackerManager:
    """
    Runs YOLO + ByteTrack on each frame and returns a list of PersonTrack objects.
    Maintains centroid history per track_id across frames.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        model_path = cfg["model"]["yolo_model"]
        self.device = cfg["model"]["device"]
        self.conf   = cfg["model"]["conf_threshold"]
        self.iou    = cfg["model"]["iou_threshold"]
        self.tracker_cfg = cfg["model"]["tracker"]

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
