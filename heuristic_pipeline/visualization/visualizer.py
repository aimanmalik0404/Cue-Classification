"""visualization/visualizer.py

Central frame annotator.  Draws:
  - Bounding boxes + track IDs
  - Pose skeletons (COCO 17-keypoint layout)
  - Alert banners (top of frame)
  - Passes frame to each detector's draw() method for cue-specific overlays
"""
from __future__ import annotations
import cv2
import numpy as np

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

CUE_LABELS = {
    "fast_approach": "FAST APPROACH",
    "crowd_convergence": "CROWD CONVERGENCE",
    "head_drop": "HEAD DROP",
    "fall_detection": "FALL",
    "slumped_posture": "SLUMPED POSTURE",
    "leaning": "LEANING",
}


class Visualizer:
    def __init__(self, cfg: dict, fps: float = 25.0):
        v = cfg["visualization"]
        self.show_skeletons     = v["show_skeletons"]
        self.show_track_ids     = v["show_track_ids"]
        self.show_quality_panel = v["show_scene_quality_panel"]
        self.banner_frames      = max(1, round(v.get("alert_banner_seconds", 2.0) * fps))
        self.skeleton_color     = tuple(v["skeleton_color"])
        self.track_id_color     = tuple(v["track_id_color"])
        self._banners: list[list] = []

    def annotate(self, frame: np.ndarray, tracks: list,
                 detector_results: dict,
                 detectors: dict) -> np.ndarray:
        vis = frame.copy()

        for name, det in detectors.items():
            result = detector_results.get(name)
            if result is None:
                continue
            if name == "scene_quality" and not self.show_quality_panel:
                continue
            vis = det.draw(vis, result, tracks)

        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)

            if self.show_track_ids:
                cv2.putText(vis, f"ID{t.track_id}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.track_id_color, 1)

            if self.show_skeletons:
                self._draw_skeleton(vis, t.keypoints)

        for evt_name, result in detector_results.items():
            if result and result.triggered:
                if not result.metadata.get("is_window_prediction", True):
                    continue
                banner_color = {
                    "fast_approach":      (0,   0,   220),
                    "crowd_convergence":  (0, 140,   255),
                    "head_drop":          (255, 0,   255),
                    "fall_detection":     (0,   0,   255),
                    "slumped_posture":    (0,   0,   255),
                    "leaning":            (0, 140,   255),
                }.get(evt_name, (0, 0, 200))
                label = CUE_LABELS.get(evt_name, evt_name.upper().replace("_", " "))
                if evt_name == "fall_detection" and result.metadata.get("type"):
                    label = result.metadata["type"].upper().replace("_", " ")
                banner_text = f"!  {label}   conf={result.confidence:.2f}"
                self._banners.append([banner_text, banner_color, self.banner_frames])

        return self._draw_banners(vis)

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray):
        kp = keypoints
        for i, j in SKELETON:
            if kp[i][2] > 0.3 and kp[j][2] > 0.3:
                p1 = (int(kp[i][0]), int(kp[i][1]))
                p2 = (int(kp[j][0]), int(kp[j][1]))
                if p1 == (0, 0) or p2 == (0, 0):
                    continue
                cv2.line(frame, p1, p2, self.skeleton_color, 1, cv2.LINE_AA)
        for i in range(17):
            if kp[i][2] > 0.3:
                pt = (int(kp[i][0]), int(kp[i][1]))
                if pt != (0, 0):
                    cv2.circle(frame, pt, 3, (255, 255, 255), -1)

    def _draw_banners(self, frame: np.ndarray) -> np.ndarray:
        active = []
        y_offset = 40
        for banner in self._banners:
            text, color, remaining = banner
            if remaining <= 0:
                continue
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y_offset - 28), (frame.shape[1], y_offset + 5),
                          color, -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, text, (12, y_offset - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            banner[2] -= 1
            active.append(banner)
            y_offset += 38

        self._banners = [b for b in active if b[2] > 0]
        return frame
