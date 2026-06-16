"""visualization/visualizer.py

Central frame annotator. Draws:
  - Bounding boxes + track IDs
  - Pose skeletons
  - Active cue highlights directly on the track (fixes the lagging box issue)
  - Alert banners (top of frame)
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

CUE_COLORS = {
    "fast_approach": (0, 165, 255),       # Orange
    "crowd_convergence": (0, 140, 255),   # Deep Orange
    "head_drop": (255, 0, 255),           # Magenta
    "fall_detection": (0, 0, 255),        # Red
    "slumped_posture": (0, 100, 255),     # Orange Red
    "leaning": (0, 200, 255),             # Yellow
    "normal": (140, 140, 140),            # Subdued Gray
}

class Visualizer:
    def __init__(self, cfg: dict, fps: float = 25.0):
        v = cfg.get("visualization", {})
        self.show_skeletons     = v.get("show_skeletons", True)
        self.show_track_ids     = v.get("show_track_ids", True)
        self.banner_frames      = max(1, round(v.get("alert_banner_seconds", 2.0) * fps))
        
        self._banners: list[list] = []
        # active_tracks: dict[track_id, dict(cue, conf, remaining_frames)]
        self._active_tracks: dict[int, dict] = {}

    def annotate(self, frame: np.ndarray, tracks: list,
                 detector_results: dict,
                 detectors: dict) -> np.ndarray:
        vis = frame.copy()

        # 1. Update active alerts state based on latest triggered results
        for evt_name, result in detector_results.items():
            if result and result.triggered and result.metadata.get("is_window_prediction", True):
                track_id = result.metadata.get("track_id")
                label = CUE_LABELS.get(evt_name, evt_name.upper().replace("_", " "))
                if evt_name == "fall_detection" and result.metadata.get("type"):
                    label = result.metadata["type"].upper().replace("_", " ")
                
                # Add banner
                banner_color = CUE_COLORS.get(evt_name, (0, 0, 255))
                banner_text = f"!  {label}   conf={result.confidence:.2f}"
                self._banners.append([banner_text, banner_color, self.banner_frames])

                # Track-level highlight
                if track_id is not None:
                    self._active_tracks[track_id] = {
                        "cue": evt_name,
                        "label": label,
                        "conf": result.confidence,
                        "frames": self.banner_frames
                    }

        # 2. Draw tracks using the clean UI and active state
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            tid = t.track_id
            
            active = self._active_tracks.get(tid)
            if active and active["frames"] > 0:
                color = CUE_COLORS.get(active["cue"], (0, 0, 255))
                thickness = 3
                label_text = f"ID{tid} | {active['label']} ({active['conf']:.2f})"
                active["frames"] -= 1
            else:
                color = CUE_COLORS["normal"]
                thickness = 1
                label_text = f"ID{tid}"

            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            
            # Draw premium label background + text
            if self.show_track_ids or active:
                self._draw_label_box(vis, label_text, x1, max(0, y1 - 6), color)

            # Draw skeleton
            if self.show_skeletons:
                self._draw_skeleton(vis, t.keypoints, color)

        # 3. Cleanup expired track highlights
        self._active_tracks = {k: v for k, v in self._active_tracks.items() if v["frames"] > 0}

        # 4. Draw global banners
        return self._draw_banners(vis)

    def _draw_label_box(self, img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        y_top = max(0, y - th - pad * 2)
        x2 = min(img.shape[1] - 1, x + tw + pad * 2)
        cv2.rectangle(img, (x, y_top), (x2, y_top + th + baseline + pad * 2), (0, 0, 0), -1)
        cv2.putText(
            img, text, (x + pad, y_top + th + pad),
            font, scale, color, thickness, cv2.LINE_AA,
        )

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]):
        kp = keypoints
        for i, j in SKELETON:
            if kp[i][2] > 0.3 and kp[j][2] > 0.3:
                p1 = (int(kp[i][0]), int(kp[i][1]))
                p2 = (int(kp[j][0]), int(kp[j][1]))
                if p1 == (0, 0) or p2 == (0, 0):
                    continue
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
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
            cv2.rectangle(overlay, (0, y_offset - 28), (frame.shape[1], y_offset + 5), color, -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, text, (12, y_offset - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            banner[2] -= 1
            active.append(banner)
            y_offset += 38

        self._banners = [b for b in active if b[2] > 0]
        return frame
