"""
detectors/person_tracker.py
Maps YOLO track IDs → FrameWindow + runs both detectors per person per frame.
Returns per-person label decisions for the visualiser.

FPS is passed in at construction time so every FrameWindow converts its
second-based thresholds to the correct frame counts for this video.
"""

from collections import defaultdict
import numpy as np

from core.frame_window import FrameWindow
from detectors.lying_detector import LyingDetector
from detectors.near_fall_detector import NearFallDetector

# Label constants
LABEL_NORMAL             = "NORMAL"
LABEL_BENDING            = "BENDING"
LABEL_NEAR_FALL          = "NEAR_FALL"
LABEL_LYING_OR_COLLAPSED = "LYING_OR_COLLAPSED"


class PersonTracker:
    """
    Maintains one FrameWindow per tracked person and orchestrates
    both detectors. Call `process_frame()` once per frame.

    Parameters
    ----------
    fps : float
        Actual FPS of the input video. Passed to every FrameWindow so all
        temporal thresholds are in real seconds, not frame counts.
    """

    def __init__(self, fps: float = 30.0):
        self.fps          = fps
        self._lying_det   = LyingDetector()
        self._near_fall_det = NearFallDetector()
        # Each new track ID gets its own FPS-calibrated window on first access
        self._windows: dict[int, FrameWindow] = {}

    def _get_window(self, track_id: int) -> FrameWindow:
        if track_id not in self._windows:
            self._windows[track_id] = FrameWindow(fps=self.fps)
        return self._windows[track_id]

    # ─── Main entry point ─────────────────────────────────────────────────────

    def process_frame(
        self,
        track_id: int,
        keypoints: np.ndarray,
        frame_height: int,
    ) -> dict:
        window = self._get_window(track_id)

        is_lying, lying_ev = self._lying_det.detect(window, keypoints, frame_height)

        if is_lying:
            is_near_fall, nf_ev = False, {}
        else:
            is_near_fall, nf_ev = self._near_fall_det.detect(window, keypoints)

        if is_lying:
            label = LABEL_LYING_OR_COLLAPSED
        elif is_near_fall:
            nf_path = nf_ev.get("path", "")
            label   = LABEL_BENDING if nf_path == "bend" else LABEL_NEAR_FALL
        else:
            label = LABEL_NORMAL

        return {
            "track_id":           track_id,
            "label":              label,
            "lying_evidence":     lying_ev,
            "near_fall_evidence": nf_ev,
        }

    def active_ids(self):
        return list(self._windows.keys())

    def prune_stale(self, active_ids: list[int]):
        stale = [tid for tid in self._windows if tid not in active_ids]
        for tid in stale:
            del self._windows[tid]
