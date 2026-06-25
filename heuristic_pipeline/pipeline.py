"""heuristic_pipeline/pipeline.py — Encapsulates the core heuristic detection logic.

This isolates the heuristic logic from the YOLO pose generation. You can
instantiate `HeuristicPipeline` and pass pre-computed tracks directly
to `process_tracks()`, bypassing `TrackerManager` completely.
"""
import logging
from pathlib import Path
from typing import Optional

from utils.alert_manager import AlertManager
from utils.cue_publisher import CuePublisher
from detectors.fast_approach.detector import FastApproachDetector
from detectors.crowd_convergence.detector import CrowdConvergenceDetector
from detectors.scene_quality.detector import SceneQualityDetector
from detectors.head_drop.detector import HeadDropDetector
from detectors.fall.detector import FallDetector
from detectors.slumped_posture.detector import SlumpedPostureDetector
from detectors.leaning.detector import LeaningDetector


class HeuristicPipeline:
    def __init__(self, cfg: dict, video_stem: str):
        self.logger = logging.getLogger("HeuristicPipeline")
        
        # 1. Setup publishers
        out_dir = Path(cfg["output"]["output_dir"])
        self.alert_mgr = AlertManager(
            output_dir=str(out_dir / "alerts"),
            save=cfg["output"]["save_alerts"],
        )
        self.cue_pub = CuePublisher(
            cfg=cfg,
            output_dir=str(out_dir / "behavior_cues"),
            video_stem=video_stem,
        )
        self.alert_mgr.cue_publisher = self.cue_pub

        # Rolling per-track bbox cache. This pipeline instance is per-camera (one per
        # camera_id) and the service runs a single-threaded message loop, so track_ids never
        # collide across cameras and the dict needs no locking. MS-1 base_detection is
        # EVENT-DRIVEN (~10% of frames), and person-level cues (head_drop / slumped_posture /
        # leaning / fall) fire as WINDOW predictions at the window boundary — so the subject
        # track is usually ABSENT from the current frame's tracks at fire time. Without this
        # cache, _attach_track_box can't recover the subject's box, the cue ships with no
        # coordinates, and MS-11 falls back to an arbitrary window-snapshot person (wrong / no
        # red box on the alert clip). Keeping each track's most-recently-seen box bridges the
        # detection gaps so the SUBJECT's box always travels with the cue.
        self._recent_boxes: dict[int, list] = {}
        self._recent_box_seen: dict[int, int] = {}
        # `... or 300` (not a .get default) so a null / blank YAML value falls back too.
        self._recent_box_ttl_frames = int(
            cfg.get("pipeline", {}).get("recent_box_ttl_frames") or 300
        )

        # 2. Setup detectors
        self.detectors = {}
        if cfg["fast_approach"]["enabled"]:
            self.detectors["fast_approach"]     = FastApproachDetector(cfg)
        if cfg["crowd_convergence"]["enabled"]:
            self.detectors["crowd_convergence"] = CrowdConvergenceDetector(cfg)
        if cfg["scene_quality"]["enabled"]:
            self.detectors["scene_quality"]     = SceneQualityDetector(cfg)
        if cfg.get("head_drop", {}).get("enabled", True):
            self.detectors["head_drop"]         = HeadDropDetector(cfg)
        if cfg.get("fall_detection", {}).get("enabled", True):
            self.detectors["fall_detection"]    = FallDetector(cfg)
        if cfg.get("slumped_posture", {}).get("enabled", True):
            self.detectors["slumped_posture"]   = SlumpedPostureDetector(cfg)
        if cfg.get("leaning", {}).get("enabled", True):
            self.detectors["leaning"]           = LeaningDetector(cfg)

    def calibrate(self, fps: float, width: int, height: int):
        """Calibrate all active detectors to the video resolution and framerate."""
        for name, det in self.detectors.items():
            det.calibrate(fps, width, height)
            self.logger.info(
                f"  Calibrated [{name}] window={det.window_frames}f "
                f"interval={det.predict_interval_frames}f"
                + (f" vel_thresh={getattr(det, 'velocity_thresh', 0.0):.2f}px/f"
                   if hasattr(det, "velocity_thresh") else "")
            )

    def process_tracks(self, frame, tracks: list, frame_idx: int, timestamp: float, flush: bool = False) -> dict:
        """
        Run all detectors on the provided tracks, fire alerts, and update cue publisher.
        Returns the raw results from each detector for optional visualization.
        """
        results = {}
        
        # 1. Run Detectors
        for name, det in self.detectors.items():
            if flush:
                if not getattr(det, "USE_WINDOW", True):
                    continue
                r = det.run_windowed_detect(
                    frame, tracks, frame_idx, timestamp,
                    update_fn=det._update,
                    predict_fn=det._predict,
                    flush=True,
                )
            else:
                r = det.detect(frame, tracks, frame_idx, timestamp)
            
            results[name] = r if isinstance(r, list) else ([r] if r is not None else [])

        # 2. Fire Alerts
        # Map track_id -> [x1,y1,x2,y2] so person-level cues that carry only a track_id
        # (leaning / fall / head_drop / slumped_posture) still get a bounding box in their
        # detector_signal. Without it the alert clip renders NO red box: those detectors
        # emit just a score + track_id, the overlay extractor needs coordinates, and the
        # cue track ids don't exist in MS-1's ByteTrack cache so the cache lookup also misses.
        track_boxes = {}
        for t in tracks:
            tid = getattr(t, "track_id", None)
            box = getattr(t, "box", None)
            if tid is None or box is None:
                continue
            try:
                track_boxes[int(tid)] = [float(v) for v in (box.tolist() if hasattr(box, "tolist") else box)]
            except (TypeError, ValueError):
                pass

        # Refresh the rolling cache with this frame's boxes, evict stale tracks, then build a
        # lookup where the current frame wins but recently-seen boxes fill the gaps. This lets
        # window-prediction cues whose subject isn't in the current frame still recover the
        # subject's last-seen box (see __init__ for the why).
        for tid, box in track_boxes.items():
            self._recent_boxes[tid] = box
            self._recent_box_seen[tid] = frame_idx
        if self._recent_box_ttl_frames > 0 and self._recent_box_seen:
            stale = [tid for tid, seen in self._recent_box_seen.items()
                     if frame_idx - seen > self._recent_box_ttl_frames]
            for tid in stale:
                self._recent_boxes.pop(tid, None)
                self._recent_box_seen.pop(tid, None)
        box_lookup = {**self._recent_boxes, **track_boxes}

        for name, results_list in results.items():
            for r_item in results_list:
                if not r_item.triggered:
                    continue
                if not r_item.metadata.get("is_window_prediction", True):
                    continue

                metadata = self._attach_track_box(r_item.metadata.get("best", r_item.metadata), box_lookup)

                self.alert_mgr.fire(
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    cue=name,
                    confidence=r_item.confidence,
                    metadata=metadata,
                )

        # 3. Refresh scene quality context
        sq_results = results.get("scene_quality", [])
        if sq_results:
            self.cue_pub.update_quality(sq_results[0].metadata)

        return results

    @staticmethod
    def _attach_track_box(metadata: dict, track_boxes: dict) -> dict:
        """Add the alerted track's [x1,y1,x2,y2] box to a person-level cue's signal when
        the detector didn't emit box coordinates, so the alert clip can draw its red box.
        Leaves alerts that already carry coords (e.g. fast_approach's *_box) untouched."""
        if not isinstance(metadata, dict) or not track_boxes:
            return metadata
        if any(isinstance(k, str) and (k.endswith("_box") or k in ("box", "bbox", "bounding_box"))
               for k in metadata):
            return metadata
        tid = metadata.get("track_id")
        try:
            tid = int(tid) if tid is not None else None
        except (TypeError, ValueError):
            tid = None
        if tid is not None and tid in track_boxes:
            return {**metadata, "box": track_boxes[tid]}
        return metadata

    def get_summary(self) -> tuple[dict, dict]:
        """Return (alert_summary, published_summary)."""
        return self.alert_mgr.summary(), self.cue_pub.summary()
