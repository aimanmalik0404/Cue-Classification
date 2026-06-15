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
        for name, results_list in results.items():
            for r_item in results_list:
                if not r_item.triggered:
                    continue
                if not r_item.metadata.get("is_window_prediction", True):
                    continue
                
                self.alert_mgr.fire(
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    cue=name,
                    confidence=r_item.confidence,
                    metadata=r_item.metadata.get("best", r_item.metadata),
                )

        # 3. Refresh scene quality context
        sq_results = results.get("scene_quality", [])
        if sq_results:
            self.cue_pub.update_quality(sq_results[0].metadata)

        return results

    def get_summary(self) -> tuple[dict, dict]:
        """Return (alert_summary, published_summary)."""
        return self.alert_mgr.summary(), self.cue_pub.summary()
