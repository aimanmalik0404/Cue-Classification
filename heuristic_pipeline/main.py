"""main.py — Inference pipeline entry point.

Usage:
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --config configs/config.yaml --device cuda:1
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from utils.config_loader import load_config
from tracking.tracker import TrackerManager
from pipeline import HeuristicPipeline
from visualization.visualizer import Visualizer


# ---------------------------------------------------------------------------
def setup_logging(log_dir: str, video_stem: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{video_stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("main")


# ---------------------------------------------------------------------------
def build_video_writer(output_path: str, fps: float, width: int, height: int, codec: str):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# ---------------------------------------------------------------------------


def _results_for_visualization(results: dict) -> dict:
    """Pick display result per cue (first triggered, else first item)."""
    viz = {}
    for name, results_list in results.items():
        if not results_list:
            viz[name] = None
            continue
        viz[name] = next(
            (r for r in results_list if r.triggered),
            results_list[0],
        )
    return viz


# ---------------------------------------------------------------------------
def run(video_path: str, cfg: dict, display: bool = False, no_viz: bool = False) -> dict:
    """
    Process one video file.
    Returns the alert summary dict so batch runners can aggregate results.
    """
    t_init_start = time.time()
    stem    = Path(video_path).stem
    log_dir = str(Path(cfg["output"]["output_dir"]) / "logs")
    logger  = setup_logging(log_dir, stem)
    logger.info(f"Starting pipeline on: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {width}x{height} @ {src_fps:.1f} FPS  ({total_frames} frames)")

    pipe = cfg.get("pipeline", {})
    window_sec = pipe.get("window_seconds", 4.0)
    interval_sec = pipe.get("predict_interval_seconds", window_sec)
    logger.info(
        f"Window: {window_sec}s (~{round(window_sec * src_fps)} frames)  "
        f"predict every {interval_sec}s"
    )

    tracker = TrackerManager(cfg)
    pipeline = HeuristicPipeline(cfg, stem)

    tracker.calibrate(src_fps, width, height)
    pipeline.calibrate(src_fps, width, height)

    if not no_viz:
        visualizer = Visualizer(cfg, fps=src_fps)

    writer = None
    if cfg["output"]["save_video"] and not no_viz:
        out_p  = str(Path(cfg["output"]["output_dir"]) / "videos" / f"{stem}_annotated.mp4")
        writer = build_video_writer(out_p, src_fps, width, height, cfg["output"]["video_codec"])
        logger.info(f"Writing annotated video to: {out_p}")

    frame_idx = 0
    init_time = time.time() - t_init_start
    logger.info(f"Initialization and Calibration completed in {init_time:.2f}s")
    logger.info(f"Starting frame processing...")

    t_start   = time.time()  # Reset t_start for pure processing time
    last_frame = None
    last_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame
        timestamp = frame_idx / src_fps
        last_tracks = tracker.update(frame)

        results = pipeline.process_tracks(frame, last_tracks, frame_idx, timestamp)

        if not no_viz:
            viz_results = _results_for_visualization(results)
            annotated = visualizer.annotate(frame, last_tracks, viz_results, pipeline.detectors)
        else:
            annotated = frame

        elapsed  = time.time() - t_start
        proc_fps = (frame_idx + 1) / max(elapsed, 1e-6)
        cv2.putText(
            annotated,
            f"Frame {frame_idx}/{total_frames}  |  {proc_fps:.1f} FPS  |  {timestamp:.1f}s",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
        )

        if writer:
            writer.write(annotated)

        if display:
            cv2.imshow("CV Pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User quit.")
                break

        if frame_idx % 100 == 0:
            summary_alerts, _ = pipeline.get_summary()
            logger.info(f"  frame {frame_idx}/{total_frames}  tracks={len(last_tracks)}  "
                        f"alerts_so_far={summary_alerts.get('total', 0)}")

        frame_idx += 1

    # Flush partial window at end of video
    if last_frame is not None and frame_idx > 0:
        timestamp = (frame_idx - 1) / src_fps
        pipeline.process_tracks(last_frame, last_tracks, frame_idx - 1, timestamp, flush=True)

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    summary, pub_summary = pipeline.get_summary()
    logger.info(f"Done. Alerts: {summary}  |  Published cues: {pub_summary}  |  Time: {time.time() - t_start:.1f}s")
    logger.info(f"behavior_cues JSON → {pipeline.cue_pub._jsonl_path}")
    return summary


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV Behavioral Cue Detection Pipeline")
    parser.add_argument("--video",   required=True,                    help="Path to input video")
    parser.add_argument("--config",  default="configs/config.yaml",    help="Path to YAML config")
    parser.add_argument("--device",  default=None,                     help="Override CUDA device")
    parser.add_argument("--display", action="store_true",              help="Show live preview window")
    parser.add_argument("--no-save", action="store_true",              help="Disable video/alert saving")
    parser.add_argument("--no-viz", action="store_true",               help="Disable visualization drawing for faster processing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["model"]["device"] = args.device
    if args.no_save:
        cfg["output"]["save_video"]  = False
        cfg["output"]["save_alerts"] = False

    run(args.video, cfg, display=args.display, no_viz=args.no_viz)
