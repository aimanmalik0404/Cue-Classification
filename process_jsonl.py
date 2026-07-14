import argparse
import logging
import sys
import time
import json
from pathlib import Path
import numpy as np
from utils.config_loader import load_config
from pipeline import HeuristicPipeline
from tracking.tracker import PersonTrack
def setup_logging(log_dir: str, stem: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
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
    return logging.getLogger("process_jsonl")
def run_jsonl(jsonl_path: str, cfg: dict):
    t_init_start = time.time()
    stem = Path(jsonl_path).stem
    log_dir = str(Path(cfg["output"]["output_dir"]) / "logs")
    logger = setup_logging(log_dir, stem)
    logger.info(f"Starting pipeline on JSONL: {jsonl_path}")
    # For JSONL without video, we need to guess or hardcode FPS and resolution.
    # We will use the hints from config, or default to 25fps / 1080p
    src_fps = float(cfg.get("pipeline", {}).get("fps_hint", 25.0))
    width = int(cfg.get("pipeline", {}).get("width_hint", 1920))
    height = int(cfg.get("pipeline", {}).get("height_hint", 1080))
    
    logger.info(f"Assuming {width}x{height} @ {src_fps:.1f} FPS for calibration")
    pipeline = HeuristicPipeline(cfg, stem)
    pipeline.calibrate(src_fps, width, height)
    logger.info(f"Initialization and Calibration completed in {time.time() - t_init_start:.2f}s")
    logger.info(f"Starting frame processing...")
    t_start = time.time()
    
    # We need a history store for PersonTrack.from_ms1_json
    history_store = {}
    history_maxlen = max(60, round(float(cfg.get("pipeline", {}).get("history_buffer_seconds", 5.0)) * src_fps))
    
    # We will pass a dummy black frame to the pipeline since we don't have video
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame_idx = 0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            frame_data = json.loads(line)
            timestamp = frame_idx / src_fps
            
            # Extract tracks from JSON
            tracks = []
            for person in frame_data.get("persons", []):
                pt = PersonTrack.from_ms1_json(person, history_store, history_maxlen=history_maxlen)
                tracks.append(pt)
                
            # Process heuristics and fire alerts
            results = pipeline.process_tracks(dummy_frame, tracks, frame_idx, timestamp)
            
            if frame_idx % 100 == 0:
                summary_alerts, _ = pipeline.get_summary()
                logger.info(f"  frame {frame_idx}  tracks={len(tracks)}  "
                            f"alerts_so_far={summary_alerts.get('total', 0)}")
                            
            frame_idx += 1
    # Flush partial window at end
    if frame_idx > 0:
        timestamp = (frame_idx - 1) / src_fps
        pipeline.process_tracks(dummy_frame, [], frame_idx - 1, timestamp, flush=True)
    summary, pub_summary = pipeline.get_summary()
    logger.info(f"Done. Alerts: {summary}  |  Published cues: {pub_summary}  |  Time: {time.time() - t_start:.1f}s")
    logger.info(f"behavior_cues JSON → {pipeline.cue_pub._jsonl_path}")
    logger.info(f"behavior_cues JSON -> {pipeline.cue_pub._jsonl_path}")
    return summary
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL through Heuristic Pipeline")
    parser.add_argument("--jsonl",   required=True,                    help="Path to input JSONL file")
    parser.add_argument("--config",  default="configs/config.yaml",    help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # Disable video saving since we don't have video
    cfg["output"]["save_video"] = False
    
    run_jsonl(args.jsonl, cfg)