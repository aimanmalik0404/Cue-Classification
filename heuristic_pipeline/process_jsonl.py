import argparse
import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from utils.config_loader import load_config
from pipeline import HeuristicPipeline
from tracking.tracker import PersonTrack


def _input_stem(path: Path) -> str:
    """Drop compression suffix so file.jsonl.gz -> file, not file.jsonl."""
    if path.suffix.lower() == ".gz":
        return Path(path.stem).stem
    return path.stem


def _inner_suffix(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        return Path(path.stem).suffix.lower()
    return path.suffix.lower()


def _open_input(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _unwrap_frame(record: dict) -> Optional[dict]:
    """Normalize one input record to an ai_detection frame, or skip it."""
    kind = record.get("kind")
    if kind == "base_detection":
        data = record.get("data")
        if isinstance(data, str):
            return json.loads(data)
        if isinstance(data, dict):
            return data
        return None
    if kind:
        return None

    if record.get("type") == "ai_detection" or "persons" in record:
        return record
    return None


def _iter_input_frames(path: Path) -> Iterator[dict]:
    inner = _inner_suffix(path)

    with _open_input(path) as f:
        if inner == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame = _unwrap_frame(json.loads(line))
                if frame is not None:
                    yield frame
            return

        if inner != ".json":
            raise ValueError(
                "Unsupported input format. Expected .json, .jsonl, or .jsonl.gz"
            )

        data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "frames" in data:
            records = data["frames"]
        else:
            raise ValueError(
                "Unsupported JSON format.\n"
                "Expected either:\n"
                "1. A list of frame objects\n"
                "2. {'frames': [...]}"
            )

        for record in records:
            frame = _unwrap_frame(record) if isinstance(record, dict) else None
            if frame is None and isinstance(record, dict) and "persons" in record:
                frame = record
            if frame is not None:
                yield frame


def _group_frames_by_camera(frames: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        camera_id = frame.get("camera_id", "unknown")
        grouped[str(camera_id)].append(frame)
    return grouped


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

    return logging.getLogger("process_input")


def _process_camera_frames(
    *,
    camera_id: str,
    frames: list[dict],
    stem: str,
    cfg: dict,
    src_fps: float,
    width: int,
    height: int,
    logger: logging.Logger,
) -> dict:
    pipeline = HeuristicPipeline(cfg, stem)
    pipeline.calibrate(src_fps, width, height)

    history_store = {}
    history_maxlen = max(
        60,
        round(
            float(cfg.get("pipeline", {}).get("history_buffer_seconds", 5.0))
            * src_fps
        ),
    )

    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame_idx = 0
    t_start = time.time()

    for frame_data in frames:
        timestamp = frame_idx / src_fps
        tracks = [
            PersonTrack.from_ms1_json(
                person,
                history_store,
                history_maxlen=history_maxlen,
            )
            for person in frame_data.get("persons", [])
        ]

        pipeline.process_tracks(
            dummy_frame,
            tracks,
            frame_idx,
            timestamp,
        )

        if frame_idx % 100 == 0:
            summary_alerts, _ = pipeline.get_summary()
            logger.info(
                f"  camera={camera_id} frame={frame_idx}  "
                f"tracks={len(tracks)}  "
                f"alerts_so_far={summary_alerts.get('total', 0)}"
            )

        frame_idx += 1

    if frame_idx > 0:
        timestamp = (frame_idx - 1) / src_fps
        pipeline.process_tracks(
            dummy_frame,
            [],
            frame_idx - 1,
            timestamp,
            flush=True,
        )

    summary, pub_summary = pipeline.get_summary()
    logger.info(
        f"Camera {camera_id} done. Alerts: {summary}  |  "
        f"Published cues: {pub_summary}  |  "
        f"Time: {time.time() - t_start:.1f}s"
    )
    logger.info(f"behavior_cues JSON -> {pipeline.cue_pub._jsonl_path}")
    return summary


def run_input(input_path: str, cfg: dict):
    t_init_start = time.time()

    path = Path(input_path)
    base_stem = _input_stem(path)

    log_dir = str(Path(cfg["output"]["output_dir"]) / "logs")
    logger = setup_logging(log_dir, base_stem)

    logger.info(f"Starting pipeline on: {input_path}")

    src_fps = float(cfg.get("pipeline", {}).get("fps_hint", 25.0))
    width = int(cfg.get("pipeline", {}).get("width_hint", 1920))
    height = int(cfg.get("pipeline", {}).get("height_hint", 1080))

    logger.info(f"Assuming {width}x{height} @ {src_fps:.1f} FPS for calibration")

    frames = list(_iter_input_frames(path))
    if not frames:
        raise ValueError(f"No ai_detection frames found in {input_path}")

    grouped = _group_frames_by_camera(frames)
    multi_camera = len(grouped) > 1

    logger.info(
        f"Loaded {len(frames)} frames across {len(grouped)} camera(s): "
        f"{', '.join(sorted(grouped))}"
    )
    logger.info(
        f"Initialization completed in {time.time() - t_init_start:.2f}s"
    )
    logger.info("Starting frame processing...")

    summaries = {}
    for camera_id in sorted(grouped):
        cam_stem = f"{base_stem}_cam{camera_id}" if multi_camera else base_stem
        summaries[camera_id] = _process_camera_frames(
            camera_id=camera_id,
            frames=grouped[camera_id],
            stem=cam_stem,
            cfg=cfg,
            src_fps=src_fps,
            width=width,
            height=height,
            logger=logger,
        )

    return summaries if multi_camera else summaries[next(iter(grouped))]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process JSON or JSONL through the Heuristic Pipeline"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .json, .jsonl, or .jsonl.gz file",
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Disable video saving since we don't have the source video
    cfg["output"]["save_video"] = False

    run_input(args.input, cfg)