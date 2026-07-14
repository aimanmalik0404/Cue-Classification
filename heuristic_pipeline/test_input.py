import sys
import json
import logging
from pathlib import Path

# Add heuristic_pipeline to path
sys.path.insert(0, str(Path(r"d:\Cue-Classification\heuristic_pipeline")))

from service import _handle_frame, _init_camera
from utils.config_loader import load_config

# Configure basic logging
logging.basicConfig(level=logging.INFO)

dummy_input = {
  "type": "ai_detection",
  "camera_id": 4,
  "organization_id": 1,
  "frame_id": "cam4_f1042",
  "frame_sequence": 1042,
  "pts_timestamp": 34.733,
  "frame_shape": [1080, 1920, 3],
  "persons": [
    {
      "track_id": 7,
      "bounding_box": [412.5, 180.2, 598.1, 720.8],
      "confidence": 0.91,
      "pose_keypoints": [
        [505.2, 210.4, 0.95],
        [498.1, 205.3, 0.92],
        [512.4, 206.8, 0.90],
        [490.3, 208.1, 0.88],
        [520.6, 209.5, 0.87],
        [475.8, 280.4, 0.94],
        [535.2, 282.1, 0.93],
        [460.5, 360.8, 0.91],
        [550.3, 365.2, 0.90],
        [455.1, 420.6, 0.89],
        [558.7, 425.3, 0.88],
        [490.2, 450.8, 0.92],
        [520.5, 452.1, 0.91],
        [485.6, 560.4, 0.90],
        [525.8, 562.7, 0.89],
        [480.3, 680.5, 0.88],
        [530.1, 682.9, 0.87]
      ]
    },
    {
      "track_id": 12,
      "bounding_box": [620.0, 190.0, 810.5, 730.2],
      "confidence": 0.88,
      "pose_keypoints": [
        [715.2, 220.4, 0.94],
        [708.1, 215.3, 0.91],
        [722.4, 216.8, 0.89],
        [700.3, 218.1, 0.87],
        [730.6, 219.5, 0.86],
        [685.8, 290.4, 0.93],
        [745.2, 292.1, 0.92],
        [670.5, 370.8, 0.90],
        [760.3, 375.2, 0.89],
        [665.1, 430.6, 0.88],
        [768.7, 435.3, 0.87],
        [700.2, 460.8, 0.91],
        [730.5, 462.1, 0.90],
        [695.6, 570.4, 0.89],
        [735.8, 572.7, 0.88],
        [690.3, 690.5, 0.87],
        [740.1, 692.9, 0.86]
      ]
    }
  ]
}

def run_test():
    # Load base config
    cfg = load_config(r"d:\Cue-Classification\heuristic_pipeline\configs\config.yaml")
    
    # Initialize camera state
    camera_state = _init_camera(
        camera_id=4, 
        base_cfg=cfg, 
        fps=25.0, 
        width=1920, 
        height=1080
    )
    
    # Mock the publisher to capture output instead of sending to Redis
    published_alerts = []
    original_publish = camera_state.pipeline.cue_pub.publish
    
    def mock_publish(alert):
        published_alerts.append(alert)
        original_publish(alert)
        
    camera_state.pipeline.cue_pub.publish = mock_publish
    
    # Process the frame
    print("Processing frame...")
    _handle_frame(dummy_input, camera_state)
    
    # Since most detectors use a rolling window (e.g. 4 seconds), one frame won't trigger an alert.
    # But Scene Quality runs every frame, so we might get context hints.
    # To force any pending windowed evaluations to emit, we call flush=True.
    print("Flushing pipeline...")
    camera_state.pipeline.process_tracks(frame=None, tracks=[], frame_idx=1043, timestamp=34.8, flush=True)
    
    print("\n--- OUTPUT GENERATED ---")
    print(json.dumps(published_alerts, indent=2))
    print(f"Total alerts generated: {len(published_alerts)}")

if __name__ == "__main__":
    run_test()
