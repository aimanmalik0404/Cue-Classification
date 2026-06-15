"""utils/alert_manager.py — Write JSON alerts to disk and maintain an in-memory log."""
import json
import logging
from pathlib import Path
from datetime import datetime


class AlertManager:
    def __init__(self, output_dir: str = "outputs/alerts", save: bool = True,
                 cue_publisher=None):
        self.save = save
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._alerts: list[dict] = []
        self._log_path = self.output_dir / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.logger = logging.getLogger("AlertManager")
        # Optional CuePublisher — if set, every alert is also formatted
        # and emitted as a behavior_cues message.
        self.cue_publisher = cue_publisher

    def fire(self, timestamp: float, frame_idx: int, cue: str,
             confidence: float, metadata: dict) -> dict:
        alert = {
            "timestamp": round(timestamp, 3),
            "frame_idx": frame_idx,
            "cue": cue,
            "confidence": round(confidence, 3),
            "metadata": metadata,
        }
        self._alerts.append(alert)
        self.logger.info(f"ALERT [{cue}] @ {timestamp:.2f}s  conf={confidence:.2f}  {metadata}")
        if self.save:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(alert) + "\n")

        # Forward to CuePublisher (Kenneth's schema)
        if self.cue_publisher is not None:
            self.cue_publisher.publish(alert)

        return alert

    @property
    def alerts(self) -> list[dict]:
        return self._alerts

    def summary(self) -> dict:
        from collections import Counter
        counts = Counter(a["cue"] for a in self._alerts)
        return {"total": len(self._alerts), "by_cue": dict(counts)}
