"""Evaluate stage: score the trained model on the validation split.

Adapted from the Kaggle notebook "Real-Time Traffic Density Estimation
with YOLOv8" by Farzad Nekouei (credited in REPORT.md).
"""

import json
import subprocess
from pathlib import Path

from ultralytics import YOLO

from src.data_utils import DATA_YAML

MODEL_PATH = Path("models/best.pt")
METRICS_PATH = Path("metrics.json")


def get_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    model = YOLO(str(MODEL_PATH))
    results = model.val(data=str(DATA_YAML), split="val")

    metrics = {
        "precision": round(results.results_dict.get("metrics/precision(B)", 0.0), 4),
        "recall": round(results.results_dict.get("metrics/recall(B)", 0.0), 4),
        "mAP50": round(results.results_dict.get("metrics/mAP50(B)", 0.0), 4),
        "mAP50_95": round(results.results_dict.get("metrics/mAP50-95(B)", 0.0), 4),
        "commit_sha": get_commit_sha(),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
