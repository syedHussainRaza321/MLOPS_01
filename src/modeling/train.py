"""Train stage: fine-tune YOLOv8 on the training split using params.yaml.

Adapted from the Kaggle notebook "Real-Time Traffic Density Estimation
with YOLOv8" by Farzad Nekouei (credited in REPORT.md).
"""

import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from src.data_utils import DATA_YAML

PARAMS_PATH = Path("params.yaml")
MODEL_OUT_PATH = Path("models/best.pt")


def main():
    params = yaml.safe_load(PARAMS_PATH.read_text())
    seed = params["seed"]
    model_params = params["model"]

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=model_params["epochs"],
        imgsz=model_params["imgsz"],
        device=device,
        patience=model_params["patience"],
        batch=model_params["batch"],
        optimizer=model_params["optimizer"],
        lr0=model_params["lr0"],
        lrf=model_params["lrf"],
        dropout=model_params["dropout"],
        seed=seed,
        project="runs/detect",
        name="train_run",
        exist_ok=True,
    )

    # Ask Ultralytics directly where it actually saved the weights, rather
    # than assuming a path — avoids issues if Ultralytics' own global
    # settings alter the effective save location.
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, MODEL_OUT_PATH)
    print(f"Copied best model weights from {best_weights} to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
