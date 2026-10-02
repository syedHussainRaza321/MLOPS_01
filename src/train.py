"""
Vehicle detection pipeline — YOLOv8 training.

Adapted from the Kaggle notebook "Real-Time Traffic Density Estimation
with YOLOv8" by Farzad Nekouei (credited in REPORT.md).

Run from the project root:
    uv run python src/train.py
"""

import torch
from ultralytics import YOLO

from data_utils import DATA_YAML, SEED, check_dataset

# Phase 2 smoke-check values (small/fast) — replaced by params.yaml in Phase 6
EPOCHS = 1
IMGSZ = 640
BATCH = 8
PATIENCE = 50
OPTIMIZER = "auto"
LR0 = 0.0001
LRF = 0.1
DROPOUT = 0.1


def train_model():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=device,
        patience=PATIENCE,
        batch=BATCH,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        dropout=DROPOUT,
        seed=SEED,
        project="runs/detect",
        name="phase2_check",
    )
    print("Training complete. Results saved under runs/detect/phase2_check/")


def main():
    check_dataset()
    train_model()


if __name__ == "__main__":
    main()
