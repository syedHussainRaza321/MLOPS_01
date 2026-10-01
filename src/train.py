"""
Vehicle detection pipeline — dataset validation + YOLOv8 training.

Adapted from the Kaggle notebook "Real-Time Traffic Density Estimation
with YOLOv8" by Farzad Nekouei (credited in REPORT.md). Visualization,
inference-on-video, and the traffic-density-counting logic from that
notebook are NOT included here — they belong in the Phase 5 EDA notebook
or as a separate optional demo script, not the core training pipeline.

Run from the project root:
    python src/train.py
"""

import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from ultralytics import YOLO

DATASET_DIR = Path("data/raw/Vehicle_Detection_Image_Dataset")
DATA_YAML = DATASET_DIR / "data.yaml"
SEED = 42

# Phase 2 smoke-check values (small/fast) — replaced by params.yaml in Phase 6
EPOCHS = 1
IMGSZ = 640
BATCH = 8
PATIENCE = 50
OPTIMIZER = "auto"
LR0 = 0.0001
LRF = 0.1
DROPOUT = 0.1


def count_and_check_sizes(images_path: Path):
    sizes = set()
    count = 0
    for filename in os.listdir(images_path):
        if filename.endswith(".jpg"):
            count += 1
            with Image.open(images_path / filename) as img:
                sizes.add(img.size)
    return count, sizes


def check_dataset():
    assert DATA_YAML.exists(), f"data.yaml not found at {DATA_YAML}"

    with open(DATA_YAML) as f:
        yaml_content = yaml.safe_load(f)
    print("data.yaml contents:")
    print(yaml.dump(yaml_content, default_flow_style=False))

    train_images_path = DATASET_DIR / "train" / "images"
    valid_images_path = DATASET_DIR / "valid" / "images"

    num_train, train_sizes = count_and_check_sizes(train_images_path)
    num_valid, valid_sizes = count_and_check_sizes(valid_images_path)

    print(f"Number of training images: {num_train}")
    print(f"Number of validation images: {num_valid}")
    print(f"Training image sizes: {train_sizes}")
    print(f"Validation image sizes: {valid_sizes}")

    assert num_train > 0, "No training images found"
    assert num_valid > 0, "No validation images found"


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
