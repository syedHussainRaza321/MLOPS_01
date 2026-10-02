"""Shared dataset path constants and validation utilities."""

import os
from pathlib import Path

import yaml
from PIL import Image

# src/data_utils.py -> parents[0] = src, parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "Vehicle_Detection_Image_Dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
SEED = 42


def count_and_check_sizes(images_path: Path):
    """Count .jpg images in a folder and collect their distinct pixel sizes."""
    sizes = set()
    count = 0
    for filename in os.listdir(images_path):
        if filename.endswith(".jpg"):
            count += 1
            with Image.open(images_path / filename) as img:
                sizes.add(img.size)
    return count, sizes


def check_dataset():
    """Validate the dataset is present and report basic counts/sizes."""
    assert DATA_YAML.exists(), (
        f"data.yaml not found at {DATA_YAML}. "
        "If the dataset folder is missing, run `uv run dvc pull`."
    )

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

    return num_train, num_valid
