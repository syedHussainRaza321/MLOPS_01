"""One-time (or re-runnable) helper: builds a tiny sample dataset for CI,
small enough to commit directly to Git (bypassing DVC), so GitHub Actions
can run a real smoke test without needing dataset credentials.
"""

import shutil
from pathlib import Path

from src.data_utils import DATASET_DIR

SAMPLE_DIR = Path("tests/fixtures/sample_dataset")
NUM_TRAIN = 16
NUM_VALID = 6


def copy_subset(split: str, count: int):
    src_images = DATASET_DIR / split / "images"
    src_labels = DATASET_DIR / split / "labels"
    dst_images = SAMPLE_DIR / split / "images"
    dst_labels = SAMPLE_DIR / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_images.glob("*.jpg"))[:count]
    for img_path in image_files:
        shutil.copy(img_path, dst_images / img_path.name)
        label_path = src_labels / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, dst_labels / label_path.name)

    print(f"Copied {len(image_files)} {split} images (+ matching labels)")


def write_data_yaml():
    content = "train: train/images\nval: valid/images\nnc: 1\nnames: ['Vehicle']\n"
    (SAMPLE_DIR / "data.yaml").write_text(content)


def main():
    copy_subset("train", NUM_TRAIN)
    copy_subset("valid", NUM_VALID)
    write_data_yaml()
    print(f"Sample dataset ready at {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
