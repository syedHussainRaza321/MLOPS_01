# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: vehicle-detection-ml-collab
#     language: python
#     name: vehicle-detection-ml-collab
# ---

# %% [markdown]
# # EDA: Top-View Vehicle Detection Dataset
# Exploring the dataset before training YOLOv8.

# %%
import random
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import cv2
import matplotlib.pyplot as plt

from src.data_utils import DATASET_DIR, check_dataset

# %% [markdown]
# ## Validate the dataset
# Uses the same `check_dataset` function as the training pipeline, so EDA
# and training never disagree about the dataset's structure.

# %%
check_dataset()

# %% [markdown]
# ## Visualize a few training images with their bounding boxes
# Converts YOLO's normalized box format (center x, center y, width,
# height, all as fractions of image size) into pixel coordinates, then
# draws them on the image.


# %%
def yolo_to_pixel_box(x_center, y_center, w, h, img_w, img_h):
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes(image_path, label_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                _cls, xc, yc, w, h = map(float, line.split())
                x1, y1, x2, y2 = yolo_to_pixel_box(xc, yc, w, h, img_w, img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


# %%
train_images_dir = DATASET_DIR / "train" / "images"
train_labels_dir = DATASET_DIR / "train" / "labels"

sample_files = random.sample(sorted(train_images_dir.glob("*.jpg")), 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, image_path in zip(axes.ravel(), sample_files):
    label_path = train_labels_dir / (image_path.stem + ".txt")
    img = draw_boxes(image_path, label_path)
    ax.imshow(img)
    ax.set_title(image_path.name)
    ax.axis("off")

plt.tight_layout()
plt.show()
