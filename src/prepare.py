"""Prepare stage: validate the dataset and record a summary report.

The dataset itself is already pre-split into train/valid by its source
(no splitting needed here), so this stage's job is validation and
producing a tracked report DVC can use to detect when the raw data changes.
"""

import json
from pathlib import Path

from src.data_utils import DATASET_DIR, check_dataset

REPORT_PATH = Path("data/processed/dataset_report.json")


def main():
    num_train, num_valid = check_dataset()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_dir": str(DATASET_DIR),
        "num_train_images": num_train,
        "num_valid_images": num_valid,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote dataset report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
