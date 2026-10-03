"""Data validation checks, run against the small committed CI sample."""

from pathlib import Path

import yaml

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample_dataset"


def test_data_yaml_exists_and_is_valid():
    data_yaml_path = SAMPLE_DIR / "data.yaml"
    assert data_yaml_path.exists()

    content = yaml.safe_load(data_yaml_path.read_text())
    assert content["nc"] == 1
    assert content["names"] == ["Vehicle"]


def test_every_split_has_images_and_labels():
    for split in ["train", "valid"]:
        images = list((SAMPLE_DIR / split / "images").glob("*.jpg"))
        labels = list((SAMPLE_DIR / split / "labels").glob("*.txt"))
        assert len(images) > 0, f"No images found in {split}"
        assert len(labels) > 0, f"No labels found in {split}"


def test_label_format_is_valid_yolo():
    """Each label line must be: class_id x_center y_center width height,
    all five values present, class_id is 0 (single-class dataset), and the
    four coordinates are normalized between 0 and 1.
    """
    for split in ["train", "valid"]:
        label_files = list((SAMPLE_DIR / split / "labels").glob("*.txt"))
        for label_file in label_files:
            for line in label_file.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5, f"Malformed line in {label_file}: {line}"

                class_id = int(parts[0])
                assert class_id == 0, f"Unexpected class_id in {label_file}: {class_id}"

                coords = [float(p) for p in parts[1:]]
                assert all(
                    0.0 <= c <= 1.0 for c in coords
                ), f"Coordinate out of range in {label_file}: {line}"
