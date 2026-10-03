"""End-to-end smoke test: proves the YOLOv8 training call runs successfully
on a tiny sample, without needing the real dataset or DVC credentials.
"""

from pathlib import Path

from ultralytics import YOLO

SAMPLE_DATA_YAML = Path(__file__).parent / "fixtures" / "sample_dataset" / "data.yaml"


def test_yolo_training_runs_end_to_end():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(SAMPLE_DATA_YAML),
        epochs=1,
        imgsz=320,
        batch=2,
        device="cpu",
        seed=42,
        project="runs/detect",
        name="ci_smoke_test",
        exist_ok=True,
        verbose=False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    assert best_weights.exists(), "Training did not produce a best.pt weights file"
