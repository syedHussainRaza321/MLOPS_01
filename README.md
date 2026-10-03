# MLOPS Course Assignment 1

Git-based collaboration MLOps Assignment01.

**Team:** Hussain Raza, Amir Ali

## Overview

Built for an MLOps course assignment, this project demonstrates how Git-based version
control extends to a full machine learning pipeline — not just code, but data and
models too. Rather than pushing large datasets or model files directly to GitHub, DVC
tracks them separately, backed by a DagsHub remote, while Git manages the code and
small pointer files. The concrete task — fine-tuning a YOLOv8 model to detect vehicles
in top-view traffic images — demonstrates a collaborative, reproducible ML workflow:
protected branches, reviewed pull requests, automated checks, and a tagged release that
can be independently reproduced from scratch.

## Dataset

**Top-View Vehicle Detection Image Dataset** (Kaggle, farzadnekouei)
https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset

- Single class: `Vehicle` (cars, trucks, buses grouped together)
- YOLO-format labels, pre-split into `train/` and `valid/`
- ~47.8 MB total

Starter code adapted from the companion Kaggle notebook "Real-Time Traffic Density
Estimation with YOLOv8" by the same author.

## Project structure

```
.
├── configs/                   # (reserved; params.yaml lives at root)
├── data/raw/                  # dataset (DVC-tracked, not in Git)
├── models/                    # trained weights (DVC-tracked, not in Git)
├── notebooks/                 # EDA notebook (jupytext-paired .py/.ipynb)
├── scripts/                   # one-off helper scripts (e.g. CI sample builder)
├── src/
│   ├── data_utils.py          # shared dataset constants and validation
│   ├── prepare.py             # DVC "prepare" stage
│   └── modeling/
│       ├── train.py           # DVC "train" stage
│       └── predict.py         # DVC "evaluate" stage
├── tests/                     # unit tests, data checks, CI smoke test
├── .github/workflows/ci.yml   # CI: lint, tests, data checks, smoke train
├── .pre-commit-config.yaml    # ruff, nbstripout, large-file block, secret scan
├── params.yaml                # hyperparameters and seed
├── dvc.yaml / dvc.lock        # pipeline stage definitions
├── metrics.json               # latest evaluation results
├── CONTRIBUTING.md            # branching rules, roles, retrospective
└── REPORT.md                  # assignment submission report
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a [DagsHub](https://dagshub.com) account
with access to this project's DVC remote.

```bash
git clone https://github.com/syedHussainRaza321/MLOPS_01.git
cd MLOPS_01
uv sync

# Configure your personal DVC remote credentials (not committed to Git)
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <your-dagshub-username>
uv run dvc remote modify origin --local password <your-dagshub-token>

uv run dvc pull
uv run pre-commit install
```

## Running the pipeline

```bash
uv run dvc repro
```
Runs `prepare` → `train` → `evaluate` in order, producing `models/best.pt` and `metrics.json`.

Run individual stages directly:
```bash
uv run python -m src.prepare
uv run python -m src.modeling.train
uv run python -m src.modeling.predict
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Current model performance

| Metric | Value |
|---|---|
| Precision | 0.9753 |
| Recall | 0.2949 |
| mAP50 | 0.8132 |
| mAP50-95 | 0.5123 |

See `REPORT.md` for the full experiment comparison and reproducibility details.