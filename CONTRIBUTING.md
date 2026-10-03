# Contributing

## Project structure
- `src/train.py` — dataset validation + YOLOv8 training (adapted from Kaggle notebook)
- `data/raw/` — raw dataset (DVC-tracked, not in Git)
- `models/` — trained model artifacts (DVC-tracked, not in Git)
- `notebooks/` — EDA and exploratory work
- `configs/`, `tests/`, `.github/workflows/` — pipeline config, tests, CI

## Branching
- main: production, tagged releases only
- staging: release candidate, reproduced and validated
- dev: integration of finished work
- feat/<name>: features/pipeline changes, from dev, into dev
- data/<name>: dataset changes (DVC), from dev, into dev
- exp/<member>-<idea>: personal experiments, from dev, never merged directly
- fix/<name>: urgent production fixes, from main, into main then back into dev

## Commit messages
Use Conventional Commits: feat: ..., data: ..., exp: ..., fix: ..., chore: ...

## Merge strategy
PRs into dev are [squash-merged / rebase-merged] — decide as a team and fill this in.

## Roles
- Data owner: Hussain
- Model owner: Amir
- Platform owner (CI/releases): Hussain
- Platform owner (pre-commit/environment): Amir

## Dataset and credits
- Top-View Vehicle Detection Image Dataset (Kaggle, farzadnekouei), single-class
  ("Vehicle") object detection, YOLO-format labels.
- Starter code adapted from "Real-Time Traffic Density Estimation with YOLOv8"
  (Kaggle notebook, same author).