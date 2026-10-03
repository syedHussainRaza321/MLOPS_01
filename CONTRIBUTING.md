# Contributing

## Project structure
- `src/data_utils.py` — shared dataset path constants and validation utilities
- `src/prepare.py` — data preparation/validation stage (DVC "prepare")
- `src/modeling/train.py` — YOLOv8 training stage (DVC "train")
- `src/modeling/predict.py` — evaluation stage, writes metrics.json (DVC "evaluate")
- `scripts/make_ci_sample.py` — builds the small committed dataset sample used by CI
- `notebooks/01-eda.py` / `01-eda.ipynb` — exploratory data analysis (jupytext-paired)
- `tests/` — unit tests, data validation checks, and the CI smoke-training test
- `data/raw/Vehicle_Detection_Image_Dataset/` — raw dataset (DVC-tracked, not in Git)
- `models/` — trained model artifacts (DVC-tracked, not in Git)
- `params.yaml` — all hyperparameters and the random seed
- `dvc.yaml` / `dvc.lock` — pipeline stage definitions and execution record
- `metrics.json` — latest evaluation results (Git-tracked, not DVC-cached)

## Branching
- `main`: production, tagged releases only
- `staging`: release candidate, reproduced and validated
- `dev`: integration of finished work
- `feat/<name>`: features/pipeline changes, from `dev`, into `dev`
- `data/<name>`: dataset changes (DVC), from `dev`, into `dev`
- `exp/<member>-<idea>`: personal experiments, from `dev`, never merged directly
- `fix/<name>`: urgent production fixes, from `main`, into `main` then back into `dev`

## Commit messages
Conventional Commits: `feat:`, `data:`, `exp:`, `fix:`, `chore:`, `docs:`, `test:`, `ci:`, `style:`

## Merge strategy
PRs into dev are squash-merged — each pull request becomes a single, clean commit
on dev, regardless of how many individual commits were made on the feature branch.
This keeps dev's history easy to read, with one commit per logical change (per PR),
rather than every intermediate "fix typo" or "try again" commit cluttering the log.

## Roles
- Data owner: Hussain Raza
- Model owner: Amir Ali
- Platform owner (CI/releases): Hussain Raza
- Platform owner (pre-commit/environment): Amir Ali

## Dataset and credits
- **Dataset:** Top-View Vehicle Detection Image Dataset (Kaggle, farzadnekouei), single-class
  (`Vehicle`) object detection, YOLO-format labels, pre-split train/valid, ~47.8 MB.
  Source: https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset
- **Starter code credit:** adapted from the Kaggle notebook "Real-Time Traffic Density
  Estimation with YOLOv8" by the same author — training call structure and dataset
  validation checks were sourced from this notebook; visualization/video-processing
  and traffic-density logic were intentionally left out of the core pipeline.

## Environment
- Dependency management: `uv` (not `pip`/`conda`)
- Python 3.11
- Run project scripts as modules, not file paths: `uv run python -m src.modeling.train`
  (running by file path breaks `from src...` imports)

## Local setup for a new clone
```bash
uv sync
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <your-dagshub-username>
uv run dvc remote modify origin --local password <your-dagshub-token>
uv run dvc pull
uv run pre-commit install
```

## Retrospective

**What broke during this project:**
- Early confusion between `uv init`'s own project template and the instructor-required
  Cookiecutter Data Science scaffold — ultimately resolved by dropping cookiecutter and
  building the layout manually, after already diverging from it.
- Running pipeline scripts by file path (`python src/modeling/train.py`) instead of as a
  module (`python -m src.modeling.train`) — caused repeated `ModuleNotFoundError: No
  module named 'src'` errors across multiple phases, since file-path execution doesn't
  add the project root to Python's import search path.
- Ultralytics' default `project=` folder handling silently doubled the output path
  (`runs/detect/runs/detect/...`), breaking our assumed path to the trained weights file.
- `optimizer: auto` in `params.yaml` silently ignored our explicit `lr0` setting,
  invalidating an early round of learning-rate experiments until the correct `AdamW`
  optimizer was set explicitly.
- `detect-secrets` repeatedly failed to read `.secrets.baseline` due to PowerShell's `>`
  redirect writing a UTF-8 BOM that the tool couldn't parse — fixed by writing the file
  through Python instead.
- `ruff` was never added as an explicit project dependency in `pyproject.toml` (it only
  ran locally via pre-commit's isolated hook environment), which caused CI to fail with
  "Failed to spawn: ruff" until added via `uv add --dev ruff`.

**What we'd standardize next time:**
- Always run project scripts as modules (`-m`), never by file path.
- Explicitly add every CLI tool used anywhere in the project (including ones pre-commit
  manages) as a real `uv` dependency, so CI and local environments can't drift apart.
- Confirm tool versions (e.g. `ruff`) match between `pyproject.toml` and
  `.pre-commit-config.yaml` to avoid local/CI formatting disagreements.
- Decide on cookiecutter usage (or not) before any folder scaffolding begins, not mid-project.

**What we added to CONTRIBUTING.md because of it:**
- The "Environment" and "Local setup for a new clone" sections above, specifically calling
  out the module-execution requirement.