# REPORT.md — MLOPS Collaboration Assignment 01

## Team

| Member | Role | GitHub |
|---|---|---|
| Hussain Raza | Data owner; Platform owner (CI, releases) | https://github.com/syedHussainRaza321 |
| Amir Ali | Model owner; Platform owner (pre-commit, environment) | https://github.com/amir-ali-asif |

## Dataset and starter code

- **Dataset:** Top-View Vehicle Detection Image Dataset (Kaggle, farzadnekouei)
  https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset
- **Starter code source:** adapted from the Kaggle notebook "Real-Time Traffic Density
  Estimation with YOLOv8" by the same author.
- **Task:** Object detection — fine-tuning YOLOv8 to detect vehicles (single class)
  in top-view traffic images.
- **Dataset structure:** pre-split `train/`/`valid/` folders with YOLO-format labels,
  single class `Vehicle`, `data.yaml` included.

## Reproducibility table (released model)

| Field | Value |
|---|---|
| Commit SHA (on `main`) | 7955b3fb7b40e3954c688f4d172afe7d1fc4ac97 |
| Tag | `model-v1.0` |
| Lock file | `dvc.lock`, as committed at the tagged commit on `main` (see `git log -1 -- dvc.lock`) |
| `params.yaml` — seed | 42 |
| `params.yaml` — model.epochs | 1 |
| `params.yaml` — model.imgsz | 640 |
| `params.yaml` — model.batch | 8 |
| `params.yaml` — model.optimizer | AdamW |
| `params.yaml` — model.lr0 | 0.001 |
| `params.yaml` — model.lrf | 0.1 |
| `params.yaml` — model.dropout | 0.1 |
| `params.yaml` — model.patience | 50 |
| Data `.dvc` hash | 87898008b45d846360b9d07c87dd3e12.dir |
| Final precision | 0.9753 |
| Final recall | 0.2949 |
| Final mAP50 | 0.8132 |
| Final mAP50-95 | 0.5123 |

## Experiment comparison

### Amir's learning-rate sweep (`exp/amir-learning-rate`)

| Run | lr0 | optimizer | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| amir-lr-0001-v2 | 0.0001 | AdamW | 0.9129 | 0.302 | 0.7789 | 0.5065 |
| **amir-lr-001-v2 (winner)** | **0.001** | **AdamW** | **0.9753** | 0.2949 | **0.8132** | **0.5123** |
| amir-lr-01-v2 | 0.01 | AdamW | 0.0012 | 0.0352 | 0.0002 | 0.0 |

**Winner:** `amir-lr-001-v2` (`lr0=0.001`, `optimizer=AdamW`)
**Why it won:** Highest precision, mAP50, and mAP50-95 of all runs tested. `lr0=0.01`
caused training divergence (model failed to learn anything meaningful); `lr0=0.0001`
was a close second but underperformed on every metric.

**Important finding during this sweep:** an earlier round of experiments showed
identical metrics across all three learning-rate values, which turned out to be
because `optimizer: auto` was silently overriding the explicit `lr0` setting
(Ultralytics logged `"optimizer=auto" found, ignoring 'lr0=...'`). Fixed by setting
`optimizer: AdamW` explicitly before re-running.

### Hussain's dropout sweep (`exp/hussein-dropout`) — abandoned

| Run | dropout | lr0 | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| hussein-dropout-0 | 0.0 | 0.01 | 0.0012 | 0.0352 | 0.0002 | 0.0 |
| hussein-dropout-01 | 0.1 | 0.01 | 0.0012 | 0.0352 | 0.0002 | 0.0 |
| hussein-dropout-03 | 0.3 | 0.01 | 0.0012 | 0.0352 | 0.0002 | 0.0 |

**Abandoned experiment:** `exp/hussein-dropout` — never merged.
**Why it was abandoned:** This branch used `lr0=0.01`, the same value Amir's parallel
sweep identified as causing training divergence. All three dropout values produced
identical, near-zero metrics because the model never successfully learned at that
learning rate — dropout's actual effect couldn't be assessed under these conditions.
Rather than re-run the sweep at the corrected `lr0=0.001` (since a winning
configuration was already found via the learning-rate experiments), this branch is
kept unmerged as a record of the finding.

## Key pull requests

| Type | Link |
|---|---|
| Learning-rate promotion PR | https://github.com/syedHussainRaza321/MLOPS_01/pull/8 |
| GitHub Actions Workflow PR | https://github.com/syedHussainRaza321/MLOPS_01/pull/9 |
| PR with a "changes requested" review | https://github.com/syedHussainRaza321/MLOPS_01/pull/8 |
| Reproducible DVC Pipeline PR | https://github.com/syedHussainRaza321/MLOPS_01/pull/6 |
| Release PR (dev → staging) | https://github.com/syedHussainRaza321/MLOPS_01/pull/10 |
| Release PR (staging → main) | https://github.com/syedHussainRaza321/MLOPS_01/pull/11 |
| Abandoned experiment branch | `exp/hussein-dropout` https://github.com/syedHussainRaza321/MLOPS_01/tree/exp/hussein-dropout |

## Data quality checks

- **Duplicate image check:** scanned all train/valid images by content hash (MD5).
  [State your actual result: "Found zero duplicates — dataset was already clean"
  or list what was removed if duplicates were found.]
- **Label format validation:** CI checks confirm every label file has exactly 5
  values per line, class ID is always 0 (single-class dataset), and all coordinates
  fall within the valid 0–1 normalized range.

## Screenshots


1. **Blocked large file** (Phase 3)
   ![Blocked large file](docs/screenshots/blocked-large-files.jpeg)
2. **Failing CI check** (Phase 8)

   ![Failing CI](docs/screenshots/ci-failure.png)

3. **Amir's Experiments Summary** (Phase 8)

   ![Amir's experiments summary](docs/screenshots/amir's-experiments.png)

4. **Hussain's Experiments Summary** (Phase 8)

   ![Hussain's experiments summary](docs/screenshots/hussain-experiments.png)

## Retrospective

**What broke:**
- Confusion over cookiecutter vs. manual scaffolding early in Phase 2, resulting in
  stray `uv init`-generated files (`src/mlops_01/`) that had to be cleaned up.
- Repeated `ModuleNotFoundError: No module named 'src'` from running scripts by file
  path instead of as Python modules — hit in Phase 6 (`dvc.yaml` stage commands) and
  again in Phase 8 (the CI sample-builder script).
- Ultralytics doubling its own output path (`runs/detect/runs/detect/...`), breaking
  our assumption about where trained weights would be saved.
- `optimizer: auto` silently overriding our explicit `lr0` parameter, invalidating an
  early round of learning-rate experiments.
- A PowerShell UTF-8 BOM encoding issue broke `detect-secrets`'s ability to read its
  own baseline file.
- `ruff` was missing from `pyproject.toml`'s real dependencies (only available locally
  via pre-commit's isolated hook environment), causing CI to fail until explicitly added.

**What we'd standardize next time:**
- Decide on project scaffolding approach (cookiecutter or manual) before writing any
  code, and get instructor confirmation up front.
- Always invoke project scripts as modules (`python -m ...`), never by file path.
- Treat every tool used anywhere in the project (including ones pre-commit manages
  internally) as a real project dependency declared in `pyproject.toml`.
- Verify review/reproducibility checks happen on open PRs *before* merging, not after
  (caught ourselves merging the DVC dataset PR before Amir's reproducibility check —
  recovered by running the check against `dev` directly afterward).

**What we added to `CONTRIBUTING.md` because of it:**
See the "Retrospective" and "Environment"/"Local setup" sections in `CONTRIBUTING.md`.

## Individual contributions

### Hussain Raza
Repository and branch setup; DagsHub remote configuration and initial DVC dataset
tracking; adapted the Kaggle notebook's training logic into the initial pipeline
code; branch protection and CI workflow setup (lint, tests, data checks, smoke
training, plus required-status-check configuration); release process — reproducibility
verification on a fresh clone, `staging`→`main` review, and tagging `model-v1.0`.

### Amir Ali
Pre-commit hook configuration (ruff, nbstripout, large-file blocking, secret
scanning); EDA notebook and extraction of shared dataset validation logic into
`src/data_utils.py` with unit tests; the reproducible DVC pipeline (`prepare`/
`train`/`evaluate` stages, `params.yaml`, `dvc.yaml`); learning-rate experiment
sweep and promotion of the winning configuration; data-update PR (duplicate check);
conflict resolution during the deliberate `params.yaml` conflict exercise.