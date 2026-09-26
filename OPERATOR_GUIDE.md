# Amazon ML Challenge 2026 — Operator Guide

This guide describes the commands implemented in the repository. Run Python commands from the repository root with the project virtual environment active. Core `python -m ...` commands work on Windows, macOS, and Linux; only the PowerShell runner and current Makefile are Windows-oriented.

## 1. Requirements and first-time setup

- Python **3.11 or newer** is required by the pinned scikit-learn dependency. Python 3.12 is the recommended common choice.
- Git is required to clone/update the repository.
- `requirements.txt` installs the Python packages, including Polars, SciPy/scikit-learn, LightGBM, RapidFuzz, psutil, and Kaggle CLI.
- Raw challenge files are not part of Git (`dataset/` and `info/` are ignored). The private data is `ashash77/amazon-ml-2026-dataset`; the Kaggle account must have access.
- `info/data/student_resource/utils/validate_submission.py` is the configured validator path. That directory is ignored by Git and may need to be provided separately for local official validation.

### Windows PowerShell

If Python 3.12 and Git are not installed, install them first (for example, with the official Python installer and Git for Windows). From the cloned repository:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either allow scripts for this terminal only with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again, or call `\.venv\Scripts\python.exe` explicitly instead of activating.

### macOS Terminal

Install Homebrew if needed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`), then install Git and Python 3.12. Homebrew's versioned Python path avoids relying on which `python3` is first on `PATH`:

```bash
brew install git python@3.12
git --version
"$(brew --prefix python@3.12)/bin/python3.12" --version
git clone https://github.com/Arsh1255/amazon-ml-2026.git
cd amazon-ml-2026
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Homebrew is not installed, use its official installation instructions first. For Linux, install Git and Python 3.11+ using the distribution package manager; the venv, pip, and Python module commands below are the same as macOS.

### Kaggle credentials and raw dataset

Create an API token in Kaggle account settings and save `kaggle.json` at:

- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- macOS/Linux: `~/.kaggle/kaggle.json` (restrict access with `chmod 600 ~/.kaggle/kaggle.json`).

Copy the downloaded token into place:

**Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force "$HOME\.kaggle" | Out-Null
Copy-Item "$HOME\Downloads\kaggle.json" "$HOME\.kaggle\kaggle.json"
```

**macOS/Linux:**

```bash
mkdir -p "$HOME/.kaggle"
cp "$HOME/Downloads/kaggle.json" "$HOME/.kaggle/kaggle.json"
chmod 600 "$HOME/.kaggle/kaggle.json"
```

With the virtual environment active, download the private dataset into the location config detects by default (`<project-root>/dataset`):

**Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force dataset | Out-Null
kaggle datasets download -d ashash77/amazon-ml-2026-dataset --unzip -p dataset
```

**macOS/Linux:**

```bash
mkdir -p dataset
kaggle datasets download -d ashash77/amazon-ml-2026-dataset --unzip -p dataset
```

The expected layout is `dataset/train/train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`, and the corresponding test TSVs under `dataset/test/`. Optional path overrides are `AMAZON_ML_BASE_DIR` and `AMAZON_ML_DATASET_DIR`. Kaggle runs use the attached dataset mount at `/kaggle/input/amazon-ml-2026-dataset`.

## 2. Installation smoke checks

These commands only inspect imports, paths, and CLI options. They do not run blocking or training:

```bash
python -c "import numpy, polars, scipy, sklearn, lightgbm, rapidfuzz, psutil; print('imports OK')"
python -c "from src.config import config; print(config.BASE_DIR); print(config.DATASET_DIR)"
python -m src.run_pipeline --help
python -m src.merge_candidates --help
python scripts/deploy.py --help
```

The `deploy.py --help` command parses options without initializing deployment. Do not use `run_pipeline --dry-run` as an installation check: it can ingest TSVs and can write mock validation TSVs.

## 3. Pipeline commands and flags

All commands use the active virtual environment. In PowerShell activate with the Windows venv activation script; on macOS/Linux run source .venv/bin/activate.

### Main pipeline: `python -m src.run_pipeline`

| Option | Behavior |
|---|---|
| `--stage setup` | Accepted by the parser but currently has no setup handler. Use the venv/pip installation steps above. |
| `--stage block` | Streams training and test TSVs, makes the validation split, blocks validation/train/test, and writes candidate Parquets. Heavy. |
| `--stage block --country US` | Streams train inputs filtered to US, writes US cleaned/validation partitions and US train/validation candidates; keeps full shared ground truth. Does not process test data. Heavy. `--country` works only with this stage. |
| `--stage train` | Builds train/validation features from global candidate Parquets and trains LightGBM. Requires candidate and entity/ground-truth artifacts. Heavy. |
| `--stage infer` | Prepares missing test partitions, test candidates, and test features; loads the model, writes test outputs, then invokes the validator if present. Requires a trained model. |
| `--stage validate` | Runs the configured official validator on existing output TSVs. |
| `--stage all` | Runs full blocking, training, and inference. Heavy; use only when the full local pipeline is intended. |
| `--dry-run` | Implemented, but not a no-op for this entrypoint: block/all still ingest source data, and inference dry-run can create mock TSV outputs. Avoid it for normal result generation. |

Examples:

```bash
python -m src.run_pipeline --stage block --country India
python -m src.run_pipeline --stage block --country US
python -m src.merge_candidates --countries India US --modes train val
python -m src.run_pipeline --stage train
python -m src.run_pipeline --stage infer
python -m src.run_pipeline --stage validate
```

`--country` is case-insensitive for selecting the partition. The selected-country block command still scans the train TSVs and the full ground-truth TSV; ingestion checkpoints are country-aware. Blocking progress entries are separated by mode and country. A running block does not resume at an individual Layer 4 query; rerunning can repeat blocking.

### Country output merge: `python -m src.merge_candidates`

Options:

- `--countries COUNTRY [COUNTRY ...]` is required, e.g. `--countries India US`.
- `--modes train val` selects train and validation; accepted modes are only `train` and `val`. If omitted, both are merged.

The utility reads `artifacts/blocked/{country}/{mode}_candidates.parquet`, checks that the Polars schemas match exactly, and writes `artifacts/blocked/{mode}_candidates.parquet`. Feature engineering already reads those global paths.

For Windows India + Mac US work, copy these Mac outputs to the Windows project before merging:

- `artifacts/blocked/US/train_candidates.parquet`
- `artifacts/blocked/US/val_candidates.parquet`
- `artifacts/cleaned/US/`
- `artifacts/val_split/US/`

The Windows workspace must also contain root-level shared label files: `artifacts/cleaned/train_ground_truth_part_*.parquet` and `artifacts/val_split/val_ground_truth_part_*.parquet`. Keep the India cleaned/validation partitions there as well. Mac and Windows runs produce the same global validation ID split from the same source, seed, and dependency behavior. Do not overwrite Windows shared ground truth with incomplete files.

### Training, results, and validation

Training reads `artifacts/blocked/train_candidates.parquet` and `val_candidates.parquet`, plus all relevant country entity partitions and ground truth. It writes:

- `artifacts/features/train_features.parquet` and `val_features.parquet`
- `artifacts/models/model.pkl`
- `artifacts/models/optimal_threshold.json` (when validation threshold optimization completes)
- progress/manifest entries under `artifacts/`

Inference writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. It runs the official validator when the configured validator file exists; otherwise the validator is skipped with a warning. `--stage validate` runs that validator separately. Inspect generated files and logs before submission; for example:

**PowerShell:**

```powershell
Get-ChildItem output
Get-Content output\matching_results.tsv -TotalCount 5
Get-Content output\candidate_pairs.tsv -TotalCount 5
Get-Content artifacts\progress.json
```

**macOS/Linux:**

```bash
ls -lh output
head -n 5 output/matching_results.tsv
head -n 5 output/candidate_pairs.tsv
python -m json.tool artifacts/progress.json
```

The model and validation artifacts are under `artifacts/models/`; loader/blocking state is recorded in `artifacts/progress.json` and `artifacts/manifest.json`. The Windows `spawn` implementation has not yet been full-benchmarked; do not assume a Windows runtime or peak-RAM result until the operator runs the benchmark.

### Runner shortcuts

| Shortcut | Windows `run.ps1` | Makefile target | Description |
|---|---|---|---|
| Setup dependencies | `.\run.ps1 setup` | `make setup` | Installs `requirements.txt` using the pre-created `.venv`; first create/activate the venv. |
| Clean | `.\run.ps1 clean` | `make clean` | Deletes cleaned, blocked, feature, model, and output folders. Current implementations do not remove every state/validation-split file; it is not a complete reset. |
| Block | `.\run.ps1 block` | `make block` | Runs full block stage (all countries, including test); no country argument is exposed by the shortcut. Use the Python module command for `--country`. |
| Train | `.\run.ps1 train` | `make train` | Runs feature engineering and LightGBM training. |
| Infer | `.\run.ps1 infer` | `make infer` | Runs test inference. |
| Validate | `.\run.ps1 validate` | `make validate` | Invokes the official validator with current output TSVs. |
| Deploy | `.\run.ps1 deploy` | `make deploy` | Runs deployment default (dry-run; does not push). Use explicit Python flags to push/download. |
| Help | `.\run.ps1 help` | `make help` | Lists runner targets. |

The PowerShell runner expects `.venv\Scripts\python.exe`. The checked-in Makefile also hardcodes a Windows venv path and uses Windows cleanup syntax; do not use it on macOS/Linux. Direct `python -m ...` commands work on Linux/macOS with dependencies installed. Kaggle's notebook runtime is Linux; deployment sets its own `/kaggle/...` paths.

## 4. Kaggle deployment, download, and upload

The dataset slug is configured as `amazon-ml-2026-dataset`; deployment combines it with username `ashash77`. `notebooks/kernel-metadata.json` attaches `ashash77/amazon-ml-2026-dataset` and sets the kernel private. The generated notebook mounts the dataset under `/kaggle/input/amazon-ml-2026-dataset`, runs the pipeline, and copies TSV/model artifacts to `/kaggle/working` for Kaggle output collection.

`python scripts/deploy.py` supports mutually exclusive flags:

| Command | Effect |
|---|---|
| `python scripts/deploy.py --dry-run` | Verifies Kaggle authentication and Git status, regenerates the notebook and metadata, but does not push/start the remote run. This does write local notebook/metadata files. |
| `python scripts/deploy.py --push` | Performs those checks, pushes the kernel, and starts the Kaggle run. This uploads the generated notebook/code bundle and metadata; the private dataset is attached by slug, not uploaded from the local `dataset/` folder. |
| `python scripts/deploy.py --download-only` | Retrieves output files from the existing Kaggle kernel run. It requires the remote run to have finished and its outputs to include both required TSVs. TSVs are copied into local `output/`; downloaded files are placed under `artifacts/models/`. |
| `python scripts/deploy.py` | Same behavior as `--dry-run`. |

On Windows, put credentials at `%USERPROFILE%\.kaggle\kaggle.json`; on macOS/Linux use `~/.kaggle/kaggle.json` with mode `600`. The CLI is found in `.venv/Scripts` on Windows, `.venv/bin` on macOS/Linux, or from `PATH`.

After the Kaggle run finishes, download its files with:

```bash
python scripts/deploy.py --download-only
```

Check `output/matching_results.tsv` and `output/candidate_pairs.tsv`, and run local validation if the official validator is available. `scripts/deploy.py --push` uploads/runs a Kaggle kernel; **it does not submit the competition entry**. Submit the required output/package through the competition submission page.

## 5. Package a submission

After inference and validation:

```bash
python scripts/package_submission.py
```

This creates `antigravity_team_submission.zip` in the project root (default team name) and includes whichever expected TSVs exist, source files, `requirements.txt`, and optional contest README/template files if those ignored `info/` files are present. It packages locally; it does not upload.

## 6. Configuration and troubleshooting

- Project root resolves from `src/config.py`; `AMAZON_ML_BASE_DIR` and `AMAZON_ML_DATASET_DIR` override the detected paths. Local default data path is `<project-root>/dataset`; Kaggle default is the mounted dataset path.
- `--country` is supported only with `--stage block`. The command processes that country's train/validation partitions and does not block test data.
- `FileNotFoundError` for `dataset/...tsv`: confirm Kaggle download/unzip produced the expected `dataset/train` and `dataset/test` layout, or set `AMAZON_ML_DATASET_DIR`.
- `ModuleNotFoundError`: activate `.venv` and run `python -m pip install -r requirements.txt` using that interpreter.
- Kaggle authentication/access error: verify the token file path/permissions and that the Kaggle user has permission to access the private dataset and kernel.
- Official validator skipped: provide `info/data/student_resource/utils/validate_submission.py`; that ignored path does not come from a fresh Git clone.
- Merge schema error: the country Parquets were produced by incompatible code/schema versions. Keep both machines on the same Git revision and regenerate their country candidate artifacts consistently.
- Layer 4 performance: Windows `spawn` is now implemented but has not been full-benchmarked. The benchmark script uses all available logical CPUs; verify observed worker count, runtime, and peak RAM on the target laptop.


## Execution Commands

### macOS / Linux
```bash
.venv/bin/python scripts/benchmark_india_val.py
```

### Windows
```powershell
.\.venv\Scripts\python scripts\benchmark_india_val.py
```

**Windows Layer 4:** Uses `spawn` with all available logical CPUs (limited by query count). Sparse matrix data are persisted once to temporary `.npy` files and opened read-only by each worker through memory mapping. Tasks contain only `(start, end)` query ranges. Fuzzy fallback stays in the parent process to avoid duplicating the full target-name list in each worker. macOS/Linux continue using the existing `fork` copy-on-write implementation. Run the India validation benchmark with `python scripts/benchmark_india_val.py` on either operating system; measure RAM/runtime before relying on performance expectations.
