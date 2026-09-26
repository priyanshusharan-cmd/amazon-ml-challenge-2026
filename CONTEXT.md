# Project Context

## Goal and current status

Amazon ML Challenge 2026 business-entity resolution. The pipeline reads train/test TSVs, partitions entities by country, creates candidate matches, builds pair features, trains a LightGBM classifier, and writes a submission.

Layer 4 in `src/blocking.py` now retrieves normalized character 3-gram TF-IDF matches through a CSC inverted index. It retains the configured threshold (`0.35`) and `TOP_K` (`20`). **No post-rewrite full validation benchmark has been run**, so recall, runtime, and peak-memory improvement are unverified. Do not treat pre-rewrite measurements as results for this implementation.

## Source map

- `src/config.py`: `pathlib` project/data roots, environment overrides, paths, and model/blocking settings.
- `src/data_loader.py`: TSV ingestion, dynamic country partitioning, deterministic validation split, and resumable chunk checkpoints.
- `src/normalizer.py`: name/address cleaning.
- `src/blocking.py`: country-local candidate generation; Layers 2–5; Layer 4 is the TF-IDF CSC posting index.
- `src/merge_candidates.py`: schema-check and merge per-country train/validation candidate Parquets into global inputs.
- `src/feature_engineering.py`: reads global candidate Parquets plus cleaned entities and ground truth; writes feature Parquets.
- `src/train.py`, `src/evaluate.py`: LightGBM training and validation threshold selection.
- `src/inference.py`: test candidate scoring, TSV writing, and optional official validator invocation.
- `scripts/deploy.py`: Kaggle kernel metadata generation, push, dry run, and output download.
- `scripts/package_submission.py`: packages existing TSV outputs and code into a ZIP; it does not upload to the competition.

## Paths and data

`BASE_DIR` resolves from the source location on Windows/macOS/Linux. `DATASET_DIR` defaults to `<project-root>/dataset`; set `AMAZON_ML_BASE_DIR` or `AMAZON_ML_DATASET_DIR` to override. On Kaggle the deployment notebook sets the project under `/kaggle/working` and uses `/kaggle/input/amazon-ml-2026-dataset`.

The private Kaggle dataset slug is `ashash77/amazon-ml-2026-dataset`. `dataset/` and `info/` are git-ignored. A fresh clone therefore needs the private raw dataset locally; the optional validator/template files under `info/data/student_resource/` may also need to be supplied separately. The validator path expected by config is `info/data/student_resource/utils/validate_submission.py`.

## Country-partition workflow

From the repository root with dependencies installed and raw TSVs in `dataset/`:

```bash
python -m src.run_pipeline --stage block --country India
python -m src.run_pipeline --stage block --country US
python -m src.merge_candidates --countries India US --modes train val
python -m src.run_pipeline --stage train
python -m src.run_pipeline --stage infer
python -m src.run_pipeline --stage validate
```

Country blocking streams training records for the selected country, writes `artifacts/cleaned/{country}/` and `artifacts/val_split/{country}/`, and generates `artifacts/blocked/{country}/{train,val}_candidates.parquet`. It does not ingest or block test data. Ground truth is kept as shared root-level Parquets. Before training on the Windows machine, copy back the US blocked candidates and US cleaned/validation directories; retain shared ground-truth files there. The merge utility writes `artifacts/blocked/{train,val}_candidates.parquet`, which are the paths feature engineering expects.

Full `--stage block` ingests all train and test sources, then blocks validation, train, and test across discovered countries. `--stage train` expects global train/validation candidate Parquets. `--stage infer` prepares missing test partitions/candidates/features, predicts, writes output TSVs, and invokes the configured validator if present.

## CLI caveats

- `--stage` accepts `setup`, `block`, `train`, `infer`, `validate`, and `all`. `setup` currently has no setup implementation; install dependencies with `pip install -r requirements.txt` instead.
- `--country` is valid only with `--stage block`.
- Pipeline `--dry-run` is not a safe no-op: block/all still ingest TSVs, and dry-run inference can write mock TSVs. Do not use it to verify a real submission.
- `scripts/deploy.py` accepts mutually exclusive `--dry-run`, `--push`, and `--download-only`; omitting a flag defaults to deployment dry run.
- `run.ps1` is the Windows shortcut. macOS/Linux should use `python -m ...` commands from `OPERATOR_GUIDE.md`; the checked-in Makefile also hardcodes a Windows virtualenv path and Windows cleanup syntax.

## Expected outputs

- Candidates: `artifacts/blocked/{mode}_candidates.parquet`; country-specific inputs live under `artifacts/blocked/{country}/` before merge.
- Features: `artifacts/features/{train,val,test}_features.parquet`.
- Model/threshold: `artifacts/models/model.pkl` and `artifacts/models/optimal_threshold.json`.
- Submission TSVs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- State: `artifacts/progress.json` and `artifacts/manifest.json`.
- Package: `<team-name>_submission.zip` in the project root when `scripts/package_submission.py` is run.

Kaggle `--push` uploads the generated notebook/kernel and starts its run; it is not a competition submission upload. `--download-only` downloads a completed kernel's output artifacts. Competition submission is a separate action through the competition's submission page.
