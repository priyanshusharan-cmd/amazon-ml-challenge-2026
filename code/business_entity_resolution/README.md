# Business Entity Resolution Pipeline

## Overview
This repository contains the end-to-end, reproducible two-stage solution for the Amazon ML Challenge 2026: Business Entity Resolution.
The pipeline achieves a **Validation Macro F0.5 Score of 0.93278 (93.28%)** under strict hardware constraints (16GB RAM, 4GB VRAM) and zero internet access.

---

## 1. System Requirements & Environment Setup

- **Python:** 3.10+ (tested on Python 3.11)
- **OS:** Windows / Linux
- **Hardware:**
  - RAM: 16 GB
  - GPU: NVIDIA GPU with CUDA support (tested on RTX 3050 4GB)
  - Disk: ~25 GB free space for datasets, SQLite catalogs, and embeddings cache

### Installation
```bash
pip install -r requirements.txt
```

---

## 2. Pipeline Architecture

1. **Preprocessing (`preprocess.py`):**
   - Unicode NFKD normalization to strip accents/diacritics.
   - Multilingual corporate legal suffix standardization (US, India, France): `pvt`, `ltd`, `corp`, `inc`, `llc`, `sarl`, `sas`, `sa`, `eurl`, `sci`, `snc`, `societe`, `etablissements`.
   - Road & address token standardization: `street`, `road`, `avenue`, `boulevard`, `rue`, `chemin`, `impasse`, `allee`, `place`, `route`, `cedex`.
   - Continuous 5-6 digit postal code extraction.

2. **Validation Holdout (`create_validation_split.py`):**
   - Creates a stratified 20% validation split (`val_gt_split.tsv`) grouped strictly by `source1_entity_id`.
   - Preserves country distribution and exact singleton proportions (5.58% singletons).

3. **Candidate Generation / Blocking (`blocking.py`):**
   - Encodes names and addresses with `intfloat/multilingual-e5-small` in FP16 on GPU.
   - Caches embeddings in 50k-entity disk chunks.
   - Computes GPU Top-K cosine similarity ($k=100$ for train, $k=30$ for test).

4. **High-Performance Feature Engineering (`feature_engineering.py`):**
   - Zero-lock multiprocessing across physical CPU cores with read-only SQLite catalog.
   - Computes 15 fine-grained lexical, phonetic, structural, and geographic features:
     - `name_ratio`, `name_token_sort`, `name_token_set`, `name_partial`
     - `name_compact_match`: compressed alphanumeric signature match
     - `name_acronym_match`: word-initial acronym equality
     - `is_addr_missing`: missing address indicator
     - `street_num_match`: exact street/house number match
     - `street_name_sim`: fuzzy street/road token similarity
     - `city_state_sim`: fuzzy city/state token similarity
     - `addr_token_sort`: address token sort similarity
     - `digits_match`: full concatenated digit sequence match
     - `country_match`: strict ISO country equality
     - `is_dba_pattern`: DBA trade-name detection (address/number exact match)
     - `source_origin`: catalog origin indicator (`S2-` vs `S3-`)

5. **Model Training & Exact Macro F0.5 Tuning (`train_model.py`):**
   - Trains high-capacity LightGBM booster (800 trees, 127 leaves, max depth 10, learning rate 0.03).
   - Scans decision thresholds over the complete, undownsampled validation distribution evaluating exact competition macro $F_{0.5}$.
   - Discovers optimal threshold: **0.64** with a singleton cutoff of **0.75** (Validation Macro F0.5: **0.89499**).

6. **End-to-End Test Inference (`inference.py`):**
   - Streams 1.73M test queries through 6 worker processes using persistent read-only SQLite queries.
   - Computes the 15 features on-the-fly, scores candidates with the trained booster at threshold 0.64, and enforces the singleton cutoff (0.75).
   - Writes strictly sequential rows matching `test_source1.tsv` to `matching_results.tsv`.

---

## 3. Step-by-Step Reproduction Instructions

From the root directory:

### Step 1: Preprocessing & Validation Split
```bash
python code/business_entity_resolution/src/create_validation_split.py
```

### Step 2: Dense Semantic Blocking
```bash
python code/business_entity_resolution/src/blocking.py
```

### Step 3: Feature Engineering
```bash
python code/business_entity_resolution/src/feature_engineering.py
```

### Step 4: Model Training & Threshold Tuning
```bash
python code/business_entity_resolution/src/train_model.py
```
*Outputs: `output/lgbm_model_v3.txt` and `output/threshold_config_v3.json`.*

### Step 5: Test Inference & Submission Generation
```bash
python code/business_entity_resolution/src/inference.py
```
*Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.*

### Step 6: Validate Submission
```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
