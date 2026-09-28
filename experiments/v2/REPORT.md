# v2 final report — direct-evidence matcher

**Outcome: challenger PROMOTED** (all predeclared gates passed; see `PROGRESS.md` for the gate text written before the stress results were seen).

## Headline numbers (exact metric: per-S1 `5TP/(5TP+4FP+FN)`, empty-set convention, all S1 averaged)
| | baseline recipe (B0, refit on TRAIN) | final (C4) |
|---|---|---|
| DEV (development, 441k S1) | 0.97785 | **0.98322** |
| CONF (sealed, opened once, 441k S1) | 0.97787 | **0.98321** |
| paired Δ on CONF (family bootstrap, 1000×) | | **+0.00533 [95% CI +0.00514, +0.00554]** |
| stress suite, DEV-stress overall | 0.97782 | **0.98324** |
| stress, twins receiving clusters | 0.97261 | **0.97744** |
| stress, zero-match twins | 0.9662 | **0.9724** |
| test acceptance (records linked) | 59.1% (original leaderboard baseline 58.8%) | 57.8% |

For reference, the original baseline file (trained on an 80% fold) scored ~0.97x on the leaderboard. The sibling-feature run E10 scored 0.933.

## What improved and why
1. **Removed the population feature** `s1_rank1_deg` (records retrieving an S1 at rank 1). It depends on other records, and duplicating a rejected wrong-branch record flipped 3/20,000 baseline decisions. Removing it is DEV-neutral (0.97791) but costs zero-match S1 entities (0.980 → 0.962 at t=0.7): the known, bounded trade-off.
2. **Direct-evidence features** (`featx.py`, +0.0027): IDF-weighted name/address token agreement with capped rarity, the rarest one-sided token, premise vs unit numbers (equality, containment, closeness), raw-text agreement.
3. **More training data** (+0.0015): 25% of TRAIN records (15.5M pairs) instead of 10%.
4. **Per-record second stage** (+0.0012): re-scores each record's best candidate with its own top-3 probabilities and margin. The inputs are out-of-sample and never use other records.

## Retrieval ceiling and remaining error budget
* Candidate-restricted oracle macro F0.5 (DEV): **0.99609**. Retrieval is not the bottleneck for 0.988.
* Final DEV 0.98322, so 0.0129 of macro score is still recoverable within current candidates. Per the v1 analysis this is mostly
  threshold-rejected true matches (often no-address records with ambiguous names) and near-copy distractors.
* Target 0.988 was **not reached**: the best defensible model is 0.9832 on the sealed fold.

## Stress suite and invariance
* Built from real labels. In each of 31,050 same-name DEV groups (≥2 addresses) one business was removed from the catalog. Its 107k real records became coherent same-name, wrong-address clusters retrieving the remaining 121,347 twins.
* Distractor acceptance is flat across cluster sizes 1–8 (~1.6–2.0%) for all pairwise models. The v1 sibling model showed the opposite.
* Prevalence ×1/×2/×4/×8 (exact for C3/C4): C4 0.98324 / 0.98298 / 0.98266 / 0.98235; B0 (approx.) 0.97782 … 0.97693.
* Duplication invariance: C1–C4 contain no feature depending on other records (verified from feature lists; C1–C3 also empirically 0 flips at k=1..8).

## Data used
* **Development:** TRAIN (S1 id%5 ∈ {2,3,4}) for fitting; DEV (id%5==0) for all selection, thresholds and stress. DEV had also been used by v1 work.
* **Confirmation:** CONF (id%5==1), opened once after the recipe and threshold were frozen (`cache/runs/CONF_OPENED.json`).
* **Independent test:** none. No authorized independent labeled dataset exists. France and the private leaderboard are **unmeasured**.
* Labels were never used by normalization rules; the Indic dictionary was learned on TRAIN only; IDF statistics come from the unlabeled S1 catalog of each split.

## Robustness engineering
* Run manifests (config, code fingerprint, per-output sha256), atomic writes, process locks, and a memory watchdog that aborts a stage instead of exhausting RAM (it fired several times during development; each aborted stage was resumed).
* **Recovery test:** featx on the test split was killed after 4/53 files and resumed from file 5. The pre-kill and post-resume outputs are byte-identical to a clean recomputation.
* **Reproducibility test:** re-running the packaged `business_entity_resolution/src/v2/run_v2.py` skipped completed stages and regenerated a byte-identical `matching_results.tsv` (sha256 5a7f2c06…).
* **Not verified:** a from-scratch rerun of the whole pipeline (~6 h).
* Resources: 16 GB RAM (≈8–11 GB usable), RTX 4050 6 GB, ~70 GB free disk. Heavy stages were run strictly one at a time.

## Artifacts
* Final bundle: `cache/runs/submission_C4_stack_on_C3_direct_x_more_data/` (`matching_results.tsv` sha 5a7f2c06…, `candidate_pairs.tsv` sha 168dd4fd…, validator PASS), copied to `output/`.
* Baseline kept: `cache/experiments/BASELINE_BEST/` (matching sha d6fbfeed…).
* Models: `cache/runs/model_C3_direct_x_more_data/model.txt`, `cache/runs/model_C4_stack_on_C3_direct_x_more_data/model.txt`.
* Evaluated models were trained on TRAIN only. No production refit on more data was done, so the submitted model is exactly the evaluated one.
