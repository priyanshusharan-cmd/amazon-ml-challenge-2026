# Overnight improvement run: stage 2/3 re-scoring with group (sibling) features

Previous best validation macro F0.5: **0.97826** (BASELINE_BEST, stage-1 LightGBM, p ≥ 0.7)
Selected submission (E10) validation macro F0.5: **0.98658** (+0.00832 absolute, +0.85% relative)
Under test-like density (simulation): baseline 0.97574 → E10 **0.98300** (+0.00726)

The baseline is preserved in `cache/experiments/BASELINE_BEST/` (model, threshold, config snapshot,
`matching_results.tsv`, `candidate_pairs.tsv`).

## What the forensics showed (`f1`–`f4`, reports in `../reports/`)
* 77% of the remaining F0.5 loss was **missed matches**, not false merges. Singleton false merges were only 5%.
* Missed matches: 49k were rejected by the classifier with the candidate present (B), 20k were missed by retrieval (A).
  72% of the A cases have no address.
* **Key signal:** when a record's house numbers differ from its S1's, other true records of the same entity share
  that deviation 63% of the time (for example, all variants dropped the unit number). A distractor's deviation is
  shared only 6% of the time. The stage-1 model scores pairs in isolation and cannot see this.

## Experiments (ledger: `ledger.jsonl`)
| id | change | val F0.5 | Δ vs best | result |
|---|---|---|---|---|
| E00 | baseline reproduced | 0.97826 | — | baseline |
| E02 | stage 2: re-score each record's best pair with stage-1 outputs + group features | 0.98539 | +0.00713 | promoted |
| E03 | ablation: stage 2 without group features | 0.98033 | −0.00507 | rejected (group features give +0.005) |
| E04 | stage 3: group features recomputed from stage-2 probabilities | 0.98571 | +0.00031 | candidate |
| E05 | E02 + fuzzy similarity to confident / non-confident siblings | 0.98608 | +0.00068 | promoted |
| E06 | E05 + stage-3 iteration | 0.98661 | +0.00053 | promoted |
| E07 | cross-fitted stage 2 (2 models), stage 3 on the full pool (5M rows) | 0.98677 | +0.00016 | promoted |
| E08 | decision rules: segment thresholds, first/additional thresholds (cross-fitted over halves) | ≤ +0.00008 | — | rejected (noise) |
| E09 | E07 + expected-F0.5 subset per S1 (p ≥ 0.5) | 0.98695 | +0.00018 | promoted, but **not submitted**, see below |
| E10 | E09 + conflict-cluster guard (stage-1 p < 0.1) | 0.98658 | −0.00037 vs E09 | **selected** |

## Why the submission is E10, not the highest-validation E09
The unguarded E09 accepted 67.4% of test records (baseline 58.8%). The investigation (`f6`–`f9`, `e10_guard.py`) found:
* **Test contains a pattern that is rare in train.** Records form clusters that agree with each other on a house
  number that conflicts with the S1 (a same-name branch at a different number). These are 16.6% of test records versus
  3.1% in train.
* In train, such records with stage-1 p < 0.1 are true matches only 11.5% of the time, and stage 3 accepts 11%.
  On test, stage 3 accepts 33.6% of 1.32M such records. That is an extrapolation driven by the non-confident-sibling
  features taking values unseen in training.
* A density simulation (19% of train S1 entities dropped, giving 5.77 records per S1 like test) confirms that the
  guard costs nothing under test-like conditions (0.98300 vs 0.98280 unguarded). Test acceptance becomes 63.7%, close
  to the ~65% implied by density-free similarity rules.

The unguarded E09 file is kept at `cache/experiments/SUBMISSION_E09/matching_results.tsv` as a higher-risk alternative.

## Reproduce (after the main pipeline has produced train/test candidates, features and stage-1 scores)
```bash
cd experiments
python f1_qtable.py                 # train: top-3 stage-1 candidates per record
python f2_classes.py                # validation error classes (needed by f3/f4 only)
python s2_build.py train            # stage-2 rows + group features
python s2_build.py test
python fz_build.py train            # fuzzy sibling features
python fz_build.py test
python s4_crossfit.py E07_crossfit fz_train.parquet 5000000 127      # stage 2 (x2) + stage 3, validation sweep
python final_predict.py config_E09.json                              # test stage-3 probabilities + E09 file
python final_e10.py                                                  # guard + expected-F -> SUBMISSION_E10
```
All scripts use a memory watchdog (`harness.py`) that aborts if system free memory drops below 1.5 GB.
