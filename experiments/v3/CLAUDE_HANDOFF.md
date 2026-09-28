# Claude handoff: improve leaderboard F0.5 toward 0.98

Prepared 26 September 2026. Repository: `/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026`.

## Objective and honest current status

The user confirmed **0.970441 is the latest v2/C4 leaderboard score**. The target is roughly 0.98 without crashing or overfitting. A new v3 candidate improves local validation and passes the declared stress checks. **It has not been submitted or scored on the leaderboard.** Its measured local gain is about 0.001, whereas the desired leaderboard gain is about 0.00956. Do not assume the target is achieved or extrapolate the local gain into a leaderboard forecast.

The model is trained and frozen. Test features are now complete. Remaining work is scoring test records, exporting/validating the files, and obtaining an actual portal score. No portal URL or access was provided.

## Work completed

1. Verified the existing output is the submitted C4 baseline and reproduced its local DEV score exactly. Kept the baseline output and original submission ZIP intact.
2. Added bounded, resumable caches for C3's best candidate per query. Base caches for train, test, and wrong-branch stress are complete.
3. Tested a 51-feature direct-only alternative. Its best standalone DEV was 0.973843 and best declared 50/50 C4 blend was 0.982283. Both failed the predeclared gates and were rejected. Do not export this model.
4. Implemented 25 additional comparison features covering premise/unit/floor roles, house-number suffix conflicts, alphabetic address agreement, fuzzy unmatched tokens, and raw name/address agreement. Unknown roles remain unknown.
5. Trained a regularized LightGBM role specialist with 108 total features: 1,114,229 fit records, 126,566 internal early-stop records, 2,195 selected trees, 127 leaves, minimum leaf size 250, L2=10, four CPU threads.
6. Excluded C3 fit/early-stop queries and DEV/CONF-owned candidates from specialist fitting. Grouped internal early stopping by country and normalized name.
7. Completed full train predictions (10,320,219 rows), selection, reused confirmation, paired family bootstrap, and wrong-branch stress predictions/evaluation.
8. Added atomic checkpoints, input/model/source hashes, bounded text gathering, memory checks, and score alignment checks. Twenty-two focused tests passed (16 features, five evaluation checks, one text-gather alignment check). Baseline metric reproduction and independent CONF count reproduction also passed.

## Frozen results and recipe

| Check | C4 baseline | v3 candidate |
|---|---:|---:|
| Local DEV | 0.983219 | **0.984226** |
| Reused CONF | 0.983206 | **0.984174** |
| DEV with unmatched FP weight x4 | 0.979618 | **0.982036** |
| Wrong-branch stress, overall | 0.983242 | **0.984302** |
| Wrong-branch stress, receiving twins | 0.977444 | **0.978384** |
| DEV entities with no true matches | 0.975234 | **0.985132** |

Both India and US improved. All predeclared gates passed. Reused-CONF gain was +0.000967735, with paired family-bootstrap interval [+0.000860940, +0.001079145] over 1,000 draws. This interval reflects sampling variation on that dataset, not protection against validation reuse or the leaderboard distribution gap.

Frozen mode: **roles**, **threshold 0.80**, **baseline_weight 0**. The specialist runs only for C3 p1 in [0.001, 0.999]; outside that interval its prediction file contains the existing C4 probability. The final 0.80 threshold applies throughout. Thus baseline_weight=0 does not remove that fallback. C3 retrieval and best-candidate identity are unchanged.

Threshold 0.80 was selected by the declared mean over negative multipliers 1/2/4, before reused CONF and wrong-branch results. Do not change it to 0.75 because ordinary DEV is slightly higher there. Do not rerun model selection or refit this model after seeing confirmation.

## Process state at handoff

Test feature enrichment **finished during handoff preparation**: **219/219 batches**, **3,387,279 selected rows**, manifest `complete: true`. Final process memory was 0.85 GiB with 7.67 GiB free RAM. Log: `experiments/v3/enrich_test.log`. A final process inventory showed **no running Python processes**. No test scoring or export process was launched during handoff preparation.

**Start at test scoring (step 2 below).** These commands can recheck state before launch; do not start duplicate heavy jobs if another agent has since resumed work.

```powershell
Set-Location '/Users/priyanshusharan/Documents/project/amazon-ml-challenge-2026'
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Select-Object ProcessId,CommandLine
Get-Content experiments/v3/enrich_test.log -Tail 5
Get-Content cache/runs/v3/role_gate_v2test/manifest.json -Raw | ConvertFrom-Json | Select-Object complete,rows
```

If the process has exited with an incomplete manifest, inspect its log, then resume the same command. Checkpoints are reusable. If an interruption left a stale lock, inspect the lock implementation and verify no owner process remains before removing only that lock.

```powershell
.venv/Scripts/python.exe -u -X utf8 experiments/v3/enrich_rows.py v2test
```

## Exact next steps

Run stages sequentially from the repository root. Keep four compute threads and at least 2 GiB free RAM. Do not edit the frozen training/feature scripts or bypass fingerprint failures to force reuse.

1. Wait for test role feature manifest `complete: true`.
2. Score the frozen model on all **9,969,589** test queries:

```powershell
.venv/Scripts/python.exe -u -X utf8 experiments/v3/train.py roles --score v2test > experiments/v3/score_test.log 2>&1
```

Expected output: `cache/runs/v3/roles/pred_v2test.parquet`. Per-file resumable checkpoints: `score_parts_v2test/`. Require successful exit and the full expected row count.

3. Export and validate a separate candidate:

```powershell
.venv/Scripts/python.exe -u -X utf8 experiments/v3/export.py roles > experiments/v3/export.log 2>&1
```

Expected directory: `output/v3_roles/`, containing `matching_results.tsv`, `candidate_pairs.tsv`, `bundle.json`, and `validation.log`. Export has not yet been run end to end. Fix any actual export error, preserving frozen model/feature behavior, then rerun.

Exporter checks model and validation fingerprints, positive reused CONF gain, passing wrong-branch checks, prediction alignment and valid probabilities. It writes results by S1 buckets, checks all matching rows against the existing candidate file one row at a time, and runs the official matching-format validator. Expected S1 row count is **1,732,544**. Candidate file is a hard link to the unchanged original candidate TSV (copy fallback); do not modify it in place.

The official validator defaults to `output/candidate_pairs.tsv` when `--candidate` is omitted. This could load about 100M candidate strings. The exporter was corrected to explicitly pass a nonexistent placeholder candidate path because candidate subset/row coverage was already checked by streaming. A skipped-candidate warning in the official log is therefore expected. Do not run the candidate-inclusive validator on this machine. ID-existence loading is also off; exported IDs come directly from original test-ID caches.

4. Inspect `bundle.json` and `validation.log`: both validation statuses must be PASS, row counts correct, output hashes present, leaderboard score still null. Verify the baseline has not changed:

```powershell
Get-FileHash output/matching_results.tsv -Algorithm SHA256
```

Expected original baseline SHA256: `5a7f2c062a2e109fa0c8837e0bb0680a7dc6ccfb972636d994378ccd754c237e`.

Original candidate SHA256: `168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32`.

5. Update `REPORT.md`, `PROGRESS.md`, and README with final artifact paths, validation results and counts. Give the user the new matching TSV to score on the portal. Do not send the old `output/matching_results.tsv` by mistake.
6. If a final competition ZIP is required, create a separate v3 archive and update its methodology/code contents. **Do not blindly run the existing `make_submission.py`: it packages the old root output and v2 documentation, and does not include these v3 scripts.** Preserve `team_submission.zip`. The v3 experiments depend on existing C3/C4 caches; a fully self-contained from-scratch v3 package still needs reproducibility work.
7. Obtain the real leaderboard result. If still below approximately 0.98, record that outcome honestly. Further improvement is a new experiment: establish a stronger validation strategy before further tuning, investigate unmatched-record prevalence, family overlap, country transfer and retrieval errors. Current changes cannot fix cases where C3's best candidate is wrong. Avoid repeatedly tuning to reused CONF or leaderboard feedback. No specific next model is proven to close the gap.

## Where to find evidence and implementation

- `experiments/v3/PLAN.md`: predeclared selection and stress gates.
- `experiments/v3/REPORT.md`: fuller methodology, results and caveats.
- `cache/runs/v3/roles/fit.json`, `model.txt`, `source_snapshot/`: frozen training and provenance.
- Same directory: `frozen_selection.json`, `comparison.json`, `confirmation_reused.json`, `confidence_reused.json`, `wrong_branch.json`.
- `experiments/v3/build_rows.py`, `enrich_rows.py`, `direct_features.py`, `train.py`: feature and model pipeline.
- `evaluation.py`, `compare.py`, `wrong_branch.py`, `confidence.py`: evaluation and checks.
- `export.py`: pending export; last validator-default correction passed Python syntax compilation.
- `cache/runs/v3/rows_v2train`, `rows_v2test`, `rows_v2stress`: complete base caches.
- `cache/runs/v3/role_gate_v2train`, `role_gate_v2stress`, `role_gate_v2test`: all complete role caches.
- `cache/runs/v3/roles_v2train` is an early ungated smoke cache, not the actual role training cache.

## Limits and preservation rules

CONF was opened during v2; it is **reused confirmation**, not a fresh holdout. Outer splitting is entity-based, not fully family-disjoint across the pipeline. France occurs only in test and has no measured local accuracy. Stress tests are useful diagnostics, not proof against every distribution shift. No guarantee of zero overfitting is justified.

No observed v3 training/enrichment/scoring job failed from memory exhaustion. Final test scoring/export are still pending and must be checked. Keep heavy processes sequential, use checkpoints, and stop safely on low RAM.

At the initial handoff no commits had been made. The user subsequently requested pushing all changes to `https://github.com/priyanshusharan-cmd/amazon-ml-challenge-2026`; code, tests, reports, this handoff, and the pre-existing untracked v2 text logs are included in that push. Original output and ZIP remain the known C4 baseline. No competition submission has been sent.

**GitHub checkout limitation:** existing `.gitignore` excludes datasets, caches, trained models, generated output, submission ZIPs, and runtime `.log` files. These remain on this computer and are not included in the push. Claude should use this local workspace to resume at test scoring. A fresh clone on another machine requires transferring the needed datasets and C3/C4/v3 caches separately or rebuilding them; cloning alone does not supply the frozen model or predictions.
