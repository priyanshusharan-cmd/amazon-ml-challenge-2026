# v3: locally improved candidate after the leaderboard gap

The user's latest **v2/C4 leaderboard F0.5 is 0.970441**. The new candidate has
improved local validation, but **its leaderboard score is not yet measured**.
Reaching 0.98 on the leaderboard is not established by these experiments.

The model and threshold are frozen. Test feature preparation is complete
(219 batches, 3,387,279 gated records). Test prediction and export are pending;
format validation must complete before using the new output. See
`CLAUDE_HANDOFF.md` for exact continuation steps. No Python job remains running
at handoff.

## Frozen comparison

Both models are evaluated on the same records, with every query allowed to
contribute false positives to the evaluated S1 entities. Metric: macro-per-S1
`5TP/(5TP + 4FP + FN)`, including empty-truth entities.

| Check | Current C4, threshold 0.70 | v3, threshold 0.80 | Delta |
|---|---:|---:|---:|
| DEV | 0.983219 | **0.984226** | +0.001007 |
| Reused CONF | 0.983206 | **0.984174** | +0.000968 |
| DEV, unmatched FP weight x4 | 0.979618 | **0.982036** | +0.002418 |
| Wrong-branch stress, overall | 0.983242 | **0.984302** | +0.001060 |
| Wrong-branch stress, receiving twins | 0.977444 | **0.978384** | +0.000941 |
| DEV entities with no true matches | 0.975234 | **0.985132** | +0.009898 |

DEV improved in both India (+0.001150) and US (+0.000911). Reused CONF improved
in both countries too. The paired family-bootstrap interval for the reused-CONF
gain is **[+0.000861, +0.001079]**, using 1,000 draws over 320,966 name/country
families (441,335 S1 entities). This measures sampling variation within that
dataset; it does not correct validation reuse or estimate the leaderboard gap.

Every predeclared gate in `PLAN.md` passed. Thresholds 0.70–0.85 all passed the
DEV gates. The frozen choice is 0.80, selected by the declared mean score over
unmatched-negative multipliers 1/2/4. The slightly higher ordinary DEV score at
0.75 was not selected. No threshold or blend changed after reused CONF or stress.

## What changed

- Added 25 direct comparison features, including raw premise/unit/floor roles,
  house-number suffix conflicts, alphabetic address agreement, fuzzy unmatched
  name/address tokens, and raw text agreement. Unknown roles remain unknown.
- A regularized LightGBM specialist uses these features alongside the existing
  pair and C3 confidence features. It runs only when C3 probability is in the
  fixed interval [0.001, 0.999]. Otherwise it uses the original C4 probability.
  The final 0.80 decision threshold applies to the combined output.
- Fit used 1,114,229 TRAIN records, excluding every query used for C3 fitting or
  early stopping and excluding candidates owned by DEV/CONF. Internal early
  stopping used 126,566 separate records grouped by country and normalized name.
- Fit parameters: 127 leaves, minimum 250 records per leaf, L2=10, 127 bins,
  four threads, deterministic CPU training. Best iteration was 2,195 within a
  predeclared 2,200-round cap. There was no production refit after validation.

The specialist does not use sibling agreement, cross-query counts, external
data, or country-specific thresholds. C3 retrieval/candidate selection remains
unchanged. Learned direct features are invariant to unrelated query duplication.

An alternative that removed retrieval context was rejected: best standalone DEV
0.973843; best declared 50/50 blend with C4 0.982283. Neither passed the gates.

## Resource and correctness checks

- Heavy stages ran sequentially. Feature/cache preparation used bounded batches;
  training and prediction used four threads. Observed process memory was roughly
  0.25–1.1 GiB during these stages, with substantial system headroom.
- No training, enrichment, or scoring process failed from memory exhaustion.
- Feature and score batches use atomic writes, fingerprints, and reusable
  checkpoints. Source/input fingerprints and a source snapshot accompany the
  role model. Changed cache contents cause a safe stop.
- Metric tests, negative-stress tests, reused-CONF guard tests, cross-source
  text-index alignment tests, and 16 direct-feature tests passed.
- Baseline reproduction matched 0.98321905839 exactly. The bootstrap's separate
  count implementation reproduced both frozen CONF scores to 1e-12.

## Limits

CONF was already opened during v2 work. It is **reused confirmation**, not a new
sealed holdout. The outer split is entity-based, not fully family-disjoint across
all stages. C3/C4 are existing frozen dependencies. France occurs only in test,
so its accuracy remains unmeasured. Unmatched-negative weighting and the existing
wrong-branch suite are diagnostics, not replicas of every test distribution shift.

These results support a candidate submission, not a claim that overfitting is
impossible or that leaderboard F0.5 is now 0.98. A full from-scratch rerun of the
older retrieval and C3/C4 pipeline was not performed.

## Reproduction and artifacts

Run from the repository root using `.venv/Scripts/python.exe -u -X utf8`, with
existing v2 C3/C4 caches available. Run heavy steps one at a time.

```text
experiments/v3/build_rows.py v2train
experiments/v3/enrich_rows.py v2train
experiments/v3/train.py roles
experiments/v3/train.py roles --score v2train
experiments/v3/compare.py roles
```

The frozen recipe and completed confirmation are retained; do not rerun model
selection after inspecting confirmation. For the already-frozen model, the test
path is:

```text
experiments/v3/build_rows.py v2test
experiments/v3/enrich_rows.py v2test
experiments/v3/train.py roles --score v2test
experiments/v3/export.py roles
```

Model, frozen choice, exact JSON metrics, and source snapshot:
`cache/runs/v3/roles/`. The original `output/matching_results.tsv` and
`team_submission.zip` remain the known C4 submission. The intended new artifact
directory is `output/v3_roles/`; its bundle and validation log will record the
final export checks.


## Test export (completed after handoff)
* Test scoring: 9,969,589 records with the frozen model (roles, threshold 0.80, baseline_weight 0).
* `output/v3_roles/matching_results.tsv`: sha256 `fa607c39297b7d9000bf1145a1a2213a0863f99e55d15792c5328c7731ac487f`,
  1,732,544 S1 rows, 1,631,363 with matches, **5,718,652 matched records (57.4%; C4 57.8%)**.
* Official matching validator PASS; streamed candidate-subset check PASS; `candidate_pairs.tsv` = unchanged original (sha `168dd4fd…`).
* Exporter fix: validator subprocess output is now read as UTF-8 (Windows code-page decode crash). Model and feature behaviour are unchanged.
* **Leaderboard: not yet scored.** The C4 file (0.970441 on the leaderboard) remains `output/matching_results.tsv`.
