# v3 progress

## Current handoff status — 26 September 2026

This section supersedes the historical work log below. See `CLAUDE_HANDOFF.md`
for exact continuation commands and `REPORT.md` for measured outcomes.
The roles model is frozen at threshold 0.80, baseline blend weight 0. DEV is
0.984226, reused CONF 0.984174, and all declared stress gates passed.
Test feature enrichment is complete: 219 batches, 3,387,279 gated rows,
manifest `complete: true`. No Python process was running at the final check.
Next: `train.py roles --score v2test`, then `export.py roles`, inspect validation
and bundle files, and obtain a real leaderboard score. No test prediction file
or final v3 export has yet been created. Original C4 output and ZIP are preserved.

## Historical work log

User-confirmed leaderboard result for the current C4 submission: **0.970441**.
Its historical local confirmation result is 0.98321. Treat this as a real
generalization gap, not as success at the requested leaderboard target.

Completed:

- Verified `output/matching_results.tsv` is C4 (SHA256 `5a7f2c062a2e109fa0c8837e0bb0680a7dc6ccfb972636d994378ccd754c237e`).
- Reproduced baseline DEV exactly: 0.98321905839. With unmatched-negative
  multiplicity x2/x4 at t=0.70: 0.981600 / 0.979618. Diagnostics are in
  `cache/runs/v3/diagnostics.json`.
- Cached all 10,320,219 C3-best training pairs using bounded feature batches;
  observed preparation RSS approximately 0.85–1.1 GiB.
- Built a direct-only 51-feature classifier: 2,055,301 fit rows, 465,316
  family-grouped internal early-stop rows; 2,200 trees (predeclared cap reached,
  loss still improving). All fit rows are TRAIN records excluded from C3 fit/ES.
- Added and tested 25 raw-text comparison features, including explicit
  premise/unit/floor distinctions; bounded cross-source text gathering checked.

Direct-only evaluation completed: best standalone DEV 0.973843, best fixed 50/50
blend DEV 0.982283. Neither passes the predeclared gates; both are rejected.
Its complete 10.3M-record scoring pass finished normally with ~0.50 GiB RSS.

Gated raw-text features completed: 2,879,954 selected rows in 213 atomic batches;
observed RSS ~0.9–1.0 GiB (`experiments/v3/enrich_train.log`). The fixed gate is C3 p1 in [.001,.999], chosen
before role-model evaluation from unlabeled score distributions (~27.9% of rows).
Outside the gate, the specialist uses existing C4 probabilities. A full-feature
44k-row smoke batch passed before applying the gate for the actual experiment.

Role specialist fit completed (`experiments/v3/train_roles.log`),
1,114,229 fit rows and 126,566 internal early-stop rows, 108 features, 127 leaves,
min leaf 250, L2=10, deterministic CPU fit with four threads. In addition to the
first-stage fit/ES query exclusion, the selected candidate must be TRAIN-owned.
Source and input fingerprints are saved with the model. Best iteration 2195
(2200-round cap), internal loss 0.117959, fit duration 459 seconds. New raw-name
and premise-similarity features rank among its top features, but no role-model
DEV or CONF outcome has been examined.

Active at last update: checkpointed scoring of role specialist on v2train
(`experiments/v3/score_roles.log`; exec session 74763, if still live). Next run
`experiments/v3/compare.py roles`. If no candidate passes, do not export it.
If a candidate passes, use the frozen recipe for reused CONF and wrong-branch
stress before any test export. `build_rows.py v2stress`, `enrich_rows.py v2stress`
(reuses unchanged-pair role features), `train.py roles --score v2stress`, and
`wrong_branch.py roles` implement the stress path. Test path is analogous using
v2test, then `export.py roles`; original output/ files are preserved.

Next: declared standalone/50-50-blend comparisons; role enrichment and classifier;
freeze any passing candidate before reused CONF and wrong-branch stress checks.
The current submission and candidate file are preserved.

Commands run from repository root with `.venv/Scripts/python.exe -u -X utf8`.
Heavy stages run one at a time. See PLAN.md for selection gates. No leaderboard
improvement can be claimed until a candidate is actually scored on the portal.

## Continuation after handoff (Claude, 26 Sep 2026 evening)
- Pre-flight: no Python processes; `role_gate_v2test` manifest complete (219/219, 3,387,279 rows); output/ = C4 (sha 5a7f2c06…).
- Test scoring: `train.py roles --score v2test` completed on the first attempt (bounded 4-attempt retry wrapper, resumable
  checkpoints): **9,969,589 rows** → `cache/runs/v3/roles/pred_v2test.parquet`. Peak RSS ~0.6 GiB. Log: `score_test.log`.
- Export: the first 3 attempts failed identically inside the validator call. The official validator prints non-ASCII (em dash)
  in the Windows code page, and `export.py` decoded it as UTF-8, so stdout was None and writing validation.log crashed.
  Fix (no model/feature change): run the validator with `PYTHONIOENCODING=utf-8`, `errors="replace"`. Re-run passed.
- `output/v3_roles/`: `matching_results.tsv` sha256 fa607c39297b7d9000bf1145a1a2213a0863f99e55d15792c5328c7731ac487f,
  1,732,544 S1 rows (1,631,363 non-empty), **5,718,652 matched records (57.4% of test records; C4 57.8%)**,
  official matching validator PASS, streamed candidate-subset PASS, candidate_pairs.tsv hard link sha 168dd4fd… (unchanged).
- Baseline preserved: output/matching_results.tsv sha 5a7f2c06… (C4), team_submission.zip untouched.
- Not done: v3 competition ZIP (v3 depends on C3/C4 caches; not yet self-contained). Leaderboard score of v3: **not measured**.
