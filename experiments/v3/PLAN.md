# v3: leaderboard gap investigation

The user reports **0.970441 on the leaderboard for v2/C4**, despite its historical
local confirmation score of 0.98321. This is a measured generalization gap. The
old confirmation set has already been opened and is not a fresh holdout.

## Frozen scope before fitting

1. Preserve the submitted C4 model and output files.
2. Build bounded, resumable best-candidate feature batches from existing C3
   predictions. Fit only on TRAIN records excluded from C3 training/early stopping.
3. Compare a regularized classifier using direct comparison features (excluding
   catalog/retrieval context) and a classifier adding explicit address-number
   roles. The role specialist uses a fixed C3 probability interval [.001,.999]
   (chosen from unlabeled score distributions before role-model outcomes), and
   falls back to frozen C4 probabilities elsewhere. Fit it on all eligible TRAIN
   records in that interval, excluding C3 fit/early-stop queries. This reduces
   expensive text parsing and focuses the new learner on ambiguous pairs.
   For each, compare standalone predictions and a 50/50 arithmetic mean
   with C4 (declared before seeing any v3 evaluation). Use a small, declared
   threshold grid, not per-country threshold fitting.
4. Select using DEV and unmatched-distractor multiplicity stresses (x1/x2/x4).
   Evaluate wrong-branch stress as well if the challenger clears ordinary DEV.
   Country-transfer experiments are diagnostics, not estimates of France accuracy.
5. Freeze the candidate before checking reused CONF. Report it explicitly as
   reused confirmation, with no claim of a new independent holdout.
6. Promote only if DEV improves by at least 0.0003, the added-distractor stress
   does not regress by more than 0.0002, and neither country nor zero-match slice
   regresses by more than 0.001. Otherwise preserve C4 and report the limitation.
   Wrong-branch check: overall delta >= -0.0002, receiving-twin delta >= -0.0005,
   and zero-match-twin delta >= -0.001, each versus C4 at its fixed 0.70 threshold.
7. A new leaderboard score requires a portal evaluation; no local result will be
   presented as an achieved 0.98 leaderboard score.

## Resource limits

One heavy Python process at a time; 4 compute threads; bounded feature batches;
atomic cache writes and stage manifests; preflight RAM checks and a 2 GiB free-RAM
reserve; no giant in-memory candidate-string table or duplicate candidate export.
Existing C4 files are never overwritten by experiments.
