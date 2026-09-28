"""Declared v3 comparison and frozen, explicitly reused CONF evaluation.

Run alone after scoring; no fitting or submission writes occur here:
    python experiments/v3/compare.py direct
    python experiments/v3/compare.py roles
    python experiments/v3/compare.py roles --fold CONF --threshold frozen --blend frozen

Declared before outcomes: both direct and roles compare standalone probabilities
and the arithmetic mean of the challenger and C4 probabilities. Each uses the
same seven thresholds. A blend never changes C3's selected candidate.

Each mode writes comparison.json. Only a passing candidate gets a mode-local
frozen_selection.json. Selection is by mean F0.5 across multiplicities 1/2/4,
then ordinary DEV F0.5. Wrong-branch stress and reused CONF remain further checks,
so passing these gates does not itself promote a submission.
"""
import argparse
import gc
import json

from common import BASELINE, RUN, atomic_json, memory, pl, qtruth, s1_table, sha
from evaluation import EvaluationContext
import numpy as np


THRESHOLDS = (.60, .65, .70, .75, .80, .85, .90)
MULTIPLICITIES = (1, 2, 4)
BASELINE_THRESHOLD = .70
GATES = {
    "ordinary_dev_min_delta": .0003,
    "each_negative_stress_min_delta": -.0002,
    "each_country_x1_min_delta": -.001,
    "zero_match_x1_min_delta": -.001,
}


def _delta(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def assess_candidate(rows, baseline_rows, threshold, baseline_weight):
    """Apply every gate independently, including zero-match and each country."""
    c = {row["negative_multiplicity"]: row for row in rows}
    b = {row["negative_multiplicity"]: row for row in baseline_rows}
    deltas = {str(k): _delta(c[k]["overall"]["macro_f05"], b[k]["overall"]["macro_f05"])
              for k in MULTIPLICITIES}
    country_deltas = {
        country: _delta(c[1]["by_country"].get(country, {}).get("macro_f05"), summary["macro_f05"])
        for country, summary in b[1]["by_country"].items()
    }
    zero_delta = _delta(c[1]["zero_match"]["macro_f05"], b[1]["zero_match"]["macro_f05"])
    checks = {
        "ordinary_dev": deltas["1"] is not None and deltas["1"] >= GATES["ordinary_dev_min_delta"],
        "each_negative_stress": all(d is not None and d >= GATES["each_negative_stress_min_delta"]
                                    for d in deltas.values()),
        "each_country_x1": bool(country_deltas) and all(d is not None and d >= GATES["each_country_x1_min_delta"]
                                                       for d in country_deltas.values()),
        "zero_match_x1": zero_delta is not None and zero_delta >= GATES["zero_match_x1_min_delta"],
    }
    return {
        "threshold": float(threshold),
        "baseline_weight": float(baseline_weight),
        "recipe": "standalone" if baseline_weight == 0 else "arithmetic_mean_50_50",
        "passes_dev_gates": all(checks.values()),
        "gate_checks": checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "delta_by_negative_multiplicity": deltas,
        "delta_by_country_x1": country_deltas,
        "delta_zero_match_x1": zero_delta,
        "mean_stress_f05": float(np.mean([c[k]["overall"]["macro_f05"] for k in MULTIPLICITIES])),
        "ordinary_f05": c[1]["overall"]["macro_f05"],
        "metrics": rows,
    }


def _rank(candidate):
    # Later fields only make exact ties deterministic; no further search occurs.
    return (candidate["mean_stress_f05"], candidate["ordinary_f05"],
            -candidate["baseline_weight"], -abs(candidate["threshold"] - .7))


def _load_scores(mode):
    memory(3)
    challenger = pl.read_parquet(RUN / mode / "pred_v2train.parquet", columns=["q_idx", "s1_idx", "p"]).sort("q_idx")
    baseline = pl.read_parquet(BASELINE / "top3_v2train.parquet", columns=["q_idx", "s1_idx", "p1"]).sort("q_idx")
    if challenger.height != baseline.height:
        raise ValueError("Challenger and baseline query coverage differs")
    for column in ("q_idx", "s1_idx"):
        if not np.array_equal(challenger[column].to_numpy(), baseline[column].to_numpy()):
            raise ValueError(f"Challenger and baseline {column} differ; blending would change candidate semantics")
    return challenger, baseline


def _inputs(mode):
    return {
        "challenger_predictions_sha256": sha(RUN / mode / "pred_v2train.parquet"),
        "baseline_predictions_sha256": sha(BASELINE / "top3_v2train.parquet"),
        "comparison_code_sha256": sha(__file__),
    }


def _recipe_predictions(challenger, baseline, baseline_weight):
    if baseline_weight == 0:
        return challenger
    # Keep the arithmetic operation and dtype fixed for CONF and export.
    probability = (challenger["p"].to_numpy() * np.float32(1 - baseline_weight)
                   + baseline["p1"].to_numpy() * np.float32(baseline_weight)).astype(np.float32)
    return challenger.with_columns(pl.Series("p", probability))


def compare_dev(mode):
    dest = RUN / mode
    dest.mkdir(exist_ok=True)
    frozen_path = dest / "frozen_selection.json"
    inputs = _inputs(mode)
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen["inputs"] != inputs:
            raise RuntimeError("Frozen selection inputs/code changed; use a new experiment, not this frozen run")
        if (dest / "confirmation_reused.json").exists():
            raise RuntimeError("This candidate has already been checked on reused CONF; do not select it again")
    challenger, baseline = _load_scores(mode)
    ctx = EvaluationContext(s1_table(), qtruth(), fold="DEV")
    baseline_eval = ctx.evaluate(baseline, thresholds=[BASELINE_THRESHOLD], negative_multiplicities=MULTIPLICITIES)
    candidates = []
    weights = (0., .5)
    for weight in weights:
        memory(2.2)
        predictions = _recipe_predictions(challenger, baseline, weight)
        evaluated = ctx.evaluate(predictions, thresholds=THRESHOLDS, negative_multiplicities=MULTIPLICITIES)
        for threshold in THRESHOLDS:
            rows = [row for row in evaluated["results"] if row["threshold"] == threshold]
            candidate = assess_candidate(rows, baseline_eval["results"], threshold, weight)
            candidates.append(candidate)
            print(f"{mode} baseline_weight={weight:.1f} t={threshold:.2f}: "
                  f"DEV={candidate['ordinary_f05']:.6f}, mean(x1,x2,x4)={candidate['mean_stress_f05']:.6f}, "
                  f"{'PASS' if candidate['passes_dev_gates'] else 'FAIL ' + ','.join(candidate['failed_gates'])}", flush=True)
        del predictions
        gc.collect()
    candidates.sort(key=_rank, reverse=True)
    eligible = [candidate for candidate in candidates if candidate["passes_dev_gates"]]
    selected = eligible[0] if eligible else None
    report = {
        "mode": mode,
        "fold": "DEV",
        "inputs": inputs,
        "thresholds_declared": THRESHOLDS,
        "baseline_weights_declared": weights,
        "baseline_threshold_fixed": BASELINE_THRESHOLD,
        "negative_multiplicities": MULTIPLICITIES,
        "gates": GATES,
        "selection_rule": "Highest mean F0.5 over x1/x2/x4 among candidates passing every gate; then ordinary DEV F0.5",
        "baseline": baseline_eval,
        "candidates_ranked": candidates,
        "n_passing": len(eligible),
        "selected": selected,
        "promotion_status": "pending wrong-branch stress and reused CONF" if selected else "no qualifying candidate; preserve C4",
        "limitations": "DEV and negative-shift diagnostics do not establish leaderboard performance or France accuracy.",
    }
    atomic_json(dest / "comparison.json", report)
    if selected is not None:
        frozen = {
            "mode": mode,
            "selected_on": "DEV",
            "threshold": selected["threshold"],
            "baseline_weight": selected["baseline_weight"],
            "recipe": selected["recipe"],
            "inputs": inputs,
            "baseline_threshold": BASELINE_THRESHOLD,
            "negative_multiplicities": MULTIPLICITIES,
            "gates": GATES,
            "selection_metrics": selected,
            "status": "frozen before reused CONF; wrong-branch stress and confirmation checks remain",
        }
        if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
            raise RuntimeError("A different selection is already frozen; refusing to overwrite it")
        atomic_json(frozen_path, frozen)
        print(f"Frozen {mode}: threshold={selected['threshold']}, baseline_weight={selected['baseline_weight']}; "
              "not yet promoted", flush=True)
    else:
        print("No candidate passes every declared gate. C4 remains the retained submission.", flush=True)


def confirm_reused(mode, threshold, blend):
    dest = RUN / mode
    path = dest / "frozen_selection.json"
    if not path.exists():
        raise RuntimeError("No passing DEV candidate was frozen; CONF evaluation is not allowed")
    frozen = json.loads(path.read_text())
    if frozen["mode"] != mode:
        raise RuntimeError("Frozen model mode differs")
    actual_threshold = frozen["threshold"] if threshold == "frozen" else float(threshold)
    actual_blend = frozen["baseline_weight"] if blend == "frozen" else float(blend)
    if actual_threshold != frozen["threshold"] or actual_blend != frozen["baseline_weight"]:
        raise ValueError("CONF can only evaluate the frozen threshold and blend; tuning is prohibited")
    inputs = _inputs(mode)
    if inputs != frozen["inputs"]:
        raise RuntimeError("Prediction or comparison code fingerprint differs from frozen selection")
    output = dest / "confirmation_reused.json"
    if output.exists():
        prior = json.loads(output.read_text())
        if prior["inputs"] != inputs:
            raise RuntimeError("Prior reused-CONF result has different inputs")
        print(f"Identical reused-CONF result already exists: {output}", flush=True)
        return
    challenger, baseline = _load_scores(mode)
    ctx = EvaluationContext(s1_table(), qtruth(), fold="CONF", reused_conf=True)
    baseline_eval = ctx.evaluate(baseline, thresholds=[BASELINE_THRESHOLD], negative_multiplicities=MULTIPLICITIES)
    predictions = _recipe_predictions(challenger, baseline, actual_blend)
    evaluation = ctx.evaluate(predictions, thresholds=[actual_threshold], negative_multiplicities=MULTIPLICITIES)
    # Report deltas without using CONF to change threshold, blend, or model.
    by_k = {row["negative_multiplicity"]: row for row in baseline_eval["results"]}
    deltas = []
    for row in evaluation["results"]:
        base = by_k[row["negative_multiplicity"]]
        deltas.append({
            "negative_multiplicity": row["negative_multiplicity"],
            "overall": _delta(row["overall"]["macro_f05"], base["overall"]["macro_f05"]),
            "by_country": {c: _delta(v["macro_f05"], base["by_country"][c]["macro_f05"])
                           for c, v in row["by_country"].items()},
            "zero_match": _delta(row["zero_match"]["macro_f05"], base["zero_match"]["macro_f05"]),
        })
    report = {
        "mode": mode, "fold": "CONF", "threshold": actual_threshold, "baseline_weight": actual_blend,
        "inputs": inputs, "baseline": baseline_eval, "candidate": evaluation, "deltas": deltas,
        "status": "reused confirmation; no new independent holdout; no selection or promotion performed here",
    }
    atomic_json(output, report)
    ordinary = evaluation["results"][0]["overall"]["macro_f05"]
    print(f"Reused CONF: {ordinary:.6f}, delta={deltas[0]['overall']:+.6f}; frozen recipe unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["direct", "roles"])
    parser.add_argument("--fold", choices=["DEV", "CONF"], default="DEV")
    parser.add_argument("--threshold", default="frozen", help="CONF only: frozen, or the exact frozen threshold")
    parser.add_argument("--blend", default="frozen", help="CONF only: frozen, or the exact frozen baseline probability weight")
    args = parser.parse_args()
    if args.fold == "CONF":
        confirm_reused(args.mode, args.threshold, args.blend)
    elif args.threshold != "frozen" or args.blend != "frozen":
        parser.error("DEV must use the complete predeclared grid; threshold/blend overrides are CONF-only")
    else:
        compare_dev(args.mode)
