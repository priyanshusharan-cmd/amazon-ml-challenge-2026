"""Reproduce C4 DEV and summarize unlabeled country/score distribution shifts.

Run alone, after other heavy processes finish:
    python experiments/v3/diagnose.py

Reads existing score files; does not fit, change predictions, or evaluate CONF.
"""
import gc

from common import BASELINE, CACHE_DIR, FIRST, RUN, atomic_json, memory, pl, qtruth, s1_table
from evaluation import EvaluationContext
import numpy as np


THRESHOLDS = (.60, .65, .70, .75, .80, .85, .90)
EXPECTED_BASELINE_DEV = .98321905839
BUCKET_CUTS = np.array([.10, .40, .60, .65, .70, .75, .80, .85, .90, .95, .99])


def country_score_profile(path, country_code, countries):
    """Small aggregate only; score file is released before the next model loads."""
    memory(2.5)
    scores = pl.read_parquet(path, columns=["q_idx", "s1_idx", "p1"])
    qi = scores["q_idx"].to_numpy()
    si = scores["s1_idx"].to_numpy()
    prob = scores["p1"].to_numpy()
    if not np.isfinite(prob).all():
        raise ValueError(f"Nonfinite model probabilities: {path}")
    code = country_code[si]
    # Exact histogram counts with bounded temporary arrays, no country string join.
    n_bucket = len(BUCKET_CUTS) + 1
    count = np.zeros((len(countries), n_bucket), dtype=np.int64)
    score_sum = np.zeros_like(count, dtype=float)
    accepted = np.zeros((len(countries), len(THRESHOLDS)), dtype=np.int64)
    for start in range(0, len(qi), 500_000):
        end = start + 500_000
        c, p = code[start:end], prob[start:end]
        bucket = np.searchsorted(BUCKET_CUTS, p, side="right")
        flat = c * n_bucket + bucket
        count += np.bincount(flat, minlength=count.size).reshape(count.shape)
        score_sum += np.bincount(flat, weights=p, minlength=count.size).reshape(count.shape)
        for j, threshold in enumerate(THRESHOLDS):
            accepted[:, j] += np.bincount(c[p >= threshold], minlength=len(countries))
    out = {}
    edges = [0., *BUCKET_CUTS.tolist(), 1.]
    for c, country in enumerate(countries):
        total = int(count[c].sum())
        buckets = []
        for b, n in enumerate(count[c]):
            bucket = {
                "lower_inclusive": edges[b],
                "upper": edges[b + 1],
                "upper_inclusive": b == n_bucket - 1,
                "n_records": int(n),
                "share_of_country": float(n / total) if total else None,
                "mean_probability": float(score_sum[c, b] / n) if n else None,
            }
            buckets.append(bucket)
        out[str(country)] = {
            "n_records": total,
            "acceptance": [{"threshold": t, "n_accepted": int(accepted[c, j]),
                            "rate": float(accepted[c, j] / total) if total else None}
                           for j, t in enumerate(THRESHOLDS)],
            "score_buckets": buckets,
        }
    del scores, code
    gc.collect()
    return out


def main():
    print(f"Read-only C4 diagnostics; {memory(3)}", flush=True)
    truth = qtruth()
    ctx = EvaluationContext(s1_table(), truth, fold="DEV")
    baseline = pl.read_parquet(BASELINE / "top3_v2train.parquet", columns=["q_idx", "s1_idx", "p1"])
    evaluation = ctx.evaluate(baseline, thresholds=THRESHOLDS, negative_multiplicities=(1, 2, 4))
    reproduced = next(row["overall"]["macro_f05"] for row in evaluation["results"]
                      if row["threshold"] == .7 and row["negative_multiplicity"] == 1)
    reproduction = {"expected_dev_f05": EXPECTED_BASELINE_DEV, "actual_dev_f05": reproduced,
                    "absolute_error": abs(reproduced - EXPECTED_BASELINE_DEV), "tolerance": 1e-8}
    reproduction["passed"] = reproduction["absolute_error"] <= reproduction["tolerance"]
    report = {
        "leaderboard_c4_reported_by_user": .970441,
        "baseline_reproduction": reproduction,
        "baseline_dev_thresholds": evaluation,
        "country_assignment": "Country of each query's top candidate; retrieval is country constrained.",
        "distribution_profiles": {},
        "limitations": (
            "Test score profiles are unlabeled. Acceptance is not accuracy or match prevalence. "
            "Training score profiles mix fitting and heldout queries and are descriptive only. "
            "Negative-multiplicity stress holds predictions fixed and is not a leaderboard forecast. "
            "No CONF fold metric is calculated by this script."
        ),
    }
    atomic_json(RUN / "diagnostics.json", report)
    del baseline, ctx
    gc.collect()
    if not reproduction["passed"]:
        raise RuntimeError(f"Baseline reproduction failed; saved diagnostic: {reproduction}")
    print(f"C4 DEV reproduced at .70: {reproduced:.11f}", flush=True)
    for row in evaluation["results"]:
        print(f"t={row['threshold']:.2f} unmatched x{row['negative_multiplicity']}: "
              f"F0.5={row['overall']['macro_f05']:.6f} "
              f"zero-match={row['zero_match']['macro_f05']:.6f}", flush=True)
    for split in ("v2train", "v2test"):
        memory(2.5)
        catalog = pl.read_parquet(CACHE_DIR / f"{split}_source1_norm.parquet", columns=["country"])
        countries, country_code = np.unique(catalog["country"].to_numpy(), return_inverse=True)
        country_code = country_code.astype(np.int16)
        del catalog
        profile = {}
        for label, folder in (("baseline_p", BASELINE), ("c3_p", FIRST)):
            profile[label] = country_score_profile(folder / f"top3_{split}.parquet", country_code, countries)
            for country, stats in profile[label].items():
                rate = next(row["rate"] for row in stats["acceptance"] if row["threshold"] == .7)
                print(f"{split}/{label}/{country}: {stats['n_records']:,} records, "
                      f"accept at .70={rate:.4%}", flush=True)
        report["distribution_profiles"][split] = profile
        atomic_json(RUN / "diagnostics.json", report)
        del country_code
        print(memory(), flush=True)
    print(f"Saved {RUN / 'diagnostics.json'}", flush=True)


if __name__ == "__main__":
    main()
