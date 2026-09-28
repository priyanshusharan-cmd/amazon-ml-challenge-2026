"""Paired family bootstrap for the already-frozen recipe on reused CONF.

This quantifies within-dataset sampling variability, not selection bias or the
train/test domain gap. It does not select a model, threshold, or blend.
"""
import json

from common import BASELINE, RUN, atomic_json, memory, pl, qtruth, s1_table
import numpy as np


def main():
    memory(3)
    dest = RUN / "roles"
    choice = json.loads((dest / "frozen_selection.json").read_text())
    confirmation = json.loads((dest / "confirmation_reused.json").read_text())
    truth = qtruth()
    catalog = s1_table().sort("s1_idx")
    n = catalog.height
    ntrue = np.bincount(truth[truth >= 0], minlength=n)
    base = pl.read_parquet(BASELINE / "top3_v2train.parquet", columns=["q_idx", "s1_idx", "p1"]).sort("q_idx")
    new = pl.read_parquet(dest / "pred_v2train.parquet").sort("q_idx")
    assert np.array_equal(base["q_idx"].to_numpy(), new["q_idx"].to_numpy())
    assert np.array_equal(base["s1_idx"].to_numpy(), new["s1_idx"].to_numpy())
    qids, sids = base["q_idx"].to_numpy(), base["s1_idx"].to_numpy()
    weight = np.float32(choice["baseline_weight"])
    probabilities = (new["p"].to_numpy() * (np.float32(1) - weight)
                     + base["p1"].to_numpy() * weight).astype(np.float32)
    def f_scores(probability, threshold):
        keep = probability >= threshold
        q, s = qids[keep], sids[keep]
        npred = np.bincount(s, minlength=n)
        tp = np.bincount(s[truth[q] == s], minlength=n)
        denominator = 4 * npred + ntrue
        result = np.ones(n, dtype=np.float64)
        np.divide(5 * tp, denominator, out=result, where=denominator != 0)
        return result
    mask = (catalog["fold"] == "CONF").to_numpy()
    fa = f_scores(base["p1"].to_numpy(), .7)[mask]
    fb = f_scores(probabilities, choice["threshold"])[mask]
    assert abs(fa.mean() - confirmation["baseline"]["results"][0]["overall"]["macro_f05"]) < 1e-12
    assert abs(fb.mean() - confirmation["candidate"]["results"][0]["overall"]["macro_f05"]) < 1e-12
    family = catalog.select(pl.struct("country", pl.col("name_core").str.replace_all(" ", "")).hash(3).alias("family"))["family"].to_numpy()[mask]
    _, inv = np.unique(family, return_inverse=True)
    sums, counts = np.bincount(inv, weights=fb-fa), np.bincount(inv)
    rng = np.random.default_rng(9021)
    draws = np.empty(1000)
    for i in range(len(draws)):
        sampled = rng.integers(0, len(sums), len(sums))
        draws[i] = sums[sampled].sum() / counts[sampled].sum()
    out = {"fold": "CONF (reused)", "delta": float((fb-fa).mean()),
           "ci95_family_bootstrap": np.percentile(draws, [2.5, 97.5]).tolist(),
           "n_entities": len(fa), "n_families": len(sums), "draws": 1000,
           "limitation": "Conditional sampling uncertainty only; does not remove validation reuse/selection bias or establish leaderboard/France performance."}
    atomic_json(dest / "confidence_reused.json", out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
