"""Check a frozen candidate against the existing labeled wrong-branch suite."""
import argparse
import json

from common import BASELINE, CACHE_DIR, RUN, atomic_json, memory, pl, qtruth, s1_table
from evaluation import EvaluationContext
import numpy as np


def check(mode):
    modeldir = RUN / mode
    frozen = json.loads((modeldir / "frozen_selection.json").read_text())
    baseline = pl.read_parquet(BASELINE / "top3_v2stress.parquet").sort("q_idx")
    new = pl.read_parquet(modeldir / "pred_v2stress.parquet").sort("q_idx")
    for col in ("q_idx", "s1_idx"):
        assert np.array_equal(baseline[col].to_numpy(), new[col].to_numpy()), col
    w = np.float32(frozen["baseline_weight"])
    p = (new["p"].to_numpy() * (np.float32(1) - w) + baseline["p1"].to_numpy() * w).astype(np.float32)
    new = new.with_columns(pl.Series("p", p))
    del p
    dropped = pl.read_parquet(CACHE_DIR / "runs/stress_v2stress/dropped_X.parquet")["s1_idx"].to_numpy()
    twins = pl.read_parquet(CACHE_DIR / "runs/stress_v2stress/twins_Y.parquet")["s1_idx"]
    table, truth = s1_table(), qtruth()
    output = {"threshold": frozen["threshold"], "baseline_weight": float(w), "slices": {}}
    for name in ("overall", "twins"):
        ev = table
        if name == "twins":
            ev = table.with_columns(pl.when(pl.col("s1_idx").is_in(twins.implode()))
                                    .then(pl.col("fold")).otherwise(pl.lit("OTHER")).alias("fold"))
        ctx = EvaluationContext(ev, truth, fold="DEV", dropped_s1=dropped)
        a = ctx.evaluate(baseline, thresholds=[.7], negative_multiplicities=[1])
        b = ctx.evaluate(new, thresholds=[frozen["threshold"]], negative_multiplicities=[1])
        output["slices"][name] = {"baseline": a, "candidate": b,
            "delta": b["results"][0]["overall"]["macro_f05"] - a["results"][0]["overall"]["macro_f05"],
            "zero_match_delta": b["results"][0]["zero_match"]["macro_f05"] - a["results"][0]["zero_match"]["macro_f05"]}
        del ctx
        memory()
    output["gates"] = {"overall_delta_min": -.0002, "twins_delta_min": -.0005, "zero_twins_delta_min": -.001}
    output["passes"] = (output["slices"]["overall"]["delta"] >= -.0002
                         and output["slices"]["twins"]["delta"] >= -.0005
                         and output["slices"]["twins"]["zero_match_delta"] >= -.001)
    atomic_json(modeldir / "wrong_branch.json", output)
    print(json.dumps({"passes": output["passes"], "deltas": {k: v["delta"] for k, v in output["slices"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["direct", "roles"])
    check(parser.parse_args().mode)
