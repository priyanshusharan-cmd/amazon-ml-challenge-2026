"""Final test inference for the promoted stage-2/3 configuration.
Requires: cache/experiments/s2_test (s2_build.py test), fz_test.parquet (fz_build.py test).
usage: python final_predict.py <config.json>
config: {"stage2": [model files...] (averaged), "stage3": model file, "extra": "fz_test.parquet",
         "rule": "threshold"|"expected_f", "threshold": t, "out_dir": "..."}"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "../business_entity_resolution/src")
from harness import EXP_DIR, mem  # noqa: E402
from features import FEATURES  # noqa: E402
from gfeat import GROUP_BASE, group_features, load_strings  # noqa: E402
from predict import write_grouped  # noqa: E402
from train import id_tables  # noqa: E402

cfg = json.load(open(sys.argv[1]))
EXTRA = pl.read_parquet(EXP_DIR / cfg["extra"])
EXTRA_COLS = [c for c in EXTRA.columns if c != "q_idx"]
S2 = FEATURES + ["p1", "p2", "p3", "margin12"] + GROUP_BASE + EXTRA_COLS
S3 = S2 + ["h_" + c for c in GROUP_BASE] + ["q2"]
files = sorted((EXP_DIR / "s2_test").glob("part_*.parquet"))
m2 = [lgb.Booster(model_file=str(EXP_DIR / m)) for m in cfg["stage2"]]
m3 = lgb.Booster(model_file=str(EXP_DIR / cfg["stage3"]))
assert m2[0].feature_name() == S2 and m3.feature_name() == S3, "feature list mismatch"

outs = []
for f in files:
    x = pl.read_parquet(f).join(EXTRA, on="q_idx", how="left")
    X = x.select(S2).to_numpy()
    q2 = np.mean([m.predict(X, num_threads=16) for m in m2], axis=0)
    outs.append(x.select("q_idx", "s1_idx").with_columns(pl.Series("prob", q2.astype(np.float32))))
q2 = pl.concat(outs)
print("stage2", q2.height, mem(), flush=True)
G = group_features(q2, load_strings("test"), prefix="h_").join(q2.select("q_idx", pl.col("prob").alias("q2")), on="q_idx")
outs = []
for f in files:
    x = pl.read_parquet(f).join(EXTRA, on="q_idx", how="left").join(G, on="q_idx", how="left")
    outs.append(x.select("q_idx", "s1_idx").with_columns(pl.Series("prob", m3.predict(x.select(S3).to_numpy(), num_threads=16).astype(np.float32))))
P = pl.concat(outs)
out_dir = EXP_DIR / cfg["out_dir"]
out_dir.mkdir(parents=True, exist_ok=True)
P.write_parquet(out_dir / "test_stage3_pred.parquet")
print("stage3", P.height, mem(), flush=True)

if cfg["rule"] == "threshold":
    sel = P.filter(pl.col("prob") >= cfg["threshold"])
else:  # expected-F0.5 subset per S1
    b2 = 0.25
    d = P.filter(pl.col("prob") >= 0.02).sort(["s1_idx", "prob"], descending=[False, True]).with_columns(
        pl.col("prob").cum_sum().over("s1_idx").alias("S"), pl.int_range(1, pl.len() + 1).over("s1_idx").alias("m"),
        pl.col("prob").sum().over("s1_idx").alias("T"),
        (1 - pl.col("prob")).clip(1e-9).log().sum().over("s1_idx").exp().alias("p_none"))
    d = d.with_columns(((1 + b2) * pl.col("S") / (b2 * pl.col("T") + pl.col("m"))).alias("ef"))
    bm = d.group_by("s1_idx").agg(pl.col("ef").max().alias("efmax"), pl.col("m").get(pl.col("ef").arg_max()).alias("mstar"),
                                  pl.col("p_none").first())
    sel = d.join(bm, on="s1_idx").filter((pl.col("efmax") > pl.col("p_none")) & (pl.col("m") <= pl.col("mstar"))
                                         & (pl.col("prob") >= cfg.get("floor", 0.0)))
s1, q = id_tables("test")
n, n_m = write_grouped(s1["entity_id"], q["entity_id"], sel.select("s1_idx", "q_idx"), "matched_entity_ids",
                       out_dir / "matching_results.tsv")
print(f"matching_results.tsv: {sel.height:,} matched records over {n_m:,}/{n:,} S1 entities -> {out_dir}")
