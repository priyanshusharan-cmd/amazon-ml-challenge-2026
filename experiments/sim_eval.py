"""Evaluate BASELINE vs E09 pipeline on the density-shifted simulation (no retraining)."""
import sys
import lightgbm as lgb
import numpy as np
import polars as pl
sys.path.insert(0, "../business_entity_resolution/src")
from harness import EXP_DIR, Ctx, mem
from features import FEATURES
from gfeat import GROUP_BASE, group_features, load_strings
ctx = Ctx()
drop = pl.read_parquet(EXP_DIR / "sim_dropped_s1.parquet")
ctx.lab = ctx.lab.join(drop.rename({"s1_idx": "true_s1"}), on="true_s1", how="anti")  # their records are now distractors
ctx.val = ctx.val.join(drop, on="s1_idx", how="anti")
ctx.n_true = ctx.n_true.join(drop, on="s1_idx", how="anti")
qt = pl.read_parquet(EXP_DIR / "qtable_sim.parquet")
print("records/S1 (sim):", round(qt.height / (2206821 - drop.height), 2))
for t in (0.6, 0.7, 0.8):
    r = ctx.f05(qt.filter(pl.col("p1") >= t)); print(f"BASELINE stage-1 t={t}: F={r['F']:.5f} P={r['P']:.5f} R={r['R']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}", flush=True)
EXTRA = pl.read_parquet(EXP_DIR / "fz_sim.parquet"); EC = [c for c in EXTRA.columns if c != "q_idx"]
S2 = FEATURES + ["p1", "p2", "p3", "margin12"] + GROUP_BASE + EC; S3 = S2 + ["h_" + c for c in GROUP_BASE] + ["q2"]
m2 = [lgb.Booster(model_file=str(EXP_DIR / f"E07_crossfit_stage2_h{i}.txt")) for i in (0, 1)]
m3 = lgb.Booster(model_file=str(EXP_DIR / "E07_crossfit_stage3.txt"))
files = sorted((EXP_DIR / "s2_sim").glob("part_*.parquet"))
o = []
for f in files:
    x = pl.read_parquet(f).join(EXTRA, on="q_idx", how="left"); X = x.select(S2).to_numpy()
    o.append(x.select("q_idx", "s1_idx").with_columns(pl.Series("prob", np.mean([m.predict(X, num_threads=16) for m in m2], 0).astype(np.float32))))
q2 = pl.concat(o)
G = group_features(q2, load_strings("sim"), prefix="h_").join(q2.select("q_idx", pl.col("prob").alias("q2")), on="q_idx")
o = []
for f in files:
    x = pl.read_parquet(f).join(EXTRA, on="q_idx", how="left").join(G, on="q_idx", how="left")
    o.append(x.select("q_idx", "s1_idx").with_columns(pl.Series("prob", m3.predict(x.select(S3).to_numpy(), num_threads=16).astype(np.float32))))
P = pl.concat(o); P.write_parquet(EXP_DIR / "sim_E07_pred.parquet")
print("stage3 acceptance (>=0.65):", round((P["prob"] >= 0.65).mean(), 4), " baseline acceptance (>=0.7):", round((qt["p1"] >= 0.7).mean(), 4), mem())
for t in (0.6, 0.65, 0.7, 0.75, 0.8):
    r = ctx.f05(P.filter(pl.col("prob") >= t)); print(f"E07 stage-3 t={t}: F={r['F']:.5f} P={r['P']:.5f} R={r['R']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}", flush=True)
b2 = 0.25
d = P.filter(pl.col("prob") >= 0.02).sort(["s1_idx", "prob"], descending=[False, True]).with_columns(
    pl.col("prob").cum_sum().over("s1_idx").alias("S"), pl.int_range(1, pl.len() + 1).over("s1_idx").alias("m"),
    pl.col("prob").sum().over("s1_idx").alias("T"), (1 - pl.col("prob")).clip(1e-9).log().sum().over("s1_idx").exp().alias("p_none"))
d = d.with_columns(((1 + b2) * pl.col("S") / (b2 * pl.col("T") + pl.col("m"))).alias("ef"))
bm = d.group_by("s1_idx").agg(pl.col("ef").max().alias("efmax"), pl.col("m").get(pl.col("ef").arg_max()).alias("mstar"), pl.col("p_none").first())
sel = d.join(bm, on="s1_idx").filter((pl.col("efmax") > pl.col("p_none")) & (pl.col("m") <= pl.col("mstar")) & (pl.col("prob") >= 0.5))
r = ctx.f05(sel); print(f"E09 (E07 + expected-F floor 0.5): F={r['F']:.5f} P={r['P']:.5f} R={r['R']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}")
