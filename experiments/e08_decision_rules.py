"""E08: decision-rule experiments on an existing prediction file (no retraining).
 a) global threshold fine grid
 b) segment thresholds: no-address records / S2 vs S3
 c) first match vs additional matches per S1 (rank records on each S1 by prob)
 d) expected-F0.5 subset selection per S1
Segment thresholds are tuned on validation half h0 and reported on half h1 (and vice versa) to avoid optimism.
usage: python e08_decision_rules.py <pred_file> <prob_col>"""
import sys

import numpy as np
import polars as pl

from harness import EXP_DIR, Ctx, best_so_far, cache_path, log, mem

pred_file, col = sys.argv[1], sys.argv[2]
ctx = Ctx()
P = pl.read_parquet(EXP_DIR / pred_file).select("q_idx", "s1_idx", pl.col(col).alias("p"))
flags = pl.concat([pl.read_parquet(cache_path(f"train_{s}_norm.parquet"), columns=["addr_missing"]).with_columns(
    pl.lit(s == "source3").alias("is_s3")) for s in ("source2", "source3")]).with_row_index("q_idx")
P = P.join(flags, on="q_idx")
best = best_so_far()["F"]
grid = np.round(np.arange(0.5, 0.86, 0.025), 3)


def score(pred):
    return ctx.f05(pred.select("q_idx", "s1_idx"))


# a) global
glob = [(t, score(P.filter(pl.col("p") >= t))) for t in grid]
tg, rg = max(glob, key=lambda z: z[1]["F"])
print(f"a) global best t={tg} F={rg['F']:.5f} h0={rg['F_h0']:.5f} h1={rg['F_h1']:.5f}", flush=True)


# b) segment thresholds, cross-fitted over halves
def seg_rule(seg_expr, name):
    res = {}
    for tseg in grid:
        pred = P.filter(pl.when(seg_expr).then(pl.col("p") >= tseg).otherwise(pl.col("p") >= tg))
        res[tseg] = score(pred)
    t0 = max(res, key=lambda t: res[t]["F_h0"])
    t1 = max(res, key=lambda t: res[t]["F_h1"])
    cross = (res[t1]["F_h0"] + res[t0]["F_h1"]) / 2  # tuned on one half, scored on the other
    print(f"b) {name}: t_h0={t0} t_h1={t1} cross-fitted F={cross:.5f} (global {rg['F']:.5f}, delta {cross-rg['F']:+.5f})", flush=True)
    return cross, t0, t1


seg_rule(pl.col("addr_missing"), "no-address records")
seg_rule(pl.col("is_s3"), "Source-3 records")

# c) additional matches: records ranked 2nd+ on their S1 need a different threshold
R = P.filter(pl.col("p") >= 0.3).with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_idx").alias("r_on_s1"))
res = {}
for t_add in grid:
    pred = R.filter(pl.when(pl.col("r_on_s1") == 1).then(pl.col("p") >= tg).otherwise(pl.col("p") >= t_add))
    res[t_add] = score(pred)
for t_first in grid:
    pred = R.filter(pl.when(pl.col("r_on_s1") == 1).then(pl.col("p") >= t_first).otherwise(pl.col("p") >= tg))
    res[("first", t_first)] = score(pred)
k0 = max(res, key=lambda k: res[k]["F_h0"]); k1 = max(res, key=lambda k: res[k]["F_h1"])
cross = (res[k1]["F_h0"] + res[k0]["F_h1"]) / 2
print(f"c) first/additional thresholds: best_h0={k0} best_h1={k1} cross-fitted F={cross:.5f} (delta {cross-rg['F']:+.5f})", flush=True)

# d) expected-F0.5 subset selection per S1
b2 = 0.25
d = P.filter(pl.col("p") >= 0.02).sort(["s1_idx", "p"], descending=[False, True]).with_columns(
    pl.col("p").cum_sum().over("s1_idx").alias("S"), pl.int_range(1, pl.len() + 1).over("s1_idx").alias("m"),
    pl.col("p").sum().over("s1_idx").alias("T"),
    (1 - pl.col("p")).clip(1e-9).log().sum().over("s1_idx").exp().alias("p_none"))
d = d.with_columns(((1 + b2) * pl.col("S") / (b2 * pl.col("T") + pl.col("m"))).alias("ef"))
bm = d.group_by("s1_idx").agg(pl.col("ef").max().alias("efmax"), pl.col("m").get(pl.col("ef").arg_max()).alias("mstar"), pl.col("p_none").first())
pred = d.join(bm, on="s1_idx").filter((pl.col("efmax") > pl.col("p_none")) & (pl.col("m") <= pl.col("mstar")))
re = score(pred)
print(f"d) expected-F0.5 per S1: F={re['F']:.5f} (delta {re['F']-rg['F']:+.5f}) h0={re['F_h0']:.5f} h1={re['F_h1']:.5f}", flush=True)
for mn in (0.3, 0.4, 0.5):
    r = score(pred.filter(pl.col("p") >= mn))
    print(f"   + min p {mn}: F={r['F']:.5f} (delta {r['F']-rg['F']:+.5f})", flush=True)
print(mem())
