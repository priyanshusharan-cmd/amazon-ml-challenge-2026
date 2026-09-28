"""Compare decision rules on E07 predictions, as-is and with 2x distractors (test-like prior).
Distractor duplication: every unmatched record's prediction is duplicated (fake id) so the S1 sees it twice."""
import polars as pl
from harness import EXP_DIR, Ctx
ctx = Ctx()
P = pl.read_parquet(EXP_DIR / "E07_crossfit_train_pred.parquet").select("q_idx", "s1_idx", pl.col("prob").alias("p"))
unm = P.join(ctx.lab, on="q_idx", how="anti")
OFF = 20_000_000
dup = unm.with_columns((pl.col("q_idx") + OFF).alias("q_idx"))
b2 = 0.25
def expf(df, floor=0.0):
    d = df.filter(pl.col("p") >= 0.02).sort(["s1_idx", "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over("s1_idx").alias("S"), pl.int_range(1, pl.len() + 1).over("s1_idx").alias("m"),
        pl.col("p").sum().over("s1_idx").alias("T"), (1 - pl.col("p")).clip(1e-9).log().sum().over("s1_idx").exp().alias("p_none"))
    d = d.with_columns(((1 + b2) * pl.col("S") / (b2 * pl.col("T") + pl.col("m"))).alias("ef"))
    bm = d.group_by("s1_idx").agg(pl.col("ef").max().alias("efmax"), pl.col("m").get(pl.col("ef").arg_max()).alias("mstar"), pl.col("p_none").first())
    return d.join(bm, on="s1_idx").filter((pl.col("efmax") > pl.col("p_none")) & (pl.col("m") <= pl.col("mstar")) & (pl.col("p") >= floor))
for tag, data in [("as-is", P), ("2x distractors", pl.concat([P, dup]))]:
    for t in (0.6, 0.65, 0.7, 0.75):
        r = ctx.f05(data.filter(pl.col("p") >= t)); print(f"{tag:15s} threshold {t}: F={r['F']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}", flush=True)
    for fl in (0.0, 0.5):
        r = ctx.f05(expf(data, fl)); print(f"{tag:15s} expected-F floor {fl}: F={r['F']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}", flush=True)
