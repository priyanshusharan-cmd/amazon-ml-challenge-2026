"""Forensics step 2: error classes, counts, probability bins and the F0.5 available from fixing each class."""
import polars as pl

from harness import EXP_DIR, Ctx, fmt, mem

T = 0.7
ctx = Ctx()
qt = pl.read_parquet(EXP_DIR / "qtable.parquet")
val = ctx.val.select("s1_idx")
nt = ctx.n_true
isval = pl.col("s1_idx").is_in(val["s1_idx"].implode())
tval = pl.col("true_s1").is_in(val["s1_idx"].implode()).fill_null(False)
b = qt.filter(isval | tval).join(nt, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0).alias("pred_ntrue"))
acc = pl.col("p1") >= T
b = b.with_columns(
    pl.when(acc & (pl.col("s1_idx") == pl.col("true_s1"))).then(pl.lit("TP"))
      .when(acc & isval & (pl.col("pred_ntrue") == 0)).then(pl.lit("FP_singleton_s1"))
      .when(acc & isval & pl.col("true_s1").is_null()).then(pl.lit("FP_distractor"))
      .when(acc & isval & (pl.col("s1_idx") != pl.col("true_s1"))).then(pl.lit("FP_wrong_s1"))
      .when(tval & ~pl.col("retrieved")).then(pl.lit("FN_A_not_retrieved"))
      .when(tval & (pl.col("s1_idx") != pl.col("true_s1"))).then(pl.lit("FN_B_best_wrong"))
      .when(tval & ~acc).then(pl.lit("FN_B_below_threshold"))
      .otherwise(pl.lit("TN_or_offfold")).alias("cls"))
b.select("q_idx", "cls").write_parquet(EXP_DIR / "val_classes.parquet")
print(b.group_by("cls").agg(pl.len().alias("n"), pl.col("p1").median().alias("p1_median"),
                            (pl.col("p1") - pl.col("p2")).median().alias("margin_median")).sort("n", descending=True))
err = b.filter(~pl.col("cls").is_in(["TP", "TN_or_offfold"])).with_columns(
    pl.col("p1").cut([0.05, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]).alias("p1_bin"))
print(err.group_by("p1_bin", "cls").len().pivot(on="cls", index="p1_bin").sort("p1_bin"))
# FPs by margin
fp = b.filter(pl.col("cls").str.starts_with("FP")).with_columns((pl.col("p1") - pl.col("p2")).cut([0.3, 0.6, 0.9]).alias("margin"))
print(fp.group_by("margin", "cls").len().pivot(on="cls", index="margin").sort("margin"))
# multi-match extra predictions: FPs on S1s that also have TPs
tp_s1 = b.filter(pl.col("cls") == "TP").select("s1_idx").unique()
print("FPs landing on S1 that also has >=1 TP (extra predictions):",
      b.filter(pl.col("cls").str.starts_with("FP")).join(tp_s1, on="s1_idx", how="semi").height,
      "| FPs on S1 with no TP:", b.filter(pl.col("cls").str.starts_with("FP")).join(tp_s1, on="s1_idx", how="anti").height)

base = qt.filter(acc).select("q_idx", "s1_idx")
r0 = ctx.f05(base)
print("\nbaseline", fmt(r0))
cls_q = b.select("q_idx", "cls")
rows = []
for c in ["FP_singleton_s1", "FP_distractor", "FP_wrong_s1"]:
    r = ctx.f05(base.join(cls_q.filter(pl.col("cls") == c), on="q_idx", how="anti"))
    rows.append((c, r["F"] - r0["F"]))
for c in ["FN_B_below_threshold", "FN_B_best_wrong", "FN_A_not_retrieved"]:
    add = b.filter(pl.col("cls") == c).select("q_idx", pl.col("true_s1").alias("s1_idx"))
    r = ctx.f05(pl.concat([base.join(add, on="q_idx", how="anti"), add]))
    rows.append((c, r["F"] - r0["F"]))
tot = sum(g for _, g in rows)
print("\nF0.5 available from fixing each class alone:")
for c, g in sorted(rows, key=lambda x: -x[1]):
    print(f"  {c:24s} +{g:.5f}  ({100*g/(1-r0['F']):.1f}% of total loss)")
print(mem())
