import polars as pl
from harness import EXP_DIR, Ctx, fmt, mem
ctx = Ctx()
T = 0.675
qt = pl.read_parquet(EXP_DIR / "qtable.parquet", columns=["q_idx", "true_s1", "retrieved"])
P = pl.read_parquet(EXP_DIR / "E02_stage2_group_train_pred.parquet").join(qt, on="q_idx")
val = ctx.val.select("s1_idx")
isval = pl.col("s1_idx").is_in(val["s1_idx"].implode()); tval = pl.col("true_s1").is_in(val["s1_idx"].implode()).fill_null(False)
b = P.filter(isval | tval).join(ctx.n_true, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0))
acc = pl.col("q2") >= T
b = b.with_columns(pl.when(acc & (pl.col("s1_idx") == pl.col("true_s1"))).then(pl.lit("TP"))
  .when(acc & isval & (pl.col("n_true") == 0)).then(pl.lit("FP_singleton_s1"))
  .when(acc & isval & pl.col("true_s1").is_null()).then(pl.lit("FP_distractor"))
  .when(acc & isval & (pl.col("s1_idx") != pl.col("true_s1"))).then(pl.lit("FP_wrong_s1"))
  .when(tval & ~pl.col("retrieved")).then(pl.lit("FN_A_not_retrieved"))
  .when(tval & (pl.col("s1_idx") != pl.col("true_s1"))).then(pl.lit("FN_B_best_wrong"))
  .when(tval & ~acc).then(pl.lit("FN_B_below_threshold")).otherwise(pl.lit("TN")).alias("cls"))
base = P.filter(acc).select("q_idx", "s1_idx"); r0 = ctx.f05(base); print("E02", fmt(r0))
cq = b.select("q_idx", "cls"); rows = []
for c in ["FP_singleton_s1", "FP_distractor", "FP_wrong_s1"]:
    rows.append((c, (b["cls"] == c).sum(), ctx.f05(base.join(cq.filter(pl.col("cls") == c), on="q_idx", how="anti"))["F"] - r0["F"]))
for c in ["FN_B_below_threshold", "FN_B_best_wrong", "FN_A_not_retrieved"]:
    add = b.filter(pl.col("cls") == c).select("q_idx", pl.col("true_s1").alias("s1_idx"))
    rows.append((c, add.height, ctx.f05(pl.concat([base.join(add, on="q_idx", how="anti"), add]))["F"] - r0["F"]))
for c, n, g in sorted(rows, key=lambda x: -x[2]):
    print(f"  {c:22s} n={n:6d} +{g:.5f} ({100*g/(1-r0['F']):.1f}% of loss)")
print(mem())
