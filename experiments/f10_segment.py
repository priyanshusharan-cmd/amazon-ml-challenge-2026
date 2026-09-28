import polars as pl
from harness import EXP_DIR, Ctx
ctx = Ctx()
def seg(split, predfile):
    P = pl.read_parquet(predfile).select("q_idx", "s1_idx", "prob")
    x = pl.concat([pl.read_parquet(f, columns=["q_idx", "p1", "num_conflict", "sup_nums_all", "sup_nums_conf"]) for f in sorted((EXP_DIR / f"s2_{split}").glob("*.parquet"))])
    return P.join(x, on="q_idx").with_columns(((pl.col("num_conflict") == 1) & (pl.col("sup_nums_all") >= 1)).alias("cc"))
tr = seg("train", EXP_DIR / "E07_crossfit_train_pred.parquet").join(ctx.lab, on="q_idx", how="left").with_columns(
    (pl.col("true_s1") == pl.col("s1_idx")).fill_null(False).alias("correct"))
te = seg("test", EXP_DIR / "SUBMISSION_E09" / "test_stage3_pred.parquet")
for tag, d in (("train", tr), ("test", te)):
    print(f"{tag}: conflict-cluster share of records {d['cc'].mean():.4f}; of those stage-1 accepts {d.filter('cc')['p1'].ge(0.7).mean():.4f}, stage-3 accepts {d.filter('cc')['prob'].ge(0.65).mean():.4f}")
c = tr.filter("cc")
print("train conflict-cluster records: truly matched to their best S1:", round(c["correct"].mean(), 4))
for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 0.7), (0.7, 1.01)):
    s = c.filter((pl.col("p1") >= lo) & (pl.col("p1") < hi))
    print(f"   stage-1 p1 in [{lo},{hi}): n={s.height:,} true-match rate={s['correct'].mean():.4f} stage-3 accept rate={s['prob'].ge(0.65).mean():.4f}")
s = te.filter("cc")
for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 0.7), (0.7, 1.01)):
    q = s.filter((pl.col("p1") >= lo) & (pl.col("p1") < hi))
    print(f"   TEST p1 in [{lo},{hi}): n={q.height:,} stage-3 accept rate={q['prob'].ge(0.65).mean():.4f}")
