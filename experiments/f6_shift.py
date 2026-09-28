"""Train-vs-test shift check: how often stage 3 flips stage-1 decisions, by country and by group size."""
import polars as pl
from harness import EXP_DIR, cache_path
tr = pl.read_parquet(EXP_DIR / "E07_crossfit_train_pred.parquet").join(pl.read_parquet(EXP_DIR / "qtable.parquet", columns=["q_idx", "p1"]), on="q_idx")
te = pl.read_parquet(EXP_DIR / "SUBMISSION_E09" / "test_stage3_pred.parquet").join(pl.read_parquet(EXP_DIR / "qtable_test.parquet", columns=["q_idx", "p1"]), on="q_idx")
for tag, d, split in [("train", tr, "train"), ("test", te, "test")]:
    s1 = pl.read_parquet(cache_path(f"{split}_source1.parquet"), columns=["country"]).with_row_index("s1_idx")
    d = d.join(s1, on="s1_idx").with_columns(pl.len().over("s1_idx").alias("grp"))
    print(f"\n== {tag}: records {d.height:,}  stage1 p1>=0.7: {(d['p1']>=0.7).mean():.4f}  stage3 prob>=0.65: {(d['prob']>=0.65).mean():.4f}")
    print(d.group_by("country").agg(pl.len(), (pl.col("p1") >= 0.7).mean().alias("s1_acc"), (pl.col("prob") >= 0.65).mean().alias("s3_acc"),
        ((pl.col("p1") < 0.7) & (pl.col("prob") >= 0.65)).mean().alias("rescued"), ((pl.col("p1") >= 0.7) & (pl.col("prob") < 0.65)).mean().alias("dropped"),
        pl.col("grp").mean().alias("mean_group")).sort("country"))
    print(d.with_columns(pl.col("grp").clip(1, 12).alias("g")).group_by("g").agg(pl.len(), ((pl.col("p1") < 0.7) & (pl.col("prob") >= 0.65)).mean().alias("rescued")).sort("g"))
