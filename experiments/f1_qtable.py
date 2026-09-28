"""Forensics step 1: compact per-query table for the baseline.
q_idx, top-3 candidates by baseline probability (s1 + p), true_s1, retrieved flag. Streaming, bounded memory."""
import polars as pl

from harness import BASELINE, EXP_DIR, Ctx, cache_path, mem

ctx = Ctx()
print("ctx", mem(), flush=True)
parts = []
scored = pl.scan_parquet(cache_path("train_scored.parquet"))
# process by q_idx ranges to bound memory
N = 10_320_219
STEP = 1_500_000
for s in range(0, N, STEP):
    d = scored.filter((pl.col("q_idx") >= s) & (pl.col("q_idx") < s + STEP)).collect()
    top = (d.sort(["q_idx", "p"], descending=[False, True]).group_by("q_idx", maintain_order=True)
           .agg(pl.col("s1_idx").head(3).alias("s"), pl.col("p").head(3).alias("pp")))
    top = top.with_columns(
        pl.col("s").list.get(0, null_on_oob=True).alias("s1_idx"), pl.col("pp").list.get(0, null_on_oob=True).alias("p1"),
        pl.col("s").list.get(1, null_on_oob=True).alias("s1_2"), pl.col("pp").list.get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
        pl.col("s").list.get(2, null_on_oob=True).alias("s1_3"), pl.col("pp").list.get(2, null_on_oob=True).fill_null(0.0).alias("p3"),
    ).drop("s", "pp")
    lab = ctx.lab.filter((pl.col("q_idx") >= s) & (pl.col("q_idx") < s + STEP))
    hit = d.join(lab, left_on=["q_idx", "s1_idx"], right_on=["q_idx", "true_s1"], how="semi").select("q_idx").with_columns(
        pl.lit(True).alias("retrieved"))
    top = top.join(lab, on="q_idx", how="left").join(hit, on="q_idx", how="left").with_columns(pl.col("retrieved").fill_null(False))
    parts.append(top)
    del d
    print(s, mem(), flush=True)
qt = pl.concat(parts)
qt.write_parquet(EXP_DIR / "qtable.parquet")
print("qtable", qt.shape, mem())
# sanity: identical to baseline best
b = pl.read_parquet(BASELINE / "train_best.parquet")
chk = qt.select("q_idx", "s1_idx", "p1").join(b.select("q_idx", pl.col("s1_idx").alias("b_s1"), pl.col("p1").alias("b_p1")), on="q_idx")
print("argmax agrees with baseline:", (chk["s1_idx"] == chk["b_s1"]).mean(), " p1 max abs diff:", (chk["p1"] - chk["b_p1"]).abs().max())
