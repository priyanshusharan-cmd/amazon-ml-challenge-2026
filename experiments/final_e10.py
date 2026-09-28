"""Final submission E10 = E07 stage-3 probabilities + conflict-cluster guard (T1=0.1) + expected-F0.5 (floor 0.5)."""
import sys
import polars as pl
sys.path.insert(0, "../business_entity_resolution/src")
from harness import EXP_DIR
from predict import write_grouped
from train import id_tables
T1 = 0.1
P = pl.read_parquet(EXP_DIR / "SUBMISSION_E09" / "test_stage3_pred.parquet")
x = pl.concat([pl.read_parquet(f, columns=["q_idx", "p1", "num_conflict", "sup_nums_all"]) for f in sorted((EXP_DIR / "s2_test").glob("*.parquet"))])
P = P.join(x, on="q_idx")
guard = (pl.col("num_conflict") == 1) & (pl.col("sup_nums_all") >= 1) & (pl.col("p1") < T1)
print("guarded records:", P.filter(guard).height)
P = P.with_columns(pl.when(guard).then(0.0).otherwise(pl.col("prob")).alias("prob"))
b2 = 0.25
d = P.filter(pl.col("prob") >= 0.02).sort(["s1_idx", "prob"], descending=[False, True]).with_columns(
    pl.col("prob").cum_sum().over("s1_idx").alias("S"), pl.int_range(1, pl.len() + 1).over("s1_idx").alias("m"),
    pl.col("prob").sum().over("s1_idx").alias("T"), (1 - pl.col("prob")).clip(1e-9).log().sum().over("s1_idx").exp().alias("p_none"))
d = d.with_columns(((1 + b2) * pl.col("S") / (b2 * pl.col("T") + pl.col("m"))).alias("ef"))
bm = d.group_by("s1_idx").agg(pl.col("ef").max().alias("efmax"), pl.col("m").get(pl.col("ef").arg_max()).alias("mstar"), pl.col("p_none").first())
sel = d.join(bm, on="s1_idx").filter((pl.col("efmax") > pl.col("p_none")) & (pl.col("m") <= pl.col("mstar")) & (pl.col("prob") >= 0.5))
out = EXP_DIR / "SUBMISSION_E10"; out.mkdir(exist_ok=True)
s1, q = id_tables("test")
n, nm = write_grouped(s1["entity_id"], q["entity_id"], sel.select("s1_idx", "q_idx"), "matched_entity_ids", out / "matching_results.tsv")
print(f"E10 matching_results.tsv: {sel.height:,} matched records over {nm:,}/{n:,} S1 -> {out}")
