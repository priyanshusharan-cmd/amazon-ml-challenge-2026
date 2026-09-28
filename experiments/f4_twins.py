"""Forensics step 4: sibling ('twin') structure among records that share the same best S1.
For each record we look at the other records whose best S1 is the same entity:
 - n_sib_nums_same : siblings with exactly the same house-number set as this record
 - dev_support     : when this record's numbers differ from the S1's, how many siblings share that deviation
If distractors come in clusters (a fake entity with several variants), dev_support separates them from typos."""
import polars as pl

from harness import EXP_DIR, cache_path, mem

qt = pl.read_parquet(EXP_DIR / "qtable.parquet", columns=["q_idx", "s1_idx", "p1"])
cls = pl.read_parquet(EXP_DIR / "val_classes.parquet")
qn = pl.concat([pl.read_parquet(cache_path(f"train_{s}_norm.parquet"), columns=["addr_nums", "name_core"])
                for s in ("source2", "source3")]).with_row_index("q_idx")
s1n = pl.read_parquet(cache_path("train_source1_norm.parquet"), columns=["addr_nums", "name_core"]).with_row_index("s1_idx")
print("loaded", mem(), flush=True)
d = qt.join(qn, on="q_idx").join(s1n.rename({"addr_nums": "s_nums", "name_core": "s_name"}), on="s1_idx")
del qn
d = d.with_columns(
    pl.len().over("s1_idx", "addr_nums").alias("same_nums_group"),
    pl.len().over("s1_idx", "name_core").alias("same_name_group"),
    pl.len().over("s1_idx").alias("group_size"),
    ((pl.col("p1") >= 0.7)).cast(pl.Int32).sum().over("s1_idx").alias("n_accepted_on_s1"),
)
d = d.with_columns((pl.col("addr_nums") != pl.col("s_nums")).alias("nums_dev"),
                   (pl.col("name_core") != pl.col("s_name")).alias("name_dev"))
e = d.join(cls, on="q_idx")
print(mem(), flush=True)
agg = e.filter(pl.col("cls").is_in(["TP", "FP_distractor", "FN_B_below_threshold", "FP_singleton_s1"])).group_by("cls").agg(
    pl.len(),
    pl.col("nums_dev").mean().alias("share_nums_dev"),
    ((pl.col("same_nums_group") - 1).filter(pl.col("nums_dev") & (pl.col("addr_nums") != ""))).mean().alias("mean_dev_support"),
    ((pl.col("same_nums_group") > 1).filter(pl.col("nums_dev") & (pl.col("addr_nums") != ""))).mean().alias("share_dev_supported"),
    ((pl.col("same_name_group") - 1).filter(pl.col("name_dev"))).mean().alias("mean_name_dev_support"),
    pl.col("group_size").mean().alias("mean_group_size"),
    pl.col("n_accepted_on_s1").mean().alias("mean_accepted_on_s1"),
).sort("cls")
print(agg)
