"""Stage-2 data: one row per S2/S3 record = its stage-1 best pair, with
  * the 64 stage-1 features of that pair, stage-1 p1/p2/p3,
  * GROUP features: how the record relates to the other records whose best S1 is the same entity.
Usage: python s2_build.py train|test   -> cache/experiments/s2_<split>/part_*.parquet"""
import shutil
import sys

import polars as pl

sys.path.insert(0, "../business_entity_resolution/src")
from harness import EXP_DIR, cache_path, mem  # noqa: E402
from features import FEATURES  # noqa: E402

CONF = 0.5
split = sys.argv[1]
out_dir = EXP_DIR / f"s2_{split}"


def qtable(split):
    """top-3 stage-1 candidates per record (train: already built; test: from test_scored)."""
    if split == "train":
        return pl.read_parquet(EXP_DIR / "qtable.parquet").select("q_idx", "s1_idx", "p1", "p2", "p3")
    path = EXP_DIR / f"qtable_{split}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    sc = pl.scan_parquet(cache_path(f"{split}_scored.parquet"))
    n = sc.select(pl.col("q_idx").max()).collect().item() + 1
    parts = []
    for s in range(0, n, 1_500_000):
        d = sc.filter((pl.col("q_idx") >= s) & (pl.col("q_idx") < s + 1_500_000)).collect()
        parts.append(d.sort(["q_idx", "p"], descending=[False, True]).group_by("q_idx", maintain_order=True).agg(
            pl.col("s1_idx").first(), pl.col("p").get(0).alias("p1"),
            pl.col("p").get(1, null_on_oob=True).fill_null(0.0).alias("p2"),
            pl.col("p").get(2, null_on_oob=True).fill_null(0.0).alias("p3")))
        print("qtable", split, s, mem(), flush=True)
    t = pl.concat(parts)
    t.write_parquet(path)
    return t


qt = qtable(split)
qn = pl.concat([pl.read_parquet(cache_path(f"{split}_{s}_norm.parquet"), columns=["addr_nums", "name_core"])
                for s in ("source2", "source3")]).with_row_index("q_idx")
d = qt.join(qn, on="q_idx")
del qn
d = d.with_columns(
    (pl.col("p1") >= CONF).cast(pl.Int32).alias("conf"),
    pl.col("addr_nums").str.split(" ").list.first().alias("num1"),
    pl.col("name_core").str.replace_all(" ", "").alias("nkey"),
)


def support(keys, tag):
    """#other records on the same S1 sharing `keys` (all / confident)."""
    grp = ["s1_idx"] + keys
    return [
        (pl.len().over(grp) - 1).alias(f"sup_{tag}_all"),
        (pl.col("conf").sum().over(grp) - pl.col("conf")).alias(f"sup_{tag}_conf"),
    ]


d = d.with_columns(
    (pl.len().over("s1_idx") - 1).alias("g_all"),
    (pl.col("conf").sum().over("s1_idx") - pl.col("conf")).alias("g_conf"),
    (pl.col("p1").sum().over("s1_idx") - pl.col("p1")).alias("g_psum"),
    *support(["addr_nums"], "nums"), *support(["num1"], "num1"), *support(["nkey"], "name"),
    *support(["addr_nums", "nkey"], "both"),
)
no_nums = pl.col("addr_nums") == ""
d = d.with_columns([pl.when(no_nums).then(-1).otherwise(pl.col(c)).alias(c)
                    for c in ["sup_nums_all", "sup_nums_conf", "sup_num1_all", "sup_num1_conf", "sup_both_all", "sup_both_conf"]])
d = d.with_columns(
    (pl.col("sup_nums_conf") / pl.col("g_conf").clip(1)).alias("sup_nums_frac"),
    (pl.col("sup_name_conf") / pl.col("g_conf").clip(1)).alias("sup_name_frac"),
    (pl.col("p1") - pl.col("p2")).alias("margin12"),
).drop("addr_nums", "name_core", "num1", "nkey", "conf")
print("group features", d.shape, mem(), flush=True)

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
pairs = d.select("q_idx", "s1_idx")
for i, f in enumerate(sorted(cache_path(f"{split}_feats").glob("*.parquet"))):
    x = pl.read_parquet(f).join(pairs, on=["q_idx", "s1_idx"], how="semi")
    x = x.join(d, on=["q_idx", "s1_idx"], how="left")
    x.write_parquet(out_dir / f"part_{i:03d}.parquet")
    if i % 10 == 0:
        print(f.name, x.shape, mem(), flush=True)
print("done", mem())
