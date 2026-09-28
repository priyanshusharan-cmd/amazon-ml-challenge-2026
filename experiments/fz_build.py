"""E05 features: fuzzy similarity of each record to its siblings (other records whose stage-1 best S1 is the same).
usage: python fz_build.py train|test  -> cache/experiments/fz_<split>.parquet"""
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from harness import EXP_DIR, cache_path, mem

split = sys.argv[1]
CONF = 0.5
qt = pl.read_parquet(EXP_DIR / ("qtable.parquet" if split == "train" else f"qtable_{split}.parquet"), columns=["q_idx", "s1_idx", "p1"])
st = pl.concat([pl.read_parquet(cache_path(f"{split}_{s}_norm.parquet"), columns=["addr_norm", "name_core"])
                for s in ("source2", "source3")]).with_row_index("q_idx")
d = qt.join(st, on="q_idx").with_columns((pl.col("p1") >= CONF).alias("conf")).drop("p1")
del st
print("loaded", d.shape, mem(), flush=True)
n_s1 = int(d["s1_idx"].max()) + 1
STEP = 150_000
outs = []
for s in range(0, n_s1, STEP):
    c = d.filter((pl.col("s1_idx") >= s) & (pl.col("s1_idx") < s + STEP))
    p = c.join(c.select("s1_idx", pl.col("q_idx").alias("o_idx"), pl.col("addr_norm").alias("o_addr"),
                        pl.col("name_core").alias("o_name"), pl.col("conf").alias("o_conf")), on="s1_idx")
    p = p.filter(pl.col("q_idx") != pl.col("o_idx"))
    a1, a2 = p["addr_norm"].to_list(), p["o_addr"].to_list()
    n1, n2 = p["name_core"].to_list(), p["o_name"].to_list()
    p = p.select("q_idx", "o_conf").with_columns(
        pl.Series("sa", process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
        pl.Series("sa_r", process.cpdist(a1, a2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
        pl.Series("sn", process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)))
    del a1, a2, n1, n2
    p = p.with_columns(pl.min_horizontal("sa_r", "sn").alias("sboth"))
    agg = p.group_by("q_idx").agg(
        pl.col("sa").filter(pl.col("o_conf")).max().alias("fz_addr_conf"),
        pl.col("sa_r").filter(pl.col("o_conf")).max().alias("fz_addr_r_conf"),
        pl.col("sn").filter(pl.col("o_conf")).max().alias("fz_name_conf"),
        pl.col("sboth").filter(pl.col("o_conf")).max().alias("fz_both_conf"),
        (pl.col("sboth").filter(pl.col("o_conf")) >= 90).sum().alias("fz_n90_conf"),
        pl.col("sa_r").filter(~pl.col("o_conf")).max().alias("fz_addr_r_non"),
        pl.col("sboth").filter(~pl.col("o_conf")).max().alias("fz_both_non"),
        (pl.col("sboth").filter(~pl.col("o_conf")) >= 90).sum().alias("fz_n90_non"),
    )
    outs.append(agg)
    if (s // STEP) % 3 == 0:
        print(s, p.height, mem(), flush=True)
    del p
FZ = pl.concat(outs)
FZ = qt.select("q_idx").join(FZ, on="q_idx", how="left").with_columns(pl.all().exclude("q_idx").fill_null(-1).cast(pl.Float32))
FZ.write_parquet(EXP_DIR / f"fz_{split}.parquet")
print("done", FZ.shape, mem())
