"""Density-shift simulation: drop 19% of train S1 entities (their records become distractors, records/S1
4.7 -> ~5.8 like test). Candidate lists keep only surviving S1s, re-ranked by retrieval score (= the top-k of
the reduced index). Writes cache/sim_* inputs for build_features('sim')."""
import shutil
import polars as pl
from harness import EXP_DIR, cache_path, mem
DROP_PCT = 19
s1 = pl.read_parquet(cache_path("train_source1.parquet"), columns=["entity_id"]).with_row_index("s1_idx")
drop = s1.filter((pl.col("s1_idx").hash(2026) % 100) < DROP_PCT).select("s1_idx")
drop.write_parquet(EXP_DIR / "sim_dropped_s1.parquet")
print("dropped S1:", drop.height, "of", s1.height)
for s in ("source1", "source2", "source3"):
    shutil.copyfile(cache_path(f"train_{s}_norm.parquet"), cache_path(f"sim_{s}_norm.parquet"))
    shutil.copyfile(cache_path(f"train_{s}.parquet"), cache_path(f"sim_{s}.parquet"))
out = cache_path("sim_cands"); out.mkdir(exist_ok=True)
tot = 0
for f in sorted(cache_path("train_cands").glob("*.parquet")):
    c = pl.read_parquet(f).join(drop, on="s1_idx", how="anti")
    c = c.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("q_idx").cast(pl.Int16).alias("rank"))
    c.write_parquet(out / f.name); tot += c.height
print("sim candidate pairs:", tot, mem())
