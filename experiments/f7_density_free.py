"""Density-free comparison: pure-similarity rules on the rank-1 candidate, train vs test.
If test acceptance under a fixed similarity rule ~ train, test has similar match density and the
lower stage-1 acceptance on test comes from count-based features (s1_rank1_deg, n_cands...)."""
import polars as pl
from harness import cache_path, mem
cols = ["q_idx", "rank", "cos_comb", "comb_gap", "nm_tset", "ad_tset", "num_q_cov", "num_conflict", "q_addr_missing", "s1_rank1_deg"]
rules = {
    "R1 near-identical (nm>=95 & ad>=95)": (pl.col("nm_tset") >= 95) & (pl.col("ad_tset") >= 95),
    "R2 strong (cos>=0.7 & gap>=0.2)": (pl.col("cos_comb") >= 0.7) & (pl.col("comb_gap") >= 0.2),
    "R3 name>=90 & all numbers shared": (pl.col("nm_tset") >= 90) & (pl.col("num_q_cov") == 1) & (pl.col("num_conflict") == 0) & (pl.col("q_addr_missing") == 0),
    "R4 weak (cos<0.4)": pl.col("cos_comb") < 0.4,
}
for split in ("train", "test"):
    parts = [pl.read_parquet(f, columns=cols).filter(pl.col("rank") == 1) for f in sorted(cache_path(f"{split}_feats").glob("*.parquet"))]
    d = pl.concat(parts)
    print(f"\n== {split}: {d.height:,} rank-1 rows", mem())
    for k, r in rules.items():
        print(f"  {k:36s} {d.select(r.mean()).item():.4f}")
    print("  s1_rank1_deg quantiles:", [round(d['s1_rank1_deg'].quantile(q), 1) for q in (0.1, 0.5, 0.9)], "mean", round(d['s1_rank1_deg'].mean(), 2))
