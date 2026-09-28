"""Cache only C3's best candidate per query; process each candidate file in bounded batches.

Usage: python experiments/v3/build_rows.py v2train [--enrich]
Base rows are independent of any v3 label selection. Enrichment can be resumed later.
"""
import argparse
import gc
import json
import math

from common import BASELINE, CACHE_DIR, FIRST, RUN, atomic_json, atomic_parquet, memory, pl, sha


def build(split):
    dest = RUN / f"rows_{split}"
    dest.mkdir(exist_ok=True)
    manifest = dest / "manifest.json"
    signature = {"version": 1, "first_scores": sha(FIRST / f"top3_{split}.parquet"),
                 "baseline_scores": sha(BASELINE / f"top3_{split}.parquet")}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["inputs"] != signature:
            raise RuntimeError("Source scores changed; use a new cache directory.")
    else:
        old = {"inputs": signature, "files": {}}
    memory(3)
    top = pl.read_parquet(FIRST / f"top3_{split}.parquet").drop("s1_2")
    base = pl.read_parquet(BASELINE / f"top3_{split}.parquet", columns=["q_idx", "p1"]).rename({"p1": "baseline_p"})
    top = top.join(base, on="q_idx", validate="1:1").with_columns((pl.col("p1") - pl.col("p2")).alias("margin12"))
    del base
    if split == "v2train":
        qf = pl.read_parquet(CACHE_DIR / "runs/v2_data/qf.parquet")
        top = top.join(qf, on="q_idx", validate="1:1")
        del qf
    countries = pl.read_parquet(CACHE_DIR / f"{split}_source1_norm.parquet", columns=["country"]).with_row_index("s1_idx")
    top = top.join(countries, on="s1_idx", validate="m:1")
    del countries
    total = 0
    for f in sorted((CACHE_DIR / f"{split}_feats").glob("*.parquet")):
        n, lo, hi = pl.scan_parquet(f).select(pl.len(), pl.col("q_idx").min(), pl.col("q_idx").max().alias("hi")).collect().row(0)
        k = max(1, math.ceil(n / 500_000))
        for i in range(k):
            out = dest / f"{f.stem}_{i:03d}.parquet"
            if out.name in old["files"] and out.exists() and sha(out) == old["files"][out.name]["sha"]:
                total += old["files"][out.name]["rows"]
                continue
            memory(2.5)
            a, b = lo + (hi + 1 - lo) * i // k, lo + (hi + 1 - lo) * (i + 1) // k
            rng = (pl.col("q_idx") >= a) & (pl.col("q_idx") < b)
            t = top.filter(rng)
            d = pl.scan_parquet(f).filter(rng).collect().join(t, on=["q_idx", "s1_idx"], how="inner", validate="1:1")
            x = pl.scan_parquet(CACHE_DIR / f"{split}_featx" / f.name).filter(rng).collect()
            d = d.join(x, on=["q_idx", "s1_idx"], how="left", validate="1:1")
            del x, t
            atomic_parquet(out, d)
            total += d.height
            old["files"][out.name] = {"sha": sha(out), "rows": d.height}
            atomic_json(manifest, old)
            del d
        print(f"{split} {f.name}: {total:,} best rows cached; {memory()}", flush=True)
    assert total == top.height, (total, top.height)
    old["complete"] = True
    old["rows"] = total
    atomic_json(manifest, old)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["v2train", "v2test", "v2stress"])
    args = ap.parse_args()
    build(args.split)
