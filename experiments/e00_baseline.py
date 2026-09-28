import json, polars as pl
from harness import Ctx, BASELINE, LEDGER, fmt
ctx = Ctx()
best = pl.read_parquet(BASELINE / "train_best.parquet")
r = ctx.f05(best.filter(pl.col("p1") >= 0.7))
print("BASELINE reproduced:", fmt(r))
with open(LEDGER, "a") as fh:
    fh.write(json.dumps({"experiment": "E00_BASELINE_BEST", **r, "threshold": 0.7, "promoted": True,
                         "note": "stage-1 LightGBM 64 feats, argmax per record, p>=0.7"}) + "\n")
