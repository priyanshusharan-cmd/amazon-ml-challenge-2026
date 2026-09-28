"""Which features make stage 3 accept low-stage-1 records on test but not on train? (pred_contrib)"""
import sys
import lightgbm as lgb
import numpy as np
import polars as pl
sys.path.insert(0, "../business_entity_resolution/src")
from harness import EXP_DIR
from features import FEATURES
from gfeat import GROUP_BASE, group_features, load_strings
m3 = lgb.Booster(model_file=str(EXP_DIR / "E07_crossfit_stage3.txt"))
m2 = [lgb.Booster(model_file=str(EXP_DIR / f"E07_crossfit_stage2_h{i}.txt")) for i in (0, 1)]
S3 = m3.feature_name(); S2 = m2[0].feature_name()
def rows(split, predfile, n=20000):
    P = pl.read_parquet(predfile).join(pl.read_parquet(EXP_DIR / ("qtable.parquet" if split == "train" else f"qtable_{split}.parquet"), columns=["q_idx", "p1"]), on="q_idx")
    # records with low stage-1 prob and a number that differs from the S1: the contested pattern
    pick = P.filter(pl.col("p1") < 0.1).sample(n * 4, seed=1).select("q_idx")
    fz = pl.read_parquet(EXP_DIR / f"fz_{split}.parquet")
    x = pl.concat([pl.read_parquet(f).join(pick, on="q_idx", how="semi") for f in sorted((EXP_DIR / f"s2_{split}").glob("*.parquet"))]).join(fz, on="q_idx")
    x = x.filter(pl.col("num_conflict") == 1).head(n)
    X2 = x.select(S2).to_numpy(); q2 = np.mean([m.predict(X2) for m in m2], 0)
    G = group_features(pl.read_parquet(EXP_DIR / f"sim_E07_pred.parquet").head(0), load_strings(split).head(0)) if False else None
    return x, q2
res = {}
for split, pf in [("train", EXP_DIR / "E07_crossfit_train_pred.parquet"), ("test", EXP_DIR / "SUBMISSION_E09" / "test_stage3_pred.parquet")]:
    x, q2 = rows(split, pf)
    # h_ features need full-split group computation; approximate with the saved stage-3 input join
    full_q2 = pl.read_parquet(pf).rename({"prob": "prob3"})
    res[split] = (x, q2)
    print(split, "contested rows:", x.height, " stage-2 mean prob:", round(float(q2.mean()), 4),
          " share stage2>=0.65:", round(float((q2 >= 0.65).mean()), 4), flush=True)
# contributions of STAGE 2 (the first model that sees group feats) on contested rows
for split in ("train", "test"):
    x, _ = res[split]
    C = np.mean([m.predict(x.select(S2).to_numpy(), pred_contrib=True) for m in m2], 0)
    mc = C[:, :-1].mean(0)
    top = sorted(zip(S2, mc), key=lambda t: -abs(t[1]))[:12]
    print(f"\n{split}: mean contribution (log-odds) of top features, contested rows")
    for k, v in top:
        print(f"   {k:18s} {v:+.3f}   mean value {x[k].mean():.3f}")
