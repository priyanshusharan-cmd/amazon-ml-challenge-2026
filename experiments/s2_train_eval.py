"""Stage-2 experiment: re-decide each record's best pair using stage-1 outputs + group features.
Trains on fold-A records that were NOT used to fit stage 1 (out-of-sample stage-1 probabilities),
evaluates macro F0.5 on fold B with the same assignment rule as the baseline.
usage: python s2_train_eval.py <exp_name> [nogroup]"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "../business_entity_resolution/src")
from harness import EXP_DIR, Ctx, best_so_far, log, mem  # noqa: E402
from features import FEATURES  # noqa: E402
from train import SEED, TRAIN_QUERY_FRACTION, query_folds  # noqa: E402

name = sys.argv[1]
nogroup = "nogroup" in sys.argv[2:]
extras = [a for a in sys.argv[2:] if a.endswith(".parquet")]
EXTRA = pl.concat([pl.read_parquet(EXP_DIR / e) for e in extras], how="horizontal") if extras else None
if EXTRA is not None and len(extras) > 1:
    raise SystemExit("one extra file at a time")
EXTRA_COLS = [c for c in EXTRA.columns if c != "q_idx"] if EXTRA is not None else []
GROUP = ["g_all", "g_conf", "g_psum", "sup_nums_all", "sup_nums_conf", "sup_num1_all", "sup_num1_conf",
         "sup_name_all", "sup_name_conf", "sup_both_all", "sup_both_conf", "sup_nums_frac", "sup_name_frac"]
S2 = FEATURES + ["p1", "p2", "p3", "margin12"] + ([] if nogroup else GROUP) + EXTRA_COLS
TRAIN_ROWS = 3_000_000

ctx = Ctx()
s1full = ctx.s1.select("s1_idx", "is_val")
qf = query_folds(ctx.lab, s1full)  # q_idx, true_s1, q_is_val
stage1_train = (pl.col("q_idx").hash(SEED) % 1000) < int(TRAIN_QUERY_FRACTION * 1000)
es_mask = (pl.col("q_idx").hash(SEED + 7) % 100) < 5
files = sorted((EXP_DIR / "s2_train").glob("part_*.parquet"))
Xtr, ytr, Xes, yes = [], [], [], []
n_pool = 0
for f in files:
    x = pl.read_parquet(f)
    if EXTRA is not None:
        x = x.join(EXTRA, on="q_idx", how="left")
    x = x.join(qf, on="q_idx", how="left").with_columns(
        (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
    a = x.filter(~pl.col("q_is_val").fill_null(True) & ~stage1_train)
    n_pool += a.height
    tr = a.filter(~es_mask).sample(fraction=min(1.0, TRAIN_ROWS / 7_500_000), seed=SEED)
    es = a.filter(es_mask).sample(fraction=0.3, seed=SEED)
    Xtr.append(tr.select(S2).to_numpy().astype(np.float32)); ytr.append(tr["y"].to_numpy())
    Xes.append(es.select(S2).to_numpy().astype(np.float32)); yes.append(es["y"].to_numpy())
Xtr, ytr, Xes, yes = map(np.concatenate, (Xtr, ytr, Xes, yes))
print(f"pool {n_pool:,} train {len(ytr):,} (pos {ytr.mean():.3f}) es {len(yes):,}", mem(), flush=True)
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=16, seed=SEED, verbose=-1)
m = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=S2), 3000, valid_sets=[lgb.Dataset(Xes, yes)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
del Xtr, ytr, Xes, yes
m.save_model(str(EXP_DIR / f"{name}.txt"))
imp = sorted(zip(S2, m.feature_importance("gain")), key=lambda t: -t[1])[:15]
print("top:", [(k, round(v)) for k, v in imp], flush=True)

preds = []
for f in files:
    x = pl.read_parquet(f)
    if EXTRA is not None:
        x = x.join(EXTRA, on="q_idx", how="left")
    preds.append(x.select("q_idx", "s1_idx", "p1").with_columns(
        pl.Series("q2", m.predict(x.select(S2).to_numpy(), num_threads=16).astype(np.float32))))
P = pl.concat(preds)
P.write_parquet(EXP_DIR / f"{name}_train_pred.parquet")
best = best_so_far()["F"]
res = []
for t in np.round(np.arange(0.40, 0.91, 0.025), 3):
    r = ctx.f05(P.filter(pl.col("q2") >= t))
    res.append((t, r))
    print(f"t={t:.3f} F={r['F']:.5f} P={r['P']:.5f} R={r['R']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}", flush=True)
t_best, r_best = max(res, key=lambda z: z[1]["F"])
log(name, {**r_best, "threshold": float(t_best)}, note=("no group feats" if nogroup else "stage2 + group feats") + (f" + {extras}" if extras else ""), best=best)
json.dump({"threshold": float(t_best), **{k: v for k, v in r_best.items() if isinstance(v, float)}},
          open(EXP_DIR / f"{name}_result.json", "w"), indent=1)
