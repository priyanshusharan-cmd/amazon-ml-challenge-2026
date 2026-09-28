"""E04: iterate the group features.
 stage 2 (same recipe as E02) fit on hash-half H0 of the out-of-sample fold-A pool -> q2 for every record
 group features recomputed from q2 (prefix h_)
 stage 3 fit on the other half H1 with stage-2 features + h_ features + q2; evaluated on fold B."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "../business_entity_resolution/src")
from harness import EXP_DIR, Ctx, best_so_far, log, mem  # noqa: E402
from features import FEATURES  # noqa: E402
from gfeat import GROUP_BASE, group_features, load_strings  # noqa: E402
from train import SEED, TRAIN_QUERY_FRACTION, query_folds  # noqa: E402

name = sys.argv[1]
extras = [a for a in sys.argv[2:] if a.endswith(".parquet")]
EXTRA = pl.read_parquet(EXP_DIR / extras[0]) if extras else None
EXTRA_COLS = [c for c in EXTRA.columns if c != "q_idx"] if EXTRA is not None else []
S2 = FEATURES + ["p1", "p2", "p3", "margin12"] + GROUP_BASE + EXTRA_COLS
H = ["h_" + c for c in GROUP_BASE]
S3 = S2 + H + ["q2"]
ROWS = 3_000_000
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=16, seed=SEED, verbose=-1)

ctx = Ctx()
qf = query_folds(ctx.lab, ctx.s1.select("s1_idx", "is_val"))
stage1_train = (pl.col("q_idx").hash(SEED) % 1000) < int(TRAIN_QUERY_FRACTION * 1000)
pool = ~pl.col("q_is_val").fill_null(True) & ~stage1_train
half = (pl.col("q_idx").hash(SEED + 11) % 2)
es_mask = (pl.col("q_idx").hash(SEED + 7) % 100) < 5
files = sorted((EXP_DIR / "s2_train").glob("part_*.parquet"))


def load(cols, row_filter, frac, extra=None):
    Xa, ya, Xe, ye = [], [], [], []
    for f in files:
        x = pl.read_parquet(f)
        if EXTRA is not None:
            x = x.join(EXTRA, on="q_idx", how="left")
        x = x.join(qf, on="q_idx", how="left").with_columns(
            (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
        if extra is not None:
            x = x.join(extra, on="q_idx", how="left")
        a = x.filter(pool & row_filter)
        tr = a.filter(~es_mask).sample(fraction=frac, seed=SEED)
        es = a.filter(es_mask).sample(fraction=0.3, seed=SEED)
        Xa.append(tr.select(cols).to_numpy().astype(np.float32)); ya.append(tr["y"].to_numpy())
        Xe.append(es.select(cols).to_numpy().astype(np.float32)); ye.append(es["y"].to_numpy())
    return map(np.concatenate, (Xa, ya, Xe, ye))


def fit(cols, Xa, ya, Xe, ye):
    return lgb.train(PARAMS, lgb.Dataset(Xa, ya, feature_name=cols), 3000, valid_sets=[lgb.Dataset(Xe, ye)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])


def predict_all(model, cols, extra=None):
    out = []
    for f in files:
        x = pl.read_parquet(f)
        if EXTRA is not None:
            x = x.join(EXTRA, on="q_idx", how="left")
        if extra is not None:
            x = x.join(extra, on="q_idx", how="left")
        out.append(x.select("q_idx", "s1_idx").with_columns(
            pl.Series("prob", model.predict(x.select(cols).to_numpy(), num_threads=16).astype(np.float32))))
    return pl.concat(out)


# ---- stage 2 on H0
Xa, ya, Xe, ye = load(S2, half == 0, ROWS / 3_700_000)
print("stage2 rows", len(ya), mem(), flush=True)
m2 = fit(S2, Xa, ya, Xe, ye); del Xa, ya, Xe, ye
m2.save_model(str(EXP_DIR / f"{name}_stage2.txt"))
q2 = predict_all(m2, S2)
strings = load_strings("train")
G = group_features(q2, strings, prefix="h_").join(q2.select("q_idx", pl.col("prob").alias("q2")), on="q_idx")
del strings
print("h features", G.shape, mem(), flush=True)
# ---- stage 3 on H1
Xa, ya, Xe, ye = load(S3, half == 1, ROWS / 3_700_000, extra=G)
print("stage3 rows", len(ya), mem(), flush=True)
m3 = fit(S3, Xa, ya, Xe, ye); del Xa, ya, Xe, ye
m3.save_model(str(EXP_DIR / f"{name}_stage3.txt"))
imp = sorted(zip(S3, m3.feature_importance("gain")), key=lambda t: -t[1])[:12]
print("top:", [(k, round(v)) for k, v in imp], flush=True)
P = predict_all(m3, S3, extra=G)
P.write_parquet(EXP_DIR / f"{name}_train_pred.parquet")
best = best_so_far()["F"]
res = []
for t in np.round(np.arange(0.45, 0.851, 0.025), 3):
    r = ctx.f05(P.filter(pl.col("prob") >= t)); res.append((t, r))
    print(f"t={t:.3f} F={r['F']:.5f} P={r['P']:.5f} R={r['R']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}", flush=True)
# also report stage-2(H0) alone for reference
r2 = max((ctx.f05(q2.filter(pl.col("prob") >= t))["F"], t) for t in (0.6, 0.65, 0.675, 0.7))
print("stage-2 (H0 fit) best F:", r2, flush=True)
t_best, r_best = max(res, key=lambda z: z[1]["F"])
log(name, {**r_best, "threshold": float(t_best)}, note="stage3: group feats recomputed from stage-2 probs" + (f" + {extras}" if extras else ""), best=best)
json.dump({"threshold": float(t_best), **{k: v for k, v in r_best.items() if isinstance(v, float)}},
          open(EXP_DIR / f"{name}_result.json", "w"), indent=1)
