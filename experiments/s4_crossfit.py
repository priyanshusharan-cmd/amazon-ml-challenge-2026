"""E07: cross-fitted stage 2 (two models, each scores the other half of the pool; both averaged for all
other records), group features from out-of-fold stage-2 probabilities, stage 3 trained on the whole pool.
usage: python s4_crossfit.py <name> fz_train.parquet [rows3=5000000] [leaves=127]"""
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

name, extra_file = sys.argv[1], sys.argv[2]
ROWS2 = 3_000_000
ROWS3 = int(sys.argv[3]) if len(sys.argv) > 3 else 5_000_000
LEAVES = int(sys.argv[4]) if len(sys.argv) > 4 else 127
EXTRA = pl.read_parquet(EXP_DIR / extra_file)
EXTRA_COLS = [c for c in EXTRA.columns if c != "q_idx"]
S2 = FEATURES + ["p1", "p2", "p3", "margin12"] + GROUP_BASE + EXTRA_COLS
S3 = S2 + ["h_" + c for c in GROUP_BASE] + ["q2"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=LEAVES, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=16, seed=SEED, verbose=-1)

ctx = Ctx()
qf = query_folds(ctx.lab, ctx.s1.select("s1_idx", "is_val"))
qf = qf.with_columns(
    (~pl.col("q_is_val").fill_null(True) & ~((pl.col("q_idx").hash(SEED) % 1000) < int(TRAIN_QUERY_FRACTION * 1000))).alias("pool"),
    (pl.col("q_idx").hash(SEED + 11) % 2).cast(pl.Int8).alias("half"),
    ((pl.col("q_idx").hash(SEED + 7) % 100) < 5).alias("es"))
files = sorted((EXP_DIR / "s2_train").glob("part_*.parquet"))


def frame(f, extra2=None):
    x = pl.read_parquet(f).join(EXTRA, on="q_idx", how="left").join(qf, on="q_idx", how="left").with_columns(
        (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
    return x if extra2 is None else x.join(extra2, on="q_idx", how="left")


def load(cols, cond, n_target, extra2=None):
    """Preallocated arrays (no concat copy). Returns train + early-stopping sets."""
    counts = []
    for f in files:
        x = frame(f, extra2)
        counts.append((x.filter(cond & ~pl.col("es")).height, x.filter(cond & pl.col("es")).height))
    tot_tr = sum(c[0] for c in counts); frac = min(1.0, n_target / max(tot_tr, 1))
    Xa = np.empty((int(tot_tr * frac) + len(files) + 10, len(cols)), np.float32); ya = np.empty(Xa.shape[0], bool)
    Xe = np.empty((int(sum(c[1] for c in counts) * 0.3) + len(files) + 10, len(cols)), np.float32); ye = np.empty(Xe.shape[0], bool)
    ia = ie = 0
    for f in files:
        x = frame(f, extra2)
        tr = x.filter(cond & ~pl.col("es")).sample(fraction=frac, seed=SEED)
        es = x.filter(cond & pl.col("es")).sample(fraction=0.3, seed=SEED)
        n, m = tr.height, es.height
        Xa[ia:ia + n] = tr.select(cols).to_numpy(); ya[ia:ia + n] = tr["y"].to_numpy(); ia += n
        Xe[ie:ie + m] = es.select(cols).to_numpy(); ye[ie:ie + m] = es["y"].to_numpy(); ie += m
    return Xa[:ia], ya[:ia], Xe[:ie], ye[:ie]


def fit(cols, data):
    Xa, ya, Xe, ye = data
    print(f"  fit rows={len(ya):,}", mem(), flush=True)
    return lgb.train(PARAMS, lgb.Dataset(Xa, ya, feature_name=cols, free_raw_data=True), 4000,
                     valid_sets=[lgb.Dataset(Xe, ye)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)])


# ---- stage 2, cross-fitted
m2 = [fit(S2, load(S2, pl.col("pool") & (pl.col("half") == h), ROWS2)) for h in (0, 1)]
for i, m in enumerate(m2):
    m.save_model(str(EXP_DIR / f"{name}_stage2_h{i}.txt"))
outs = []
for f in files:
    x = frame(f)
    X = x.select(S2).to_numpy()
    pa, pb = m2[0].predict(X, num_threads=16), m2[1].predict(X, num_threads=16)
    pool, half = x["pool"].fill_null(False).to_numpy(), x["half"].to_numpy()
    q2 = np.where(pool & (half == 1), pa, np.where(pool & (half == 0), pb, (pa + pb) / 2))  # out-of-fold on the pool
    outs.append(x.select("q_idx", "s1_idx").with_columns(pl.Series("prob", q2.astype(np.float32))))
q2 = pl.concat(outs)
strings = load_strings("train")
G = group_features(q2, strings, prefix="h_").join(q2.select("q_idx", pl.col("prob").alias("q2")), on="q_idx")
del strings
print("stage2 done", mem(), flush=True)
# ---- stage 3 on the whole pool
m3 = fit(S3, load(S3, pl.col("pool"), ROWS3, extra2=G))
m3.save_model(str(EXP_DIR / f"{name}_stage3.txt"))
outs = []
for f in files:
    x = frame(f, G)
    outs.append(x.select("q_idx", "s1_idx").with_columns(pl.Series("prob", m3.predict(x.select(S3).to_numpy(), num_threads=16).astype(np.float32))))
P = pl.concat(outs)
P.write_parquet(EXP_DIR / f"{name}_train_pred.parquet")
best = best_so_far()["F"]
res = []
for t in np.round(np.arange(0.5, 0.801, 0.025), 3):
    r = ctx.f05(P.filter(pl.col("prob") >= t)); res.append((t, r))
    print(f"t={t:.3f} F={r['F']:.5f} P={r['P']:.5f} R={r['R']:.5f} h0={r['F_h0']:.5f} h1={r['F_h1']:.5f}", flush=True)
t_best, r_best = max(res, key=lambda z: z[1]["F"])
log(name, {**r_best, "threshold": float(t_best)}, note=f"cross-fitted stage2 + stage3 on full pool rows3={ROWS3} leaves={LEAVES}", best=best)
json.dump({"threshold": float(t_best), **{k: v for k, v in r_best.items() if isinstance(v, float)}},
          open(EXP_DIR / f"{name}_result.json", "w"), indent=1)
