"""Regularized best-pair classifiers with family-grouped internal early stopping.

No DEV/CONF query is fitted. The first model's fit/early-stop queries are excluded.
Usage: python experiments/v3/train.py direct [--country US|India]
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

from common import CACHE_DIR, RUN, atomic_json, atomic_parquet, memory, pl, s1_table, sha, role_gate, role_dir
import lightgbm as lgb
import numpy as np
from features import FEATURES

XCOLS = ["nm_wcov_q", "nm_wcov_s", "nm_qonly_idf_max", "nm_sonly_idf_max", "nm_shared_idf",
         "ad_wcov_q", "ad_qonly_idf_max", "ad_qonly_n_rare", "ad_shared_idf", "prem_eq", "prem_in_s",
         "s_prem_in_q", "q_nums_subset", "prem_close", "raw_name_eq", "raw_addr_eq"]
CONTEXT = {"cos_name", "cos_addr", "cos_comb", "rank", "n_cands", "comb_max", "comb_gap", "comb_gap_next",
           "name_gap", "addr_gap", "s1_name_dup", "s1_rank1_deg", "n_name_ties", "q_is_s3", "q_indic"}
CONTEXT.update(c for c in FEATURES if c.startswith("m_"))
DIRECT = [c for c in FEATURES if c not in CONTEXT] + [
    "nm_wcov_q", "nm_wcov_s", "ad_wcov_q", "prem_in_s", "s_prem_in_q", "q_nums_subset", "raw_name_eq", "raw_addr_eq"]


def train(mode, country=None):
    dest = RUN / (mode + ("_" + country if country else ""))
    dest.mkdir(exist_ok=True)
    files = sorted((RUN / "rows_v2train").glob("*.parquet"))
    base_state = json.loads((RUN / "rows_v2train/manifest.json").read_text())
    if not base_state.get("complete"):
        raise RuntimeError("Best-row cache is incomplete")
    if mode == "direct":
        feats = DIRECT
    elif mode == "roles":
        from direct_features import FEATURES as ROLE_FEATURES
        feats = [c for c in FEATURES if c != "s1_rank1_deg"] + XCOLS + ["p1", "p2", "p3", "margin12"] + ROLE_FEATURES
        role_state = json.loads((role_dir("v2train") / "manifest.json").read_text())
        if not role_state.get("complete"):
            raise RuntimeError("Role feature cache is incomplete")
    else:
        raise ValueError(mode)
    cfg = dict(mode=mode, country=country, features=feats, seed=9021,
               fraction=(1.0 if mode == "roles" else 0.50) if country is None else 0.30,
               group="country/name_core(no spaces), hash 9021 mod10==0 early-stop",
               excludes="C3 fit/ES queries and all DEV/CONF queries", num_leaves=127 if mode == "roles" else 63, min_data_in_leaf=250,
               lambda_l2=10, num_threads=4, max_rounds=2200, early_stopping=80)
    if mode == "roles":
        cfg["gate"] = "C3 p1 in [.001,.999]; otherwise use frozen C4 probability"
        cfg["candidate_train_only"] = True
    if (dest / "fit.json").exists():
        prior = json.loads((dest / "fit.json").read_text())
        if prior["config"] != cfg or prior["code_sha"] != sha(__file__):
            raise RuntimeError("Fit code/config changed; use a new run name")
        print(f"Fit cached: {dest}", flush=True)
        return
    input_fingerprints = {"rows": sha(RUN / "rows_v2train/manifest.json"),
                          "query_labels": sha(CACHE_DIR / "runs/v2_data/qf.parquet"),
                          "folds": sha(CACHE_DIR / "runs/v2_data/split_s1.parquet"),
                          "catalog": sha(CACHE_DIR / "v2train_source1_norm.parquet")}
    if mode == "roles":
        input_fingerprints["roles"] = sha(role_dir("v2train") / "manifest.json")
    snapshot = dest / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in ("train.py", "common.py", "direct_features.py"):
        shutil.copy2(Path(__file__).parent / filename, snapshot / filename)
    memory(4)
    catalog = s1_table()
    train_targets = catalog.filter(pl.col("fold") == "TRAIN")["s1_idx"]
    groups = catalog.select("s1_idx", pl.struct("country", pl.col("name_core").str.replace_all(" ", ""))
                             .hash(9021).mod(10).eq(0).alias("is_es"))
    del catalog
    parts_a, parts_b, labels_a, labels_b = [], [], [], []
    used_first = ((pl.col("q_idx").hash(42) % 1000) < 250) | ((pl.col("q_idx").hash(43) % 1000) < 15)
    pool = (pl.col("fold") == "TRAIN") & ~used_first
    for f in files:
        assert sha(f) == base_state["files"][f.name]["sha"], f"Changed base cache: {f}"
        d = pl.read_parquet(f).filter(pool)
        if mode == "roles":
            d = d.filter(role_gate() & pl.col("s1_idx").is_in(train_targets.implode()))
        if country:
            d = d.filter(pl.col("country") == country)
        d = d.with_columns(pl.coalesce("true_s1", "s1_idx").alias("group_idx"))
        d = d.join(groups.rename({"s1_idx": "group_idx"}), on="group_idx", validate="m:1")
        d = d.with_columns((pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"))
        keep = (pl.col("q_idx").hash(9022) % 1000) < int(cfg["fraction"] * 1000)
        a = d.filter(~pl.col("is_es") & keep)
        b = d.filter(pl.col("is_es"))
        if mode == "roles":
            role_file = role_dir("v2train") / f.name
            assert sha(role_file) == role_state["files"][f.name]["sha"], f"Changed role cache: {role_file}"
            extras = pl.read_parquet(role_file)
            a = a.join(extras, on=["q_idx", "s1_idx"], validate="1:1")
            b = b.join(extras, on=["q_idx", "s1_idx"], validate="1:1")
        parts_a.append(a.select(feats).to_numpy().astype(np.float32))
        labels_a.append(a["y"].to_numpy().astype(np.float32))
        parts_b.append(b.select(feats).to_numpy().astype(np.float32))
        labels_b.append(b["y"].to_numpy().astype(np.float32))
        del d, a, b
    del groups
    memory(3)
    Xa, ya, Xe, ye = map(np.concatenate, (parts_a, labels_a, parts_b, labels_b))
    del parts_a, labels_a, parts_b, labels_b
    print(f"{mode}/{country}: train={len(ya):,}, es={len(ye):,}, features={len(feats)}; {memory()}", flush=True)
    params = dict(objective="binary", learning_rate=0.04, num_leaves=cfg["num_leaves"], min_data_in_leaf=250,
                  feature_fraction=0.9, bagging_fraction=0.85, bagging_freq=1, lambda_l2=10,
                  max_bin=127, num_threads=4, seed=9021, verbosity=-1, force_col_wise=True, deterministic=True)
    # Construct with the same thread budget (construction otherwise uses all CPUs).
    dtr = lgb.Dataset(Xa, ya, feature_name=feats, params=params, free_raw_data=True).construct()
    dva = lgb.Dataset(Xe, ye, reference=dtr, params=params, free_raw_data=True).construct()
    sizes = {"fit_rows": len(ya), "early_stop_rows": len(ye)}
    del Xa, ya, Xe, ye
    memory(2.5)
    begin = time.time()
    def report_and_check(env):
        if (env.iteration + 1) % 200 == 0:
            print(f"iteration {env.iteration + 1}; {memory(2)}", flush=True)
    model = lgb.train(params, dtr, cfg["max_rounds"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg["early_stopping"]), lgb.log_evaluation(200), report_and_check])
    tmp = dest / "model.tmp"
    model.save_model(str(tmp))
    tmp.replace(dest / "model.txt")
    atomic_json(dest / "fit.json", {"config": cfg, "code_sha": sha(__file__), "model_sha": sha(dest / "model.txt"),
                "input_fingerprints": input_fingerprints, "params": params,
                "versions": {"lightgbm": lgb.__version__, "polars": pl.__version__, "numpy": np.__version__},
                **sizes, "iterations": model.best_iteration, "fit_seconds": time.time()-begin,
                "importance": sorted(zip(feats, map(float, model.feature_importance("gain"))), key=lambda x:-x[1])})
    print(f"Fit finished: {dest}, {model.best_iteration} trees; {memory()}", flush=True)


def predict(mode, split, country=None):
    dest = RUN / (mode + ("_" + country if country else ""))
    fit = json.loads((dest / "fit.json").read_text())
    feats = fit["config"]["features"]
    model = lgb.Booster(model_file=str(dest / "model.txt"))
    out = dest / f"pred_{split}.parquet"
    row_manifest = RUN / f"rows_{split}/manifest.json"
    base_state = json.loads(row_manifest.read_text())
    assert base_state["complete"], "Base rows are incomplete"
    signature = {"model": sha(dest / "model.txt"), "features": feats, "rows": sha(row_manifest),
                 "code": sha(__file__), "common": sha(Path(__file__).parent / "common.py")}
    if mode == "roles":
        role_manifest = role_dir(split) / "manifest.json"
        role_state = json.loads(role_manifest.read_text())
        assert role_state["complete"], "Role rows are incomplete"
        signature["roles"] = sha(role_manifest)
    partdir = dest / f"score_parts_{split}"
    partdir.mkdir(exist_ok=True)
    state_path = partdir / "manifest.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"inputs": signature, "files": {}}
    if state["inputs"] != signature:
        raise RuntimeError("Scoring inputs changed; preserve this run and use a new score directory")
    parts = []
    for f in sorted((RUN / f"rows_{split}").glob("*.parquet")):
        assert sha(f) == base_state["files"][f.name]["sha"], f"Changed base cache: {f}"
        if mode == "roles":
            assert sha(role_dir(split) / f.name) == role_state["files"][f.name]["sha"], f"Changed role cache: {f.name}"
        checkpoint = partdir / f.name
        if checkpoint.exists() and state["files"].get(f.name) == sha(checkpoint):
            parts.append(pl.read_parquet(checkpoint))
            continue
        memory(2.2)
        d = pl.read_parquet(f)
        if mode == "roles":
            selected = d.filter(role_gate()).join(pl.read_parquet(role_dir(split) / f.name), on=["q_idx", "s1_idx"], validate="1:1")
            assert selected.height == d.filter(role_gate()).height
            p = d["baseline_p"].to_numpy().copy()
            gate_mask = d.select(role_gate()).to_series().to_numpy()
            # Polars joins may reorder rows: align scored rows by q_idx explicitly.
            if selected.height:
                scored = selected.select("q_idx").with_columns(pl.Series("role_p", model.predict(selected.select(feats).to_numpy(), num_threads=4).astype(np.float32)))
                aligned = d.select("q_idx").join(scored, on="q_idx", how="left", maintain_order="left")
                p[gate_mask] = aligned.filter(pl.Series(gate_mask))["role_p"].to_numpy()
        else:
            p = model.predict(d.select(feats).to_numpy(), num_threads=4).astype(np.float32)
        part = d.select("q_idx", "s1_idx").with_columns(pl.Series("p", p))
        atomic_parquet(checkpoint, part)
        state["files"][f.name] = sha(checkpoint)
        atomic_json(state_path, state)
        parts.append(part)
        print(f"Scored {f.name}: {sum(x.height for x in parts):,} rows; {memory()}", flush=True)
    result = pl.concat(parts)
    assert result["q_idx"].n_unique() == result.height
    atomic_parquet(out, result)
    print(f"Scored {split}: {result.height:,}; {memory()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["direct", "roles"])
    ap.add_argument("--country", choices=["US", "India"])
    ap.add_argument("--score", choices=["v2train", "v2test", "v2stress"])
    args = ap.parse_args()
    if args.score:
        predict(args.mode, args.score, args.country)
    else:
        train(args.mode, args.country)
