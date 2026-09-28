"""Experiment harness: fast, integer-id macro F0.5 on the validation fold (identical definition
to business_entity_resolution/src/metrics.py), plus a two-half split for stability checks."""
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "business_entity_resolution" / "src"))
from config import cache_path  # noqa: E402
from train import label_table  # noqa: E402

import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import psutil  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
MIN_FREE_MB = 1500  # below this the experiment kills itself rather than starve the machine


def _watchdog():
    me = psutil.Process()
    while True:
        avail = psutil.virtual_memory().available / 2**20
        if avail < MIN_FREE_MB:
            print(f"\n!!! WATCHDOG: system free memory {avail:.0f} MB (process RSS {me.memory_info().rss/2**20:.0f} MB) "
                  f"- aborting to protect the machine", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=_watchdog, daemon=True).start()


def mem():
    return f"rss={psutil.Process().memory_info().rss/2**30:.2f}GB free={psutil.virtual_memory().available/2**30:.2f}GB"


EXP_DIR = cache_path("experiments")
BASELINE = EXP_DIR / "BASELINE_BEST"
LEDGER = ROOT / "experiments" / "ledger.jsonl"
BETA2 = 0.25


class Ctx:
    """Validation context with integer ids only."""

    def __init__(self):
        s1, q, lab, _ = label_table()
        self.s1 = s1.select("s1_idx", "is_val", "country",
                            (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 2 == 0).alias("half"))
        self.lab = lab  # q_idx -> true_s1
        self.val = self.s1.filter("is_val").select("s1_idx", "country", "half")
        self.n_true = (lab.group_by("true_s1").agg(pl.len().alias("n_true"))
                       .rename({"true_s1": "s1_idx"}))

    def f05(self, pred: pl.DataFrame, detail=False):
        """pred: (q_idx, s1_idx) assignments (each q at most once). Returns dict of macro scores."""
        p = pred.select("q_idx", "s1_idx").join(self.val.select("s1_idx"), on="s1_idx", how="semi")
        p = p.join(self.lab, on="q_idx", how="left").with_columns(
            (pl.col("true_s1") == pl.col("s1_idx")).fill_null(False).alias("hit"))
        agg = p.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("hit").sum().alias("tp"))
        d = (self.val.join(agg, on="s1_idx", how="left").join(self.n_true, on="s1_idx", how="left")
             .with_columns(pl.col("n_pred", "tp", "n_true").fill_null(0)))
        d = d.with_columns(
            pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred"))
              .otherwise(pl.when(pl.col("n_true") == 0).then(1.0).otherwise(0.0)).alias("P"),
            pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true"))
              .otherwise(pl.when(pl.col("n_pred") == 0).then(1.0).otherwise(0.0)).alias("R"),
        ).with_columns(
            pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
              .when(pl.col("tp") == 0).then(0.0)
              .otherwise((1 + BETA2) * pl.col("P") * pl.col("R") / (BETA2 * pl.col("P") + pl.col("R"))).alias("F"))
        out = {"F": d["F"].mean(), "P": d["P"].mean(), "R": d["R"].mean(),
               "F_h0": d.filter(~pl.col("half"))["F"].mean(), "F_h1": d.filter(pl.col("half"))["F"].mean()}
        for c in d["country"].unique().sort().to_list():
            out[f"F_{c}"] = d.filter(pl.col("country") == c)["F"].mean()
        if detail:
            out["per_entity"] = d
        return out


def fmt(r):
    return " ".join(f"{k}={v:.5f}" for k, v in r.items() if isinstance(v, float))


def log(name, result, note="", best=None):
    rec = {"experiment": name, **{k: v for k, v in result.items() if isinstance(v, (int, float, str))}, "note": note}
    if best is not None:
        rec["delta_vs_best"] = result["F"] - best
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{name}] {fmt(result)} {'delta=%+.5f' % rec['delta_vs_best'] if best is not None else ''} {note}", flush=True)


def best_so_far():
    rows = [json.loads(line) for line in open(LEDGER)] if LEDGER.exists() else []
    promoted = [r for r in rows if r.get("promoted")]
    return max(promoted, key=lambda r: r["F"]) if promoted else None
