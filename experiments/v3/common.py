"""Small, bounded utilities for the leaderboard-gap experiments."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.stdout.reconfigure(encoding="utf-8")
import polars as pl
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "business_entity_resolution" / "src"))
from config import CACHE_DIR

RUN = CACHE_DIR / "runs" / "v3"
RUN.mkdir(parents=True, exist_ok=True)
FIRST = CACHE_DIR / "runs" / "model_C3_direct_x_more_data"
BASELINE = CACHE_DIR / "runs" / "model_C4_stack_on_C3_direct_x_more_data"


def role_gate():
    """Fixed before role-model outcomes; C3 is out-of-sample for eligible fit q."""
    return pl.col("p1").is_between(.001, .999, closed="both")


def role_dir(split):
    return RUN / f"role_gate_{split}"


def memory(required_gb=2.0):
    gc.collect()
    available = psutil.virtual_memory().available / 2**30
    if available < required_gb:
        raise RuntimeError(f"Safe stop: only {available:.2f} GiB free; need {required_gb:.2f}. Completed batches are reusable.")
    return f"RAM {psutil.Process().memory_info().rss / 2**30:.2f} GiB; free {available:.2f} GiB"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    data.write_parquet(tmp)
    assert pl.scan_parquet(tmp).select(pl.len()).collect().item() == data.height
    os.replace(tmp, path)


def s1_table():
    return pl.read_parquet(CACHE_DIR / "runs/v2_data/split_s1.parquet", columns=["s1_idx", "country", "fold"]).join(
        pl.read_parquet(CACHE_DIR / "v2train_source1_norm.parquet", columns=["name_core"])
        .with_row_index("s1_idx"), on="s1_idx")


def qtruth():
    import numpy as np
    qf = pl.read_parquet(CACHE_DIR / "runs/v2_data/qf.parquet", columns=["q_idx", "true_s1"])
    out = np.full(int(qf["q_idx"].max()) + 1, -1, dtype=np.int32)
    out[qf["q_idx"].to_numpy()] = qf["true_s1"].fill_null(-1).cast(pl.Int32).to_numpy()
    return out
