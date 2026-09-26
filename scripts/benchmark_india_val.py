import gc
import hashlib
import logging
import multiprocessing as mp
import os
import sys
import sys
from pathlib import Path

# Add project root to path so we can import src from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from pathlib import Path
import numpy as np
import polars as pl
import psutil

# Add project root to sys.path
sys.path.insert(0, "/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026")

from src.blocking import MultiLayerBlocker
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IndiaValBenchmark")

def sha256_parquet(df: pl.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def monitor_cpu(interval=0.5, stop_event=None, cpu_log=None):
    while not stop_event.is_set():
        usage = psutil.cpu_percent(interval=interval, percpu=True)
        if cpu_log is not None:
            cpu_log.append(usage)

if __name__ == "__main__":
    logger.info("=== Starting 40,021-query India Validation Benchmark ===")

    from src.config import config
    cleaned_india = config.CLEANED_DIR / "India"
    val_india = config.VAL_SPLIT_DIR / "India"


    s1_files = list(val_india.glob("val_s1_part_*.parquet"))
    target_files = list(cleaned_india.glob("train_source2_part_*.parquet")) + list(cleaned_india.glob("train_source3_part_*.parquet"))

    logger.info("Reading parquet files...")
    s1_df = pl.concat([pl.read_parquet(f) for f in s1_files])
    target_df = pl.concat([pl.read_parquet(f) for f in target_files])

    logger.info("Loaded India: %d S1 val queries, %d Target records", len(s1_df), len(target_df))

    blocker_bench = MultiLayerBlocker()
    true_matches = blocker_bench._load_validation_truth(set(s1_df["entity_id"].to_list()))
    logger.info("Loaded validation truth for %d queries", len(true_matches))

    # --- 2. Parallel Run (Full worker count = 10) ---
    logger.info(">>> Running Parallel Fork COW (num_workers=10)...")
    gc.collect()
    proc = psutil.Process()
    ram_before_par = proc.memory_info().rss / 1e6

    # Setup CPU monitoring
    import threading
    stop_event = threading.Event()
    cpu_measurements = []
    monitor_thread = threading.Thread(target=monitor_cpu, kwargs={"interval": 0.5, "stop_event": stop_event, "cpu_log": cpu_measurements})
    monitor_thread.daemon = True
    monitor_thread.start()

    blocker_par = MultiLayerBlocker(num_workers=10)
    t0_par = time.perf_counter()
    df_par, stats_par = blocker_par.block_country_partition("India", s1_df, target_df, true_matches=true_matches)
    total_time_par = time.perf_counter() - t0_par
    ram_peak_par = blocker_par.peak_memory_mb

    stop_event.set()
    monitor_thread.join(timeout=2.0)

    logger.info("Parallel Finished in %.2fs (Peak RAM: %.1f MB)", total_time_par, ram_peak_par)
    logger.info("Parallel Stats: %s", stats_par["diagnostics"]["survival"])

    # Average CPU utilization per core during parallel run
    if cpu_measurements:
        avg_per_core = np.mean(cpu_measurements, axis=0)
        overall_avg = np.mean(avg_per_core)
        logger.info("CPU Utilization during parallel execution:")
        for idx, u in enumerate(avg_per_core):
            logger.info("  Core %d: %.1f%%", idx, u)
        logger.info("  Overall Average CPU: %.1f%% across %d cores", overall_avg, len(avg_per_core))

    logger.info("=== SUMMARY REPORT ===")
    logger.info("Queries:                  %d", len(s1_df))
    logger.info("Target records:           %d", len(target_df))
    logger.info("Workers:                  10")
    logger.info("Parallel Total Time:      %.2fs", total_time_par)
    logger.info("Peak RAM Parallel:        %.1f MB", ram_peak_par)
