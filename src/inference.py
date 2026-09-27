import sys
import json
import logging
import os
import sqlite3
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple

import joblib
import numpy as np
import polars as pl
import psutil

from src.config import config
from src.state_manager import StateManager
from src.feature_engineering import FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("InferencePipeline")


class InferencePipeline:
    """
    Test Inference & Output Formatting Pipeline.
    Loads trained LightGBM model, scores test candidate pairs, applies optimal threshold,
    exports matching_results.tsv and candidate_pairs.tsv, and executes official validator.
    """
    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_inference(
        self,
        test_features_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        threshold_path: Optional[Path] = None,
        dry_run: bool = False,
        force: bool = False,
        chunk_size: int = 50_000,
    ) -> Dict[str, Any]:
        stage_name = "stage_6_inference"
        feat_path = Path(test_features_path or (config.FEATURES_DIR / "test_features.parquet"))
        m_path = Path(model_path or (config.MODELS_DIR / "model.pkl"))
        t_path = Path(threshold_path or (config.MODELS_DIR / "optimal_threshold.json"))
        matching_path = config.OUTPUT_DIR / "matching_results.tsv"
        candidate_path = config.OUTPUT_DIR / "candidate_pairs.tsv"

        if not dry_run and StateManager.should_skip(matching_path, force=force) and StateManager.should_skip(candidate_path, force=force):
            logger.info("Submission TSVs already exist; skipping inference (use --force to regenerate).")
            return {"status": "SKIPPED", "matching_results": str(matching_path), "candidate_pairs": str(candidate_path)}

        t_start = time.perf_counter()

        if dry_run or not feat_path.exists() or not m_path.exists():
            if not dry_run:
                missing = [str(path) for path in (feat_path, m_path) if not path.exists()]
                raise FileNotFoundError("Inference artifacts missing; refusing to emit mock predictions: " + ", ".join(missing))
            logger.info("[DRY RUN MODE] Initialized inference pipeline successfully. Skipping full dataset prediction.")
            self._create_mock_outputs_for_validation_smoke_test()
            val_res = self.run_official_validator()
            self.state_manager.mark_completed(stage_name, meta={"dry_run": True, "validator_passed": val_res})
            return {"status": "DRY_RUN_SUCCESS", "validator_passed": val_res}

        self.state_manager.require_artifact(feat_path, "Test features (run feature generation first)")
        self.state_manager.require_artifact(m_path, "Trained model")
        self.state_manager.require_artifact(t_path, "Optimized validation threshold")
        candidate_artifact = config.BLOCKED_DIR / "test_candidates.parquet"
        self.state_manager.require_artifact(candidate_artifact, "Test candidates (run test blocking first)")
        self.state_manager.require_artifact(candidate_path, "Official candidate TSV (run test blocking first)")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.state_manager.mark_in_progress(stage_name, meta={"chunk_size": chunk_size})

        # Real inference keeps one feature batch in memory; accepted pairs spill to SQLite.
        logger.info("[Inference] Loading model and scoring test features in batches...")
        model = joblib.load(m_path)
        with open(t_path, "r", encoding="utf-8") as f:
            tau = float(json.load(f).get("optimal_threshold", config.DEFAULT_THRESHOLD))

        db_path = config.OUTPUT_DIR / ".inference_matches.sqlite3"
        db_meta_path = db_path.with_name(f"{db_path.name}.meta.json")
        pipeline_fingerprint = {
            "config_hash": self.state_manager.get_config_hash(),
            "features": self.state_manager.compute_file_fingerprint(feat_path),
            "model": self.state_manager.compute_file_fingerprint(m_path),
            "threshold": self.state_manager.compute_file_fingerprint(t_path),
            "chunk_size": chunk_size,
            "threshold_value": tau,
        }
        resume = False
        if not force and db_path.is_file() and db_meta_path.is_file():
            try:
                with db_meta_path.open(encoding="utf-8") as stream:
                    resume_meta = json.load(stream)
                resume = resume_meta.get("metadata", {}).get("pipeline_fingerprint") == pipeline_fingerprint
            except (OSError, ValueError):
                resume = False
        if not resume:
            db_path.unlink(missing_ok=True)
            db_meta_path.unlink(missing_ok=True)
        db = sqlite3.connect(db_path)
        db.execute("CREATE TABLE IF NOT EXISTS predictions (source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, PRIMARY KEY (source1_entity_id, candidate_entity_id))")
        db.execute("CREATE TABLE IF NOT EXISTS completed_batches (batch_idx INTEGER PRIMARY KEY)")
        StateManager.write_artifact_metadata(db_path, 0, meta={"pipeline_fingerprint": pipeline_fingerprint, "stage": "inference_checkpoint"})
        total_pairs = 0
        for batch_idx, batch in enumerate(pl.scan_parquet(feat_path).collect_batches(chunk_size=chunk_size)):
            completed = db.execute("SELECT 1 FROM completed_batches WHERE batch_idx = ?", (batch_idx,)).fetchone()
            if completed:
                total_pairs += len(batch)
                continue
            probs = model.predict(batch.select(FEATURE_COLUMNS).to_numpy())
            accepted = [
                (s1_id, candidate_id)
                for s1_id, candidate_id, probability in zip(batch["source1_entity_id"], batch["candidate_entity_id"], probs)
                if probability >= tau
            ]
            with db:
                db.executemany("INSERT OR IGNORE INTO predictions VALUES (?, ?)", accepted)
                db.execute("INSERT INTO completed_batches VALUES (?)", (batch_idx,))
            total_pairs += len(batch)
            if (batch_idx + 1) % 5 == 0:
                saved_rows = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                completed_count = db.execute("SELECT COUNT(*) FROM completed_batches").fetchone()[0]
                StateManager.write_artifact_metadata(
                    db_path, saved_rows,
                    meta={"pipeline_fingerprint": pipeline_fingerprint, "completed_batches": completed_count, "stage": "inference_checkpoint"},
                )

        saved_rows = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        completed_count = db.execute("SELECT COUNT(*) FROM completed_batches").fetchone()[0]
        StateManager.write_artifact_metadata(
            db_path, saved_rows,
            meta={"pipeline_fingerprint": pipeline_fingerprint, "completed_batches": completed_count, "stage": "inference_checkpoint"},
        )
        self._export_matching_results_sqlite(db)
        db.close()
        with matching_path.open(encoding="utf-8") as stream:
            matching_rows = max(0, sum(1 for _ in stream) - 1)
        with candidate_path.open(encoding="utf-8") as stream:
            candidate_rows = max(0, sum(1 for _ in stream) - 1)
        self.state_manager.record_artifact("matching_results_tsv", matching_path, matching_rows, meta={"threshold": tau})
        self.state_manager.record_artifact("candidate_pairs_tsv", candidate_path, candidate_rows, meta={"source": str(candidate_artifact)})
        db_path.unlink(missing_ok=True)
        db_meta_path.unlink(missing_ok=True)
        val_success = self.run_official_validator()

        elapsed = time.perf_counter() - t_start
        self.state_manager.mark_completed(stage_name, meta={"total_pairs": total_pairs, "time_sec": elapsed, "validator_success": val_success})

        return {"status": "INFERENCE_SUCCESS", "total_pairs": total_pairs, "time_sec": elapsed, "validator_passed": val_success}

    def _export_matching_results_sqlite(self, db: sqlite3.Connection) -> None:
        """Write every test query, joining matches from disk without an in-memory global map."""
        out_tsv = config.OUTPUT_DIR / "matching_results.tsv"
        test_s1_files = sorted(config.CLEANED_DIR.rglob("test_source1_part_*.parquet"))
        tmp_tsv = out_tsv.with_name("matching_results.tsv.tmp")
        with tmp_tsv.open("w", encoding="utf-8") as stream:
            stream.write("source1_entity_id\tmatched_entity_ids\n")
            if test_s1_files:
                for path in test_s1_files:
                    for batch in pl.scan_parquet(path).select("entity_id").collect_batches(chunk_size=100_000):
                        for source_id in batch["entity_id"]:
                            rows = db.execute(
                                "SELECT candidate_entity_id FROM predictions WHERE source1_entity_id = ? ORDER BY candidate_entity_id",
                                (source_id,),
                            )
                            stream.write(f"{source_id}\t{','.join(row[0] for row in rows)}\n")
            else:
                for (source_id,) in db.execute("SELECT DISTINCT source1_entity_id FROM predictions ORDER BY source1_entity_id"):
                    rows = db.execute(
                        "SELECT candidate_entity_id FROM predictions WHERE source1_entity_id = ? ORDER BY candidate_entity_id",
                        (source_id,),
                    )
                    stream.write(f"{source_id}\t{','.join(row[0] for row in rows)}\n")
        os.replace(tmp_tsv, out_tsv)
        logger.info("Exported streaming matching results to %s", out_tsv)

    def _export_matching_results_tsv(self, match_dict: Dict[str, List[str]]):
        """Generates official output/matching_results.tsv."""
        out_tsv = config.OUTPUT_DIR / "matching_results.tsv"
        
        # Load all test S1 IDs to guarantee exact 1-to-1 entity row presence
        test_s1_files = list(config.CLEANED_DIR.rglob("test_source1_part_*.parquet"))
        all_s1_ids = []
        if test_s1_files:
            for f in test_s1_files:
                all_s1_ids.extend(pl.read_parquet(f)["entity_id"].to_list())
        else:
            all_s1_ids = list(match_dict.keys())

        tmp_tsv = out_tsv.with_name("matching_results.tsv.tmp")
        with open(tmp_tsv, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in all_s1_ids:
                m_list = sorted(match_dict.get(s1_id, []))
                m_str = ",".join(m_list)
                f.write(f"{s1_id}\t{m_str}\n")
        os.replace(tmp_tsv, out_tsv)
        logger.info("Exported official matching results to %s (%d rows)", out_tsv, len(all_s1_ids))

    def _create_mock_outputs_for_validation_smoke_test(self):
        """Creates valid minimal TSVs so validator smoke-test passes cleanly."""
        test_s1_file = config.DATASET_DIR / "test" / "test_source1.tsv"
        if not test_s1_file.exists():
            return

        logger.info("[Smoke Test] Creating formatting-compliant TSV mock outputs...")
        s1_df = pl.scan_csv(test_s1_file, separator="\t").select("entity_id").collect()
        sample_s1 = s1_df["entity_id"].to_list()

        m_out = config.OUTPUT_DIR / "matching_results.tsv"
        c_out = config.OUTPUT_DIR / "candidate_pairs.tsv"

        with open(m_out, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1 in sample_s1:
                f.write(f"{s1}\t\n")

        with open(c_out, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1 in sample_s1:
                f.write(f"{s1}\t\n")

    def run_official_validator(self) -> bool:
        """Executes official validate_submission.py script."""
        val_script = config.VALIDATOR_SCRIPT
        m_tsv = config.OUTPUT_DIR / "matching_results.tsv"
        c_tsv = config.OUTPUT_DIR / "candidate_pairs.tsv"
        t_dir = config.DATASET_DIR / "test"

        if not val_script.exists():
            logger.warning("[Validator] Script not found at %s. Skipping.", val_script)
            return False

        logger.info("[Official Validator] Invoking validate_submission.py...")
        cmd = [
            sys.executable, str(val_script),
            "--matching", str(m_tsv),
            "--candidate", str(c_tsv),
            "--test-dir", str(t_dir)
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("[Official Validator] PASS — Output files are fully compliant and safe for submission!")
            return True
        else:
            logger.error("[Official Validator] FAIL — Issues detected:\n%s", res.stdout.strip() or res.stderr.strip())
            return False

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Test Inference Pipeline")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if they already exist")
    parser.add_argument("--dry-run", action="store_true", help="Generate mock outputs for validator testing")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Candidate processing chunk size (default: 50000)")
    args = parser.parse_args()
    InferencePipeline().run_inference(dry_run=args.dry_run, force=args.force, chunk_size=args.chunk_size)

if __name__ == "__main__":
    main()
