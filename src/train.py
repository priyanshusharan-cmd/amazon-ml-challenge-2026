import gc
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil

from src.config import config
from src.state_manager import StateManager
from src.feature_engineering import FEATURE_COLUMNS
from src.evaluate import ThresholdOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TrainModule")


class EntityMatcherTrainer:
    """
    LightGBM GBDT Entity Matching Trainer.
    Trains binary classifier with early stopping on leak-free validation split.
    Integrates with threshold optimizer for macro F0.5 evaluation.
    """
    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_memory_mb(self) -> Tuple[float, float]:
        rss = self.process.memory_info().rss / (1024 * 1024)
        if rss > self.peak_memory_mb:
            self.peak_memory_mb = rss
        return rss, self.peak_memory_mb

    def _prepare_training_matrix(self, feature_path: Path, row_count: int, chunk_size: int = 100_000, force: bool = False) -> Tuple[np.memmap, np.memmap]:
        """Materialize selected training columns into resumable disk-backed arrays."""
        matrix_path = config.MODELS_DIR / ".train_matrix.float32.mmap"
        labels_path = config.MODELS_DIR / ".train_labels.int8.mmap"
        matrix_meta_path = matrix_path.with_name(f"{matrix_path.name}.meta.json")
        labels_meta_path = labels_path.with_name(f"{labels_path.name}.meta.json")
        fingerprint = {
            "features": self.state_manager.compute_file_fingerprint(feature_path),
            "config_hash": self.state_manager.get_config_hash(),
            "rows": row_count,
            "columns": FEATURE_COLUMNS,
            "chunk_size": chunk_size,
        }
        expected_matrix_bytes = row_count * len(FEATURE_COLUMNS) * np.dtype(np.float32).itemsize
        expected_labels_bytes = row_count * np.dtype(np.int8).itemsize
        completed_rows = 0
        if not force and matrix_path.is_file() and labels_path.is_file() and matrix_meta_path.is_file() and labels_meta_path.is_file():
            try:
                with matrix_meta_path.open(encoding="utf-8") as stream:
                    matrix_meta = json.load(stream)
                with labels_meta_path.open(encoding="utf-8") as stream:
                    labels_meta = json.load(stream)
                meta = matrix_meta.get("metadata", {})
                if (
                    meta.get("pipeline_fingerprint") == fingerprint
                    and labels_meta.get("metadata", {}).get("pipeline_fingerprint") == fingerprint
                    and matrix_path.stat().st_size == expected_matrix_bytes
                    and labels_path.stat().st_size == expected_labels_bytes
                ):
                    completed_rows = int(meta.get("completed_rows", 0))
            except (OSError, ValueError, TypeError):
                completed_rows = 0
        if not completed_rows:
            matrix_path.unlink(missing_ok=True); labels_path.unlink(missing_ok=True)
            matrix_meta_path.unlink(missing_ok=True); labels_meta_path.unlink(missing_ok=True)
            matrix = np.memmap(matrix_path, mode="w+", dtype=np.float32, shape=(row_count, len(FEATURE_COLUMNS)))
            labels = np.memmap(labels_path, mode="w+", dtype=np.int8, shape=(row_count,))
        else:
            matrix = np.memmap(matrix_path, mode="r+", dtype=np.float32, shape=(row_count, len(FEATURE_COLUMNS)))
            labels = np.memmap(labels_path, mode="r+", dtype=np.int8, shape=(row_count,))

        if completed_rows < row_count:
            feature_scan = pl.scan_parquet(feature_path).select(FEATURE_COLUMNS + ["label"])
            row_offset = 0
            for batch_idx, batch in enumerate(feature_scan.collect_batches(chunk_size=chunk_size)):
                start = row_offset
                end = start + len(batch)
                row_offset = end
                if end <= completed_rows:
                    continue
                if start < completed_rows:
                    raise RuntimeError("Training matrix checkpoint is not aligned with its configured chunk size; rerun with --force.")
                matrix[start:end] = batch.select(pl.col(column).cast(pl.Float32) for column in FEATURE_COLUMNS).to_numpy()
                labels[start:end] = batch["label"].to_numpy().astype(np.int8, copy=False)
                if (batch_idx + 1) % 5 == 0 or end == row_count:
                    matrix.flush(); labels.flush()
                    checkpoint = {"pipeline_fingerprint": fingerprint, "completed_rows": end, "stage": "training_matrix"}
                    StateManager.write_artifact_metadata(matrix_path, end, meta=checkpoint)
                    StateManager.write_artifact_metadata(labels_path, end, meta=checkpoint)
            matrix.flush(); labels.flush()
        StateManager.write_artifact_metadata(
            matrix_path, row_count,
            meta={"pipeline_fingerprint": fingerprint, "completed_rows": row_count, "stage": "training_matrix"},
        )
        StateManager.write_artifact_metadata(
            labels_path, row_count,
            meta={"pipeline_fingerprint": fingerprint, "completed_rows": row_count, "stage": "training_labels"},
        )
        return matrix, labels

    def train_model(
        self,
        train_features_path: Optional[Path] = None,
        val_features_path: Optional[Path] = None,
        dry_run: bool = False,
        force: bool = False,
        device: str = "auto",
        early_stopping_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Trains LightGBM GBDT binary classifier.
        If dry_run is True, initializes data structures and verifies pipeline readiness without full training.
        """
        stage_name = "stage_5_training"
        train_path = Path(train_features_path or (config.FEATURES_DIR / "train_features.parquet"))
        val_path = Path(val_features_path or (config.FEATURES_DIR / "val_features.parquet"))
        model_path = config.MODELS_DIR / "model.pkl"
        threshold_path = config.MODELS_DIR / "optimal_threshold.json"
        self.state_manager.require_artifact(train_path, "Training features (run feature generation first)")
        self.state_manager.require_artifact(val_path, "Validation features required for early stopping and threshold selection")
        if not dry_run and StateManager.should_skip(model_path, force=force) and StateManager.should_skip(threshold_path, force=force):
            logger.info("Model and threshold already exist; skipping training (use --force to retrain).")
            return {"status": "SKIPPED", "model_path": str(model_path), "threshold_path": str(threshold_path)}
        if device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError("device must be one of auto, cpu, gpu, cuda")
        if early_stopping_rounds is not None and early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")

        self.state_manager.mark_in_progress(stage_name, meta={"device_requested": device})
        t_start = time.perf_counter()

        logger.info("[Training Initialization] Loading feature matrices from Parquet...")
        train_schema = set(pl.read_parquet_schema(train_path))
        val_schema = set(pl.read_parquet_schema(val_path))
        required_train = set(FEATURE_COLUMNS + ["label"])
        required_val = required_train | {"source1_entity_id", "candidate_entity_id"}
        if required_train - train_schema:
            raise ValueError(f"Training feature artifact is missing columns: {', '.join(sorted(required_train - train_schema))}")
        if required_val - val_schema:
            raise ValueError(f"Validation feature artifact is missing columns: {', '.join(sorted(required_val - val_schema))}")
        n_train = int(pl.scan_parquet(train_path).select(pl.len()).collect().item())
        n_val = int(pl.scan_parquet(val_path).select(pl.len()).collect().item())
        logger.info("Train rows: %d | Val rows: %d | Features: %s", n_train, n_val, FEATURE_COLUMNS)
        if not dry_run and n_val == 0:
            raise ValueError(f"Validation feature artifact is empty: {val_path}")

        if not dry_run and n_train == 0:
            raise ValueError(f"Training feature artifact is empty: {train_path}")
        if dry_run:
            logger.info("[DRY RUN MODE] Initialized training pipeline successfully. Skipping full epoch training.")
            self.state_manager.mark_completed(stage_name, meta={"dry_run": True, "train_rows": n_train, "val_rows": n_val})
            return {
                "status": "DRY_RUN_SUCCESS",
                "train_rows": n_train,
                "val_rows": n_val,
                "features_count": len(FEATURE_COLUMNS),
                "peak_ram_mb": self.peak_memory_mb
            }

        val_df = pl.read_parquet(val_path, columns=FEATURE_COLUMNS + ["label", "source1_entity_id", "candidate_entity_id"])

        # Extract numpy X and y arrays
        X_train, y_train = self._prepare_training_matrix(train_path, n_train, force=force)

        if n_val > 0:
            X_val = val_df.select(pl.col(column).cast(pl.Float32) for column in FEATURE_COLUMNS).to_numpy()
            y_val = val_df["label"].to_numpy()
            val_pair_ids = val_df.select("source1_entity_id", "candidate_entity_id")

        # Build LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURE_COLUMNS) if n_val > 0 else None
        dtrain.construct()
        if dval is not None:
            dval.construct()
        del X_train, y_train
        if n_val > 0:
            del val_df, y_val
        gc.collect()

        selected_device = self._select_device(device)
        params = dict(config.LIGHTGBM_PARAMS)
        params["device_type"] = selected_device
        num_boost_round = int(params.pop("n_estimators", 100))
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds or config.EARLY_STOPPING_ROUNDS)] if n_val > 0 else []

        logger.info("[LightGBM] Starting training with params: %s", params)
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval] if dval else [dtrain],
            valid_names=["train", "val"] if dval else ["train"],
            callbacks=callbacks,
            num_boost_round=num_boost_round,
        )

        # Save serialized model binary atomically.
        import joblib
        tmp_model = model_path.with_name(f"{model_path.name}.tmp")
        joblib.dump(model, tmp_model)
        tmp_model.replace(model_path)
        logger.info("Saved trained LightGBM binary to %s", model_path)

        if n_val > 0:
            val_probs = model.predict(X_val, num_iteration=model.best_iteration or None)
            del X_val
            ThresholdOptimizer(state_manager=self.state_manager).optimize_threshold(val_probs, val_pair_ids)
            del val_probs, val_pair_ids
            gc.collect()

        # Compute feature importances
        importance_dict = dict(zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain").tolist()))
        logger.info("Top Feature Importances (Gain): %s", sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])

        elapsed = time.perf_counter() - t_start
        rss_mb, peak_mb = self._get_memory_mb()

        self.state_manager.record_artifact(
            "model_binary",
            model_path,
            row_count=n_train,
            meta={"best_iteration": model.best_iteration, "feature_importance_gain": importance_dict, "device_type": selected_device}
        )
        self.state_manager.mark_completed(stage_name, meta={
            "train_rows": n_train,
            "best_iteration": model.best_iteration,
            "device_type": selected_device,
            "time_sec": elapsed,
            "peak_ram_mb": peak_mb
        })

        return {
            "status": "TRAINED_SUCCESSFULLY",
            "train_rows": n_train,
            "val_rows": n_val,
            "best_iteration": model.best_iteration,
            "time_sec": elapsed,
            "peak_ram_mb": peak_mb,
            "model_path": str(model_path)
        }

    @staticmethod
    def _select_device(requested: str) -> str:
        """Use CPU locally; on Kaggle probe actual LightGBM GPU support before selecting it."""
        if requested == "cpu":
            return "cpu"
        candidates = [requested] if requested in {"gpu", "cuda"} else []
        if requested == "auto" and Path("/kaggle").exists():
            candidates = ["cuda", "gpu"]
        for candidate in candidates:
            try:
                probe = lgb.Dataset(np.array([[0.0], [1.0]], dtype=np.float32), label=np.array([0, 1]))
                lgb.train({"objective": "binary", "verbosity": -1, "device_type": candidate, "num_leaves": 2}, probe, num_boost_round=1)
                return candidate
            except Exception as exc:
                if requested != "auto":
                    raise RuntimeError(f"Requested LightGBM device '{candidate}' is unavailable: {exc}") from exc
                logger.warning("LightGBM %s is unavailable; checking fallback: %s", candidate, exc)
        if requested in {"gpu", "cuda"}:
            raise RuntimeError(f"Requested LightGBM device '{requested}' is unavailable in this LightGBM build.")
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the entity matching LightGBM model")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    parser.add_argument("--early-stopping", type=int, default=None, help="Early stopping rounds (default: config value)")
    parser.add_argument("--force", action="store_true", help="Retrain even if model and threshold artifacts exist")
    parser.add_argument("--dry-run", action="store_true", help="Check training artifact readiness without fitting")
    args = parser.parse_args()
    EntityMatcherTrainer().train_model(
        dry_run=args.dry_run, force=args.force, device=args.device,
        early_stopping_rounds=args.early_stopping,
    )


if __name__ == "__main__":
    main()
