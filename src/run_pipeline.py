import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.state_manager import StateManager
from src.data_loader import PolarsChunkLoader
from src.blocking import MultiLayerBlocker
from src.feature_engineering import FeatureExtractor
from src.train import EntityMatcherTrainer
from src.inference import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PipelineOrchestrator")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Master Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        choices=["setup", "block", "train", "infer", "validate", "all"],
        default="all",
        help="Stage to execute"
    )
    parser.add_argument("--dry-run", action="store_true", help="Execute lightweight verification without heavy compute")
    parser.add_argument("--country", help="Process one existing country partition during blocking")
    parser.add_argument("--force", action="store_true", help="Rebuild generated outputs even when checkpoints already exist")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto", help="LightGBM device for training")
    parser.add_argument("--early-stopping", type=int, help="Override LightGBM early stopping rounds")
    args = parser.parse_args()
    if args.country and args.stage != "block":
        parser.error("--country is supported only with --stage block")

    sm = StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
    logger.info("=== Amazon ML Challenge 2026 Pipeline Orchestrator (Stage: %s) ===", args.stage)

    if args.stage in ["block", "all"]:
        logger.info("--- STAGE 1..3: DATA LOADING & BLOCKING ---")
        loader = PolarsChunkLoader(state_manager=sm)
        val_ids = loader.get_or_create_val_ids(force=args.force)
        for source in ("train_source1", "train_source2", "train_source3"):
            loader.stream_and_partition_source(
                source, config.DATASET_DIR / "train" / f"{source}.tsv", val_ids,
                country=args.country,
                force=args.force,
            )
        # Ground truth is small and shared by downstream labeling; keep the complete split.
        loader.stream_and_partition_ground_truth(val_ids, force=args.force)
        if not args.country:
            # Test data stays on the normal full-dataset inference path.
            for source in ("source1", "source2", "source3"):
                loader.stream_and_partition_source(
                    f"test_{source}", config.DATASET_DIR / "test" / f"test_{source}.tsv", force=args.force
                )

        if not args.dry_run:
            blocker = MultiLayerBlocker(state_manager=sm)
            block_modes = ["val", "train"] + ([] if args.country else ["test"])
            for mode in block_modes:
                partition_output = (config.BLOCKED_DIR / args.country / f"{mode}_candidates.parquet") if args.country else (config.BLOCKED_DIR / f"{mode}_candidates.parquet")
                if StateManager.should_skip(partition_output, force=args.force):
                    logger.info("[%s blocking] Existing artifact %s; skipping.", mode, partition_output)
                    continue
                blocker.run_blocking_pipeline(mode=mode, country=args.country)

    if args.stage in ["train", "all"] and not args.dry_run:
        logger.info("--- STAGE 4..5: FEATURE ENGINEERING & MODEL TRAINING ---")
        fe = FeatureExtractor(state_manager=sm)
        fe.extract_features_for_mode("val", force=args.force)
        fe.extract_features_for_mode("train", force=args.force)

        trainer = EntityMatcherTrainer(state_manager=sm)
        trainer.train_model(dry_run=args.dry_run, force=args.force, device=args.device, early_stopping_rounds=args.early_stopping)

    if args.stage in ["infer", "all"] and not args.dry_run:
        existing_submission = (
            config.OUTPUT_DIR / "matching_results.tsv",
            config.OUTPUT_DIR / "candidate_pairs.tsv",
        )
        if not args.force and all(StateManager.should_skip(path) for path in existing_submission):
            logger.info("Existing submission TSVs found; skipping test preparation and inference (use --force to rebuild).")
        else:
            # Standalone infer must also prepare test data, blocking, and features.
            missing_test_sources = [
                source for source in ("source1", "source2", "source3")
                if args.force or not list(config.CLEANED_DIR.rglob(f"test_{source}_part_*.parquet"))
            ]
            if missing_test_sources:
                loader = PolarsChunkLoader(state_manager=sm)
                for source in missing_test_sources:
                    loader.stream_and_partition_source(
                        f"test_{source}", config.DATASET_DIR / "test" / f"test_{source}.tsv", force=args.force
                    )
            test_candidates = config.BLOCKED_DIR / "test_candidates.parquet"
            if not test_candidates.exists() or args.force:
                MultiLayerBlocker(state_manager=sm).run_blocking_pipeline(mode="test")
            test_features = config.FEATURES_DIR / "test_features.parquet"
            if not test_features.exists() or args.force:
                FeatureExtractor(state_manager=sm).extract_features_for_mode("test", force=args.force)

    if args.stage in ["infer", "all"] or args.dry_run:
        logger.info("--- STAGE 6: INFERENCE & SUBMISSION GENERATION ---")
        infer = InferencePipeline(state_manager=sm)
        infer.run_inference(dry_run=args.dry_run, force=args.force)

    if args.stage == "validate":
        infer = InferencePipeline(state_manager=sm)
        infer.run_official_validator()

    logger.info("=== Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
