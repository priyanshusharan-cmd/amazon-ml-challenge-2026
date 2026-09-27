import gc
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple

import numpy as np
import polars as pl
import psutil
from rapidfuzz import fuzz, distance

from src.config import config
from src.normalizer import EntityNormalizer
from src.state_manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FeatureEngineering")

FEATURE_COLUMNS = [
    "name_rapidfuzz_ratio",
    "name_token_sort_ratio",
    "name_jaro_winkler",
    "name_jaccard_similarity",
    "name_len_ratio",
    "address_token_overlap",
    "address_numeric_agreement",
    "same_country",
    "legal_suffix_match",
    "missing_address_flag",
    "heuristic_score",
    "layers_matched"
]


class FeatureExtractor:
    """
    Vectorized Pairwise Feature Engineering Engine.
    Generates exact similarity metrics on candidate pairs in memory-bounded chunks.
    Fully resumable with metadata checkpoints.
    """
    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.normalizer = EntityNormalizer()
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    def _ensure_normalized(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize legacy partitions that predate persisted clean columns."""
        required = {"name_clean", "address_clean", "legal_suffix", "has_address"}
        return df if required.issubset(df.columns) else self.normalizer.normalize_polars_df(df)

    def _get_memory_mb(self) -> Tuple[float, float]:
        rss = self.process.memory_info().rss / (1024 * 1024)
        if rss > self.peak_memory_mb:
            self.peak_memory_mb = rss
        return rss, self.peak_memory_mb

    def _atomic_write_parquet(self, df: pl.DataFrame, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f"{out_path.name}.tmp")
        df.write_parquet(tmp_path, compression="snappy")
        os.replace(tmp_path, out_path)

    # --------------------------------------------------------------------------
    # RapidFuzz & Numeric Vectorized Metric Computation
    # --------------------------------------------------------------------------
    def compute_pair_features(
        self,
        s1_names: List[str],
        s1_addrs: List[str],
        s1_suffixes: List[str],
        cand_names: List[str],
        cand_addrs: List[str],
        cand_suffixes: List[str],
        heuristic_scores: List[float],
        layers_matched: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Computes 12 pairwise feature vectors for candidate pairs.
        """
        n_pairs = len(s1_names)

        ratio_feat = np.zeros(n_pairs, dtype=np.float32)
        token_sort_feat = np.zeros(n_pairs, dtype=np.float32)
        jaro_feat = np.zeros(n_pairs, dtype=np.float32)
        jaccard_feat = np.zeros(n_pairs, dtype=np.float32)
        name_len_feat = np.zeros(n_pairs, dtype=np.float32)

        addr_overlap_feat = np.zeros(n_pairs, dtype=np.float32)
        num_agree_feat = np.zeros(n_pairs, dtype=np.float32)
        suffix_match_feat = np.zeros(n_pairs, dtype=np.float32)
        missing_addr_feat = np.zeros(n_pairs, dtype=np.float32)

        for i in range(n_pairs):
            n1 = s1_names[i] or ""
            n2 = cand_names[i] or ""
            a1 = s1_addrs[i] or ""
            a2 = cand_addrs[i] or ""
            suf1 = s1_suffixes[i] or "none"
            suf2 = cand_suffixes[i] or "none"

            # 1. RapidFuzz Name Similarity
            ratio_feat[i] = fuzz.ratio(n1, n2) / 100.0
            token_sort_feat[i] = fuzz.token_sort_ratio(n1, n2) / 100.0
            jaro_feat[i] = distance.JaroWinkler.similarity(n1, n2)

            # 2. Token Jaccard Similarity
            toks1 = set(n1.split())
            toks2 = set(n2.split())
            union_len = len(toks1 | toks2)
            jaccard_feat[i] = (len(toks1 & toks2) / union_len) if union_len > 0 else 0.0

            # 3. Name Length Ratio
            l1, l2 = len(n1), len(n2)
            name_len_feat[i] = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

            # 4. Address Features
            has_a1 = len(a1.strip()) > 0
            has_a2 = len(a2.strip()) > 0

            if not has_a1 or not has_a2:
                missing_addr_feat[i] = 1.0
                addr_overlap_feat[i] = 0.0
                num_agree_feat[i] = 0.0
            else:
                missing_addr_feat[i] = 0.0
                atok1 = set(a1.split())
                atok2 = set(a2.split())
                a_union = len(atok1 | atok2)
                addr_overlap_feat[i] = (len(atok1 & atok2) / a_union) if a_union > 0 else 0.0

                # Numeric Token Agreement
                nums1 = set(re.findall(r'\b\d+\b', a1))
                nums2 = set(re.findall(r'\b\d+\b', a2))
                num_union = len(nums1 | nums2)
                num_agree_feat[i] = (len(nums1 & nums2) / num_union) if num_union > 0 else 0.0

            # 5. Legal Suffix Match
            if suf1 != "none" and suf2 != "none":
                suffix_match_feat[i] = 1.0 if suf1 == suf2 else 0.0
            elif suf1 == "none" and suf2 == "none":
                suffix_match_feat[i] = 0.5
            else:
                suffix_match_feat[i] = 0.25

        return {
            "name_rapidfuzz_ratio": ratio_feat,
            "name_token_sort_ratio": token_sort_feat,
            "name_jaro_winkler": jaro_feat,
            "name_jaccard_similarity": jaccard_feat,
            "name_len_ratio": name_len_feat,
            "address_token_overlap": addr_overlap_feat,
            "address_numeric_agreement": num_agree_feat,
            "same_country": np.ones(n_pairs, dtype=np.float32),
            "legal_suffix_match": suffix_match_feat,
            "missing_address_flag": missing_addr_feat,
            "heuristic_score": np.array(heuristic_scores, dtype=np.float32),
            "layers_matched": np.array(layers_matched, dtype=np.int8)
        }

    # --------------------------------------------------------------------------
    # Pipeline Runner for Feature Extraction
    # --------------------------------------------------------------------------
    def _lookup_for_batch(self, files: List[Path], entity_ids: List[str]) -> Dict[str, Tuple[str, str, str]]:
        """Read and normalize only entity rows needed by one candidate batch."""
        if not files or not entity_ids:
            return {}
        frame = (
            pl.scan_parquet([str(path) for path in files])
            .filter(pl.col("entity_id").is_in(entity_ids))
            .collect()
        )
        if frame.is_empty():
            return {}
        frame = self._ensure_normalized(frame)
        return {
            entity_id: (name or "", address or "", suffix or "none")
            for entity_id, name, address, suffix in zip(
                frame["entity_id"].to_list(),
                frame["name_clean"].to_list(),
                frame["address_clean"].to_list(),
                frame["legal_suffix"].to_list(),
            )
        }

    def extract_features_for_mode(
        self,
        mode: str = "train",
        force: bool = False,
        chunk_size: int = 50_000
    ) -> Dict[str, Any]:
        """Generate features in resumable, bounded candidate batches."""
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported feature mode: {mode}")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        stage_name = f"stage_4_feature_engineering_{mode}"
        out_parquet = config.FEATURES_DIR / f"{mode}_features.parquet"
        cand_path = self.state_manager.require_artifact(
            config.BLOCKED_DIR / f"{mode}_candidates.parquet",
            f"{mode} candidate pairs (run blocking/merge first)",
        )
        if StateManager.should_skip(out_parquet, force=force):
            logger.info("[%s features] Existing output %s; skipping (use --force to rebuild).", mode.upper(), out_parquet)
            return {"status": "SKIPPED", "mode": mode, "out_parquet": str(out_parquet)}

        if mode == "val":
            s1_files = list(config.VAL_SPLIT_DIR.rglob("val_s1_part_*.parquet"))
            gt_files = list(config.VAL_SPLIT_DIR.glob("val_ground_truth_part_*.parquet"))
        elif mode == "test":
            s1_files = list(config.CLEANED_DIR.rglob("test_source1_part_*.parquet"))
            gt_files = []
        else:
            s1_files = list(config.CLEANED_DIR.rglob("train_source1_part_*.parquet"))
            gt_files = list(config.CLEANED_DIR.glob("train_ground_truth_part_*.parquet"))
        target_files = (
            list(config.CLEANED_DIR.rglob("test_source2_part_*.parquet"))
            + list(config.CLEANED_DIR.rglob("test_source3_part_*.parquet"))
            if mode == "test" else
            list(config.CLEANED_DIR.rglob("train_source2_part_*.parquet"))
            + list(config.CLEANED_DIR.rglob("train_source3_part_*.parquet"))
        )
        missing = [label for label, paths in (("source1 entity partitions", s1_files), ("target entity partitions", target_files)) if not paths]
        if missing:
            raise FileNotFoundError(f"Missing required {mode} feature inputs: {', '.join(missing)} under {config.ARTIFACTS_DIR}")
        if mode in {"train", "val"} and not gt_files:
            raise FileNotFoundError(f"Missing {mode} ground-truth partitions under {config.ARTIFACTS_DIR}; cannot generate labels.")
        candidate_scan = pl.scan_parquet(cand_path)
        required_candidate_columns = {"source1_entity_id", "candidate_entity_id", "heuristic_score", "layers_matched"}
        missing_columns = required_candidate_columns - set(candidate_scan.collect_schema().names())
        if missing_columns:
            raise ValueError(f"Candidate artifact {cand_path} is missing required columns: {', '.join(sorted(missing_columns))}")

        self.state_manager.mark_in_progress(stage_name, meta={"chunk_size": chunk_size})
        started = time.perf_counter()

        # Batch outputs stay on disk across interruption. The final legacy Parquet
        # filename remains available to existing training/inference commands.
        parts_dir = config.FEATURES_DIR / f".{mode}_feature_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        if force:
            for old in parts_dir.glob("part_*.parquet*"):
                old.unlink(missing_ok=True)
        gt_scans = [str(path) for path in gt_files]
        candidate_fingerprint = self.state_manager.compute_file_fingerprint(cand_path)
        input_fingerprints = {
            "candidates": candidate_fingerprint,
            "source1": [self.state_manager.compute_file_fingerprint(path) for path in s1_files],
            "targets": [self.state_manager.compute_file_fingerprint(path) for path in target_files],
            "ground_truth": [self.state_manager.compute_file_fingerprint(path) for path in gt_files],
        }
        part_paths: List[Path] = []
        total_rows = 0
        batch_count = 0
        scan = candidate_scan

        try:
            for batch_idx, batch in enumerate(scan.collect_batches(chunk_size=chunk_size)):
                part_path = parts_dir / f"part_{batch_idx:06d}.parquet"
                part_meta = part_path.with_name(f"{part_path.name}.meta.json")
                reusable = False
                if part_path.is_file() and part_path.stat().st_size > 0 and part_meta.is_file():
                    try:
                        with part_meta.open(encoding="utf-8") as stream:
                            saved = json.load(stream)
                        chunk_meta = saved.get("metadata", {})
                        reusable = (
                            saved.get("config_hash") == self.state_manager.get_config_hash()
                            and chunk_meta.get("chunk_size") == chunk_size
                            and chunk_meta.get("input_fingerprints") == input_fingerprints
                        )
                        if reusable:
                            rows = int(saved["row_count"])
                            total_rows += rows
                    except (OSError, ValueError, KeyError, TypeError):
                        reusable = False
                if not reusable:
                    part_path.unlink(missing_ok=True)
                    part_meta.unlink(missing_ok=True)
                    s1_ids = batch["source1_entity_id"].to_list()
                    candidate_ids = batch["candidate_entity_id"].to_list()
                    s1_lookup = self._lookup_for_batch(s1_files, list(set(s1_ids)))
                    target_lookup = self._lookup_for_batch(target_files, list(set(candidate_ids)))

                    s1_names, s1_addrs, s1_suffixes = [], [], []
                    candidate_names, candidate_addrs, candidate_suffixes = [], [], []
                    for s1_id, candidate_id in zip(s1_ids, candidate_ids):
                        n1, a1, suffix1 = s1_lookup.get(s1_id, ("", "", "none"))
                        n2, a2, suffix2 = target_lookup.get(candidate_id, ("", "", "none"))
                        s1_names.append(n1); s1_addrs.append(a1); s1_suffixes.append(suffix1)
                        candidate_names.append(n2); candidate_addrs.append(a2); candidate_suffixes.append(suffix2)
                    del s1_lookup, target_lookup

                    features = self.compute_pair_features(
                        s1_names, s1_addrs, s1_suffixes,
                        candidate_names, candidate_addrs, candidate_suffixes,
                        batch["heuristic_score"].to_list(), batch["layers_matched"].to_list(),
                    )
                    columns = {
                        "source1_entity_id": s1_ids,
                        "candidate_entity_id": candidate_ids,
                        **features,
                    }
                    if gt_scans:
                        query_ids = list(set(s1_ids))
                        gt_batch = (
                            pl.scan_parquet(gt_scans)
                            .filter(pl.col("source1_entity_id").is_in(query_ids))
                            .collect()
                        )
                        truth = set()
                        for source_id, matched_ids in zip(gt_batch["source1_entity_id"], gt_batch["matched_entity_ids"]):
                            if matched_ids:
                                truth.update((source_id, candidate_id) for candidate_id in matched_ids.split(",") if candidate_id)
                        columns["label"] = np.fromiter(
                            (int((source_id, candidate_id) in truth) for source_id, candidate_id in zip(s1_ids, candidate_ids)),
                            dtype=np.int8,
                            count=len(s1_ids),
                        )
                    chunk_df = pl.DataFrame(columns)
                    tmp_part = part_path.with_name(f"{part_path.name}.tmp")
                    chunk_df.write_parquet(tmp_part, compression="snappy")
                    os.replace(tmp_part, part_path)
                    StateManager.write_artifact_metadata(
                        part_path, len(chunk_df),
                        meta={"mode": mode, "batch_index": batch_idx, "chunk_size": chunk_size, "input_fingerprints": input_fingerprints},
                    )
                    total_rows += len(chunk_df)
                    del chunk_df, columns, features
                    gc.collect()
                part_paths.append(part_path)
                batch_count += 1
                if batch_count % 20 == 0:
                    rss_mb, peak_mb = self._get_memory_mb()
                    logger.info("[%s features] batches=%d rows=%d RSS=%.0f MB peak=%.0f MB", mode.upper(), batch_count, total_rows, rss_mb, peak_mb)

            if not part_paths:
                empty = {"source1_entity_id": [], "candidate_entity_id": []}
                empty.update({column: [] for column in FEATURE_COLUMNS})
                if mode != "test":
                    empty["label"] = []
                tmp_final = out_parquet.with_name(f"{out_parquet.name}.tmp")
                pl.DataFrame(empty).write_parquet(tmp_final, compression="snappy")
                os.replace(tmp_final, out_parquet)
            else:
                tmp_final = out_parquet.with_name(f"{out_parquet.name}.tmp")
                pl.scan_parquet([str(path) for path in part_paths]).sink_parquet(tmp_final, compression="snappy")
                os.replace(tmp_final, out_parquet)
        except Exception:
            # Completed batch Parquets and sidecars are intentionally retained for resume.
            raise

        elapsed = time.perf_counter() - started
        actual_rows = int(pl.scan_parquet(out_parquet).select(pl.len()).collect().item())
        rss_mb, peak_mb = self._get_memory_mb()
        self.state_manager.record_artifact(
            f"{mode}_features", out_parquet, actual_rows,
            meta={"features": FEATURE_COLUMNS, "time_sec": elapsed, "peak_ram_mb": peak_mb, "batch_count": batch_count, "chunk_size": chunk_size},
        )
        self.state_manager.mark_completed(stage_name, meta={"total_pairs": actual_rows, "batches": batch_count, "time_sec": elapsed})
        shutil.rmtree(parts_dir, ignore_errors=True)
        logger.info("[%s features] complete: %d rows, %d batches, %.1fs, peak RSS %.0f MB", mode.upper(), actual_rows, batch_count, elapsed, peak_mb)
        return {
            "mode": mode, "total_pairs": actual_rows,
            "throughput_pairs_per_sec": actual_rows / elapsed if elapsed else 0.0,
            "time_sec": elapsed, "peak_ram_mb": peak_mb,
            "feature_columns": FEATURE_COLUMNS, "out_parquet": str(out_parquet),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pairwise Feature Extraction Engine")
    parser.add_argument("--mode", choices=["train", "val", "test", "all"], default="all", help="Mode to extract features for")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if output artifact exists")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Candidate processing chunk size (default: 50000)")
    args = parser.parse_args()

    fe = FeatureExtractor()
    modes = ["val", "train", "test"] if args.mode == "all" else [args.mode]
    for m in modes:
        fe.extract_features_for_mode(m, force=args.force, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
