import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.blocking import (
    MultiLayerBlocker,
    accumulate_tfidf_scores,
    score_tfidf_query_sparse,
    select_layer4_worker_count,
)
from src.config import config
from src.data_loader import PolarsChunkLoader
from src.feature_engineering import FeatureExtractor
from src.merge_candidates import merge_mode
from src.state_manager import StateManager


class CorrectnessTests(unittest.TestCase):
    def test_layer4_worker_count_respects_cpu_and_ram_budget(self):
        gib = 1024 ** 3
        workers, plan = select_layer4_worker_count(
            n_queries=40_021,
            n_targets=300_000,
            indices_dtype=np.dtype(np.int32),
            cpu_count=16,
            available_ram_bytes=13 * gib,
            total_ram_bytes=16 * gib,
        )
        self.assertEqual(workers, 16)
        self.assertLessEqual(workers, plan["cpu_limit"])
        self.assertEqual(plan["reserved_ram_bytes"], 3 * gib)
        self.assertLessEqual(
            workers * plan["estimated_worker_ram_bytes"],
            plan["available_ram_bytes"] - plan["reserved_ram_bytes"],
        )

        low_memory_workers, _ = select_layer4_worker_count(
            n_queries=40_021,
            n_targets=300_000,
            indices_dtype=np.dtype(np.int32),
            cpu_count=16,
            available_ram_bytes=4 * gib,
            total_ram_bytes=16 * gib,
        )
        self.assertLess(low_memory_workers, workers)

    def test_score_accumulation_matches_unique_bincount_reduction(self):
        target = sp.csr_matrix(np.array([
            [0.5, 0.0, 0.25],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.75],
        ], dtype=np.float64)).tocsc()
        features = np.array([0, 1, 2])
        weights = np.array([0.2, 0.3, 0.4])

        posting_ids = []
        posting_scores = []
        for feature, weight in zip(features, weights):
            start, end = target.indptr[feature:feature + 2]
            posting_ids.append(target.indices[start:end])
            posting_scores.append(target.data[start:end] * weight)
        ids, inverse = np.unique(np.concatenate(posting_ids), return_inverse=True)
        expected = np.bincount(inverse, weights=np.concatenate(posting_scores), minlength=len(ids))

        actual = np.zeros(target.shape[0], dtype=np.float64)
        touched = np.empty(target.shape[0], dtype=target.indices.dtype)
        reached = accumulate_tfidf_scores(target, features, weights, actual, touched)
        sparse_ids, sparse_scores = score_tfidf_query_sparse(target, features, weights)
        np.testing.assert_array_equal(reached, ids)
        np.testing.assert_array_equal(actual[ids], expected)
        np.testing.assert_array_equal(sparse_ids, ids)
        np.testing.assert_array_equal(sparse_scores, expected)
        legacy_candidates = ids[expected >= 0.15]
        legacy_scores = expected[expected >= 0.15]
        legacy_top = np.argpartition(legacy_scores, -2)[-2:]
        new_candidates = reached[actual[reached] >= 0.15]
        new_scores = actual[new_candidates]
        new_top = np.argpartition(new_scores, -2)[-2:]
        np.testing.assert_array_equal(new_candidates[new_top], legacy_candidates[legacy_top])

    def test_tiny_blocking_is_deterministic_and_reports_all_boundaries(self):
        s1 = pl.DataFrame({
            "entity_id": ["q1", "q2"],
            "business_name": ["Acme Trading LLC", "Unrelated Name"],
            "business_address": ["12 Main Street", "9 Other Road"],
            "country": ["US", "US"],
        })
        targets = pl.DataFrame({
            "entity_id": ["t1", "t2", "t3"],
            "business_name": ["ACME TRADING", "Acme Traders", "Different Company"],
            "business_address": ["12 Main St", "44 Side Street", "1 Elsewhere"],
            "country": ["US", "US", "US"],
        })
        truth = {"q1": {"t1"}, "q2": {"missing-target"}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = StateManager(root / "progress.json", root / "manifest.json")
            blocker = MultiLayerBlocker(state_manager=state, top_k=2, layer4_internal_top_k=3)
            first, first_stats = blocker.block_country_partition("US", s1, targets, truth)
            second, second_stats = blocker.block_country_partition("US", s1, targets, truth)
            streamed, streamed_stats = blocker.block_country_partition(
                "US", s1, targets, truth, output_parts_dir=root / "parts"
            )
            streamed_rows = pl.concat([
                pl.read_parquet(path) for path in (root / "parts").glob("part_*.parquet")
            ])

        self.assertTrue(first.equals(second))
        self.assertTrue(first.equals(streamed_rows))
        self.assertEqual(streamed.height, 0)
        self.assertEqual(streamed_stats["candidates_generated"], len(first))
        self.assertLessEqual(first.group_by("source1_entity_id").len()["len"].max(), 2)
        self.assertEqual(first.schema, second.schema)
        self.assertEqual(first_stats["diagnostics"], second_stats["diagnostics"])
        json.dumps(first_stats)
        self.assertEqual(first_stats["diagnostics"]["truth_pairs"], 2)
        self.assertEqual(first_stats["diagnostics"]["target_present_truth_pairs"], 1)
        self.assertEqual(
            set(first_stats["diagnostics"]["survival"]),
            {
                "after_layer2", "after_layer3", "after_layer4_before_threshold",
                "after_layer4_before_top_k", "after_layer4_top_k",
                "after_combined_pool", "after_final_top2",
            },
        )

    def test_validation_scope_counts_queries_with_no_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            val_dir = Path(tmp)
            pl.DataFrame({
                "source1_entity_id": ["india-hit", "india-miss", "us-hit"],
                "matched_entity_ids": ["i1", "i2", "u1"],
            }).write_parquet(val_dir / "val_ground_truth_part_00000.parquet")
            candidates = pl.DataFrame({
                "source1_entity_id": ["india-hit"],
                "candidate_entity_id": ["i1"],
                "l2": [True], "l3": [False], "l4": [False],
            })
            blocker = MultiLayerBlocker.__new__(MultiLayerBlocker)
            with patch.object(config, "VAL_SPLIT_DIR", val_dir):
                metrics = blocker._estimate_validation_recall(
                    candidates, {"india-hit", "india-miss"}
                )
        self.assertEqual(metrics["overall"], 0.5)
        self.assertEqual(metrics["layer2"], 0.5)

    def test_loader_does_not_replace_full_run_country_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.tsv"
            source.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "q1\tOne LLC\t1 Main St\tIndia\n"
                "q2\tTwo LLC\t2 Main St\tUS\n"
                "q3\tThree LLC\t3 Main St\tIndia\n"
                "q4\tFour LLC\t4 Main St\tUS\n",
                encoding="utf-8",
            )
            cleaned = root / "cleaned"
            validation = root / "validation"
            state = StateManager(root / "progress.json", root / "manifest.json")
            with patch.object(config, "CLEANED_DIR", cleaned), patch.object(config, "VAL_SPLIT_DIR", validation):
                loader = PolarsChunkLoader(state_manager=state, chunk_size=2)
                loader.stream_and_partition_source(
                    "train_source1", source, val_ids={"q1", "q2"}
                )
            india = pl.scan_parquet(cleaned / "India" / "*.parquet").collect()
            us = pl.scan_parquet(cleaned / "US" / "*.parquet").collect()
        self.assertEqual(india["entity_id"].to_list(), ["q3"])
        self.assertEqual(us["entity_id"].to_list(), ["q4"])

    def test_feature_extractor_normalizes_legacy_partitions(self):
        legacy = pl.DataFrame({
            "entity_id": ["x"],
            "business_name": ["Café LLC"],
            "business_address": ["1 Main St."],
            "country": ["US"],
        })
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        from src.normalizer import EntityNormalizer
        extractor.normalizer = EntityNormalizer()
        normalized = extractor._ensure_normalized(legacy)
        self.assertEqual(normalized["name_clean"][0], "cafe")
        self.assertEqual(normalized["address_clean"][0], "1 main street")

    def test_candidate_schema_remains_merge_compatible(self):
        candidate = pl.DataFrame({
            "source1_entity_id": pl.Series(["q1"], dtype=pl.String),
            "candidate_entity_id": pl.Series(["t1"], dtype=pl.String),
            "heuristic_score": pl.Series([2.0], dtype=pl.Float32),
            "layers_matched": pl.Series([1], dtype=pl.Int8),
            "l2": pl.Series([True], dtype=pl.Boolean),
            "l3": pl.Series([False], dtype=pl.Boolean),
            "l4": pl.Series([False], dtype=pl.Boolean),
        })
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp)
            for country in ("India", "US"):
                (blocked / country).mkdir()
                candidate.write_parquet(blocked / country / "val_candidates.parquet")
            output = blocked / "val_candidates.parquet"
            with patch.object(config, "BLOCKED_DIR", blocked):
                merge_mode("val", ["India", "US"], output)
            merged = pl.read_parquet(output)
        self.assertEqual(merged.schema, candidate.schema)
        self.assertEqual(merged.height, 2)


if __name__ == "__main__":
    unittest.main()
