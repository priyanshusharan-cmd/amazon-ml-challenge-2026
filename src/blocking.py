import gc
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple

import numpy as np
import polars as pl
import psutil
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import config
from src.state_manager import StateManager
from src.normalizer import EntityNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BlockingEngine")

GLOBAL_STOP_WORDS = {
    'and', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', '&',
    'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'sarl', 'sas', 'sci', 'sa', 'eurl',
    'services', 'enterprises', 'solutions', 'traders', 'trading', 'group',
    'center', 'centre', 'industries', 'international', 'india', 'us', 'france'
}


def accumulate_tfidf_scores(index, features, weights, scores, touched):
    """Exact dot products using a reusable float64 target accumulator.

    The CSC index must be canonical (each target occurs once per feature).
    Contributions are added in query-feature order, as in the former
    concatenate/unique/bincount implementation. No postings are pruned.
    """
    touched_count = 0
    for feature, weight in zip(features, weights):
        start, end = index.indptr[feature:feature + 2]
        ids = index.indices[start:end]
        new_ids = ids[scores[ids] == 0]
        new_count = len(new_ids)
        touched[touched_count:touched_count + new_count] = new_ids
        touched_count += new_count
        scores[ids] += index.data[start:end] * weight
    # np.unique returned ascending IDs. Preserve that order so thresholding,
    # argpartition input, and tie behavior remain unchanged.
    return np.sort(touched[:touched_count])


def add_diagnostic_losses(diagnostics):
    """Add explicit loss buckets derived from the survival counters."""
    survival = diagnostics["survival"]
    found = lambda stage: survival[stage]["found_truth_pairs"]
    diagnostics["losses"] = {
        "ingestion_missing_truth_pairs": diagnostics["missing_target_truth_pairs"],
        "layer4_no_shared_gram_pairs": diagnostics["target_present_truth_pairs"] - found("after_layer4_before_threshold"),
        "layer4_threshold_loss_pairs": found("after_layer4_before_threshold") - found("after_layer4_before_top_k"),
        "layer4_top_k_loss_pairs": found("after_layer4_before_top_k") - found("after_layer4_top_k"),
        "combined_pool_missing_pairs": diagnostics["target_present_truth_pairs"] - found("after_combined_pool"),
        "layer5_ranking_loss_pairs": found("after_combined_pool") - found("after_final_top20"),
    }
    return diagnostics


class MultiLayerBlocker:
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        top_k: int = config.TOP_K,
        layer4_internal_top_k: int = config.LAYER4_INTERNAL_TOP_K,
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if layer4_internal_top_k < top_k:
            raise ValueError("layer4_internal_top_k must be at least top_k")
        self.top_k = top_k
        self.layer4_internal_top_k = layer4_internal_top_k
        self.state_manager = state_manager or StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.normalizer = EntityNormalizer()
        self.process = psutil.Process()
        self.peak_memory_mb = 0.0
        config.BLOCKED_DIR.mkdir(parents=True, exist_ok=True)

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

    def block_country_partition(
        self,
        country: str,
        s1_df: pl.DataFrame,
        target_df: pl.DataFrame,
        true_matches: Optional[Dict[str, Set[str]]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        t0 = time.perf_counter()
        
        normalized_columns = {"name_clean", "address_clean", "legal_suffix", "has_address"}
        if not normalized_columns.issubset(s1_df.columns):
            s1_df = self.normalizer.normalize_polars_df(s1_df)
        if not normalized_columns.issubset(target_df.columns):
            target_df = self.normalizer.normalize_polars_df(target_df)

        n_s1 = len(s1_df)
        n_target = len(target_df)
        
        if n_s1 == 0 or n_target == 0:
            empty_df = pl.DataFrame({
                "source1_entity_id": pl.Series([], dtype=pl.String),
                "candidate_entity_id": pl.Series([], dtype=pl.String),
                "heuristic_score": pl.Series([], dtype=pl.Float32),
                "layers_matched": pl.Series([], dtype=pl.Int8),
                "l2": pl.Series([], dtype=pl.Boolean),
                "l3": pl.Series([], dtype=pl.Boolean),
                "l4": pl.Series([], dtype=pl.Boolean),
            })
            stats = {
                "country": country,
                "s1_entities": n_s1,
                "target_entities": n_target,
                "candidates_generated": 0,
                "avg_candidates_per_entity": 0.0,
                "time_sec": 0.0,
                "peak_ram_mb": self.peak_memory_mb,
            }
            if true_matches is not None:
                truth_pairs = sum(len(true_matches.get(entity_id, set())) for entity_id in s1_df["entity_id"])
                diagnostics = {
                    "query_entities": n_s1,
                    "queries_with_ground_truth_row": sum(entity_id in true_matches for entity_id in s1_df["entity_id"]),
                    "truth_pairs": truth_pairs,
                    "target_present_truth_pairs": 0,
                    "missing_target_truth_pairs": truth_pairs,
                    "target_ingestion_coverage": 0.0 if truth_pairs else 1.0,
                    "survival": {
                        stage: {
                            "found_truth_pairs": 0,
                            "recall_of_all_truth": 0.0 if truth_pairs else 1.0,
                            "recall_of_ingested_truth": 1.0,
                        }
                        for stage in (
                            "after_layer2", "after_layer3", "after_layer4_before_threshold",
                            "after_layer4_before_top_k", "after_layer4_top_k",
                            "after_combined_pool", "after_final_top20"
                        )
                    },
                }
                stats["diagnostics"] = add_diagnostic_losses(diagnostics)
            return empty_df, stats

        logger.info("[%s] Blocking %d S1 entities against %d target (S2/S3) records...", country, n_s1, n_target)

        s1_ids = s1_df["entity_id"].to_list()
        s1_names = s1_df["name_clean"].to_list()
        s1_addrs = s1_df["address_clean"].to_list()
        
        target_ids = target_df["entity_id"].to_list()
        target_names = target_df["name_clean"].to_list()
        target_addrs = target_df["address_clean"].to_list()

        target_id_to_idx = {entity_id: idx for idx, entity_id in enumerate(target_ids)}
        true_target_indices = []
        total_truth_pairs = 0
        target_present_pairs = 0
        for s1_id in s1_ids:
            truth_ids = true_matches.get(s1_id, set()) if true_matches is not None else set()
            total_truth_pairs += len(truth_ids)
            indices = {target_id_to_idx[target_id] for target_id in truth_ids if target_id in target_id_to_idx}
            target_present_pairs += len(indices)
            true_target_indices.append(indices)
        diagnostic_counts = {
            "after_layer2": 0,
            "after_layer3": 0,
            "after_layer4_before_threshold": 0,
            "after_layer4_before_top_k": 0,
            "after_layer4_top_k": 0,
            "after_combined_pool": 0,
            "after_final_top20": 0,
        }

        candidates_map = defaultdict(lambda: defaultdict(lambda: {"tfidf": 0.0, "token_hit": 0, "num_hit": 0}))

        # Layer 2: Rare Token Index
        t_l2 = time.perf_counter()
        token_doc_freq = defaultdict(int)
        for name in target_names:
            if not name: continue
            for tok in set(name.split()):
                if len(tok) >= config.MIN_TOKEN_LEN and tok not in GLOBAL_STOP_WORDS:
                    token_doc_freq[tok] += 1

        max_freq = max(1, int(n_target * config.MAX_TOKEN_DOC_FREQ))
        valid_rare_tokens = {tok for tok, freq in token_doc_freq.items() if freq <= max_freq}

        target_token_index = defaultdict(list)
        for t_idx, name in enumerate(target_names):
            if not name: continue
            for tok in set(name.split()):
                if tok in valid_rare_tokens:
                    target_token_index[tok].append(t_idx)

        for s1_idx, name in enumerate(s1_names):
            if not name: continue
            for tok in set(name.split()):
                if tok in valid_rare_tokens:
                    for t_idx in target_token_index[tok][:100]:
                        candidates_map[s1_idx][t_idx]["token_hit"] += 1

        if true_matches is not None:
            diagnostic_counts["after_layer2"] = sum(
                len(indices & set(candidates_map.get(s1_idx, {})))
                for s1_idx, indices in enumerate(true_target_indices)
            )

        logger.info("[%s | Layer 2] Rare token index completed in %.2fs", country, time.perf_counter() - t_l2)

        # Layer 3: Numeric Anchor Blocking
        t_l3 = time.perf_counter()
        target_num_index = defaultdict(list)
        for t_idx, (name, addr) in enumerate(zip(target_names, target_addrs)):
            comb = f"{name} {addr}"
            nums = set(re.findall(r'\b\d+\b', comb))
            for num in nums:
                if len(num) >= 2:
                    target_num_index[num].append(t_idx)

        for s1_idx, (name, addr) in enumerate(zip(s1_names, s1_addrs)):
            comb = f"{name} {addr}"
            nums = set(re.findall(r'\b\d+\b', comb))
            for num in nums:
                if len(num) >= 2 and num in target_num_index:
                    for t_idx in target_num_index[num][:50]:
                        candidates_map[s1_idx][t_idx]["num_hit"] += 1

        if true_matches is not None:
            diagnostic_counts["after_layer3"] = sum(
                len(indices & set(candidates_map.get(s1_idx, {})))
                for s1_idx, indices in enumerate(true_target_indices)
            )

        logger.info("[%s | Layer 3] Numeric anchor blocking completed in %.2fs", country, time.perf_counter() - t_l3)

        # Layer 4: Character 3-gram inverted TF-IDF retrieval
        t_l4 = time.perf_counter()
        
        if n_target > 0 and n_s1 > 0:
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=config.TFIDF_NGRAM_RANGE,
                max_features=config.TFIDF_MAX_FEATURES,
                sublinear_tf=True
            )
            all_names_for_vocab = target_names + [n for n in s1_names if n]
            vectorizer.fit(all_names_for_vocab)
            # CSC columns are an inverted index: feature -> target posting list.
            # Keep the same char_wb 3-gram vocabulary and normalized TF-IDF weights,
            # but never construct a query-by-target similarity matrix.
            X_target_index = vectorizer.transform(target_names).tocsc()
            X_target_index.sum_duplicates()
            X_s1 = vectorizer.transform(s1_names).tocsr()

            scores = np.zeros(n_target, dtype=np.float64)
            touched = np.empty(n_target, dtype=X_target_index.indices.dtype)
            query_indptr = X_s1.indptr
            query_features = X_s1.indices
            query_weights = X_s1.data

            for s1_idx in range(n_s1):
                q_start, q_end = query_indptr[s1_idx:s1_idx + 2]
                if q_start == q_end:
                    continue

                reached_ids = accumulate_tfidf_scores(
                    X_target_index, query_features[q_start:q_end],
                    query_weights[q_start:q_end], scores, touched
                )
                if true_matches is not None:
                    truth_indices = true_target_indices[s1_idx]
                    diagnostic_counts["after_layer4_before_threshold"] += sum(
                        scores[t_idx] > 0 for t_idx in truth_indices
                    )
                    diagnostic_counts["after_layer4_before_top_k"] += sum(
                        scores[t_idx] >= config.TFIDF_MIN_SIMILARITY for t_idx in truth_indices
                    )
                # Ascending target order preserves the former unique() ordering,
                # including the input order used by argpartition for ties.
                candidate_ids = reached_ids[scores[reached_ids] >= config.TFIDF_MIN_SIMILARITY]
                if not len(candidate_ids):
                    scores[reached_ids] = 0.0
                    continue
                similarities = scores[candidate_ids]
                if len(similarities) > self.layer4_internal_top_k:
                    top = np.argpartition(similarities, -self.layer4_internal_top_k)[-self.layer4_internal_top_k:]
                    candidate_ids = candidate_ids[top]
                    similarities = similarities[top]

                if true_matches is not None:
                    diagnostic_counts["after_layer4_top_k"] += len(
                        true_target_indices[s1_idx] & set(map(int, candidate_ids))
                    )

                for t_idx, similarity in zip(candidate_ids, similarities):
                    candidates_map[s1_idx][int(t_idx)]["tfidf"] = max(
                        candidates_map[s1_idx][int(t_idx)].get("tfidf", 0.0),
                        float(similarity)
                    )
                scores[reached_ids] = 0.0

            del X_target_index, X_s1, scores, touched
            gc.collect()

        logger.info("[%s | Layer 4] Character 3-gram inverted retrieval completed in %.2fs", country, time.perf_counter() - t_l4)

        if true_matches is not None:
            diagnostic_counts["after_combined_pool"] = sum(
                len(indices & set(candidates_map.get(s1_idx, {})))
                for s1_idx, indices in enumerate(true_target_indices)
            )

        # Layer 5: Union, Scoring, Capping to TOP_K
        t_l5 = time.perf_counter()
        logger.info(
            "[%s | Layer 5] score = 2.0*tfidf + 1.2*min(token_hits,3) + 1.5*min(numeric_hits,2) "
            "+ %.2f*I(address_jaccard>=0.5); final TOP_K=%d",
            country, config.LAYER5_STRONG_ADDRESS_BONUS, self.top_k
        )
        pair_s1_ids, pair_target_ids, pair_scores, pair_layers = [], [], [], []
        l2_hits, l3_hits, l4_hits = [], [], []
        ranking_audit = {"lost_true": [], "boundary_displacer": [], "paired": []}

        for s1_idx in range(n_s1):
            s1_id = s1_ids[s1_idx]
            cand_dict = candidates_map.get(s1_idx, {})
            if not cand_dict:
                continue

            scored_cands = []
            s1_address_tokens = set((s1_addrs[s1_idx] or "").split())
            for t_idx, hits in cand_dict.items():
                tfidf_score = hits["tfidf"]
                token_hit = hits["token_hit"]
                num_hit = hits["num_hit"]
                address_jaccard = self._address_jaccard(
                    s1_address_tokens, set((target_addrs[t_idx] or "").split())
                )
                composite_score = (
                    (2.0 * tfidf_score)
                    + (1.2 * min(token_hit, 3))
                    + (1.5 * min(num_hit, 2))
                    + (config.LAYER5_STRONG_ADDRESS_BONUS if address_jaccard >= 0.5 else 0.0)
                )
                layers_count = (1 if tfidf_score > 0 else 0) + (1 if token_hit > 0 else 0) + (1 if num_hit > 0 else 0)
                scored_cands.append((t_idx, composite_score, layers_count))

            scored_cands.sort(key=lambda x: x[1], reverse=True)
            score_by_target = {t_idx: score for t_idx, score, _ in scored_cands}
            top_cands = scored_cands[:self.top_k]

            if true_matches is not None:
                truth_indices = true_target_indices[s1_idx]
                selected_indices = {t_idx for t_idx, _, _ in top_cands}
                diagnostic_counts["after_final_top20"] += len(truth_indices & selected_indices)
                lost_true = (truth_indices & set(cand_dict)) - selected_indices
                false_selected = [candidate for candidate in top_cands if candidate[0] not in truth_indices]
                if lost_true and false_selected:
                    boundary = min(false_selected, key=lambda candidate: candidate[1])
                    boundary_metrics = self._ranking_metrics(
                        s1_idx, boundary[0], boundary[1], cand_dict[boundary[0]],
                        s1_names, s1_addrs, target_names, target_addrs
                    )
                    for lost_idx in lost_true:
                        lost_metrics = self._ranking_metrics(
                            s1_idx, lost_idx, score_by_target[lost_idx], cand_dict[lost_idx],
                            s1_names, s1_addrs, target_names, target_addrs
                        )
                        ranking_audit["lost_true"].append(lost_metrics)
                        ranking_audit["boundary_displacer"].append(boundary_metrics)
                        ranking_audit["paired"].append({
                            "lost_exact_name_displaced_by_numeric_only": int(
                                lost_metrics["exact_name"] and boundary_metrics["numeric_only"]
                            ),
                            "lost_strong_address_displaced_by_numeric_only": int(
                                lost_metrics["strong_address"] and boundary_metrics["numeric_only"]
                            ),
                            "score_margin": boundary_metrics["score"] - lost_metrics["score"],
                        })

            for t_idx, score, layers in top_cands:
                pair_s1_ids.append(s1_id)
                pair_target_ids.append(target_ids[t_idx])
                pair_scores.append(score)
                pair_layers.append(layers)
                
                hits = cand_dict[t_idx]
                l2_hits.append(True if hits["token_hit"] > 0 else False)
                l3_hits.append(True if hits["num_hit"] > 0 else False)
                l4_hits.append(True if hits["tfidf"] > 0 else False)

        res_df = pl.DataFrame({
            "source1_entity_id": pl.Series(pair_s1_ids, dtype=pl.String),
            "candidate_entity_id": pl.Series(pair_target_ids, dtype=pl.String),
            "heuristic_score": pl.Series(pair_scores, dtype=pl.Float32),
            "layers_matched": pl.Series(pair_layers, dtype=pl.Int8),
            "l2": pl.Series(l2_hits, dtype=pl.Boolean),
            "l3": pl.Series(l3_hits, dtype=pl.Boolean),
            "l4": pl.Series(l4_hits, dtype=pl.Boolean)
        })

        elapsed_total = time.perf_counter() - t0
        avg_cands = len(res_df) / max(1, n_s1)
        rss_mb, peak_mb = self._get_memory_mb()

        logger.info(
            "[%s | Layer 5] Blocking complete in %.2fs | Candidates: %d (Avg %.1f/entity) | RAM: %.1f MB",
            country, elapsed_total, len(res_df), avg_cands, rss_mb
        )

        stats = {
            "country": country,
            "s1_entities": n_s1,
            "target_entities": n_target,
            "candidates_generated": len(res_df),
            "avg_candidates_per_entity": avg_cands,
            "time_sec": elapsed_total,
            "peak_ram_mb": peak_mb
        }
        if true_matches is not None:
            diagnostics = {
                "query_entities": n_s1,
                "queries_with_ground_truth_row": sum(s1_id in true_matches for s1_id in s1_ids),
                "truth_pairs": total_truth_pairs,
                "target_present_truth_pairs": target_present_pairs,
                "missing_target_truth_pairs": total_truth_pairs - target_present_pairs,
                "target_ingestion_coverage": target_present_pairs / total_truth_pairs if total_truth_pairs else 1.0,
                "survival": {}
            }
            for stage, found in diagnostic_counts.items():
                found = int(found)
                diagnostics["survival"][stage] = {
                    "found_truth_pairs": found,
                    "recall_of_all_truth": found / total_truth_pairs if total_truth_pairs else 1.0,
                    "recall_of_ingested_truth": found / target_present_pairs if target_present_pairs else 1.0,
                }
            stats["diagnostics"] = add_diagnostic_losses(diagnostics)
            stats["diagnostics"]["ranking_audit"] = self._summarize_ranking_audit(ranking_audit)
            logger.info("[%s | Diagnostics] %s", country, diagnostics)
        return res_df, stats

    @staticmethod
    def _address_jaccard(left: Set[str], right: Set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    @staticmethod
    def _ranking_metrics(
        s1_idx, target_idx, score, hits, s1_names, s1_addrs, target_names, target_addrs
    ) -> Dict[str, float]:
        s1_name = s1_names[s1_idx] or ""
        target_name = target_names[target_idx] or ""
        s1_addr_tokens = set((s1_addrs[s1_idx] or "").split())
        target_addr_tokens = set((target_addrs[target_idx] or "").split())
        address_jaccard = MultiLayerBlocker._address_jaccard(s1_addr_tokens, target_addr_tokens)
        return {
            "score": float(score),
            "tfidf": float(hits["tfidf"]),
            "token_hits": float(hits["token_hit"]),
            "numeric_hits": float(hits["num_hit"]),
            "exact_name": float(bool(s1_name) and s1_name == target_name),
            "address_jaccard": float(address_jaccard),
            "strong_address": float(address_jaccard >= 0.5),
            "numeric_only": float(
                hits["num_hit"] > 0 and hits["token_hit"] == 0 and hits["tfidf"] == 0
            ),
        }

    @staticmethod
    def _summarize_ranking_audit(audit: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
        summary = {}
        for group in ("lost_true", "boundary_displacer"):
            records = audit[group]
            summary[group] = {"count": len(records)}
            if records:
                for metric in records[0]:
                    values = np.asarray([record[metric] for record in records], dtype=np.float64)
                    summary[group][f"mean_{metric}"] = float(values.mean())
                    summary[group][f"median_{metric}"] = float(np.median(values))
        paired = audit["paired"]
        summary["paired"] = {"count": len(paired)}
        if paired:
            for metric in paired[0]:
                values = np.asarray([record[metric] for record in paired], dtype=np.float64)
                summary["paired"][f"sum_{metric}"] = float(values.sum())
                summary["paired"][f"mean_{metric}"] = float(values.mean())
        return summary

    def run_blocking_pipeline(self, mode: str = "train", country: Optional[str] = None) -> Dict[str, Any]:
        requested_country = country
        country_suffix = f"_{requested_country}" if requested_country else ""
        stage_name = f"stage_3_blocking_{mode}{country_suffix}"
        self.state_manager.mark_in_progress(stage_name)
        t_start = time.perf_counter()

        country_dirs = [d for d in config.CLEANED_DIR.iterdir() if d.is_dir()]
        if requested_country:
            country_dirs = [d for d in country_dirs if d.name.casefold() == requested_country.casefold()]
            if not country_dirs:
                raise FileNotFoundError(f"Country partition not found: {config.CLEANED_DIR / requested_country}")
        logger.info("Found %d dynamic country partitions under %s", len(country_dirs), config.CLEANED_DIR)

        all_candidate_dfs = []
        country_stats = {}
        evaluated_val_ids = set()
        zero_candidate_entities = 0

        for c_dir in country_dirs:
            country = c_dir.name
            s1_pattern = "val_s1_part_*.parquet" if mode == "val" else ("test_source1_part_*.parquet" if mode == "test" else "train_source1_part_*.parquet")
            search_dir = config.VAL_SPLIT_DIR / country if mode == "val" else c_dir
            
            s1_files = list(search_dir.glob(s1_pattern))
            if not s1_files:
                continue

            s1_df = pl.concat([pl.read_parquet(f) for f in s1_files])
            
            s2_files = list(c_dir.glob("test_source2_part_*.parquet" if mode == "test" else "train_source2_part_*.parquet"))
            s3_files = list(c_dir.glob("test_source3_part_*.parquet" if mode == "test" else "train_source3_part_*.parquet"))
            
            target_files = s2_files + s3_files
            if not target_files:
                continue

            target_df = pl.concat([pl.read_parquet(f) for f in target_files])

            true_matches = self._load_validation_truth(set(s1_df["entity_id"].to_list())) if mode == "val" else None
            cand_df, stats = self.block_country_partition(country, s1_df, target_df, true_matches=true_matches)
            zero_candidate_entities += len(s1_df) - cand_df["source1_entity_id"].n_unique()
            if mode == "val":
                evaluated_val_ids.update(s1_df["entity_id"].to_list())
            all_candidate_dfs.append(cand_df)
            country_stats[country] = stats

            del s1_df, target_df
            gc.collect()

        if all_candidate_dfs:
            final_cand_df = pl.concat(all_candidate_dfs)
        else:
            final_cand_df = pl.DataFrame({
                "source1_entity_id": pl.Series([], dtype=pl.String),
                "candidate_entity_id": pl.Series([], dtype=pl.String),
                "heuristic_score": pl.Series([], dtype=pl.Float32),
                "layers_matched": pl.Series([], dtype=pl.Int8),
                "l2": pl.Series([], dtype=pl.Boolean),
                "l3": pl.Series([], dtype=pl.Boolean),
                "l4": pl.Series([], dtype=pl.Boolean)
            })

        out_parquet = (
            config.BLOCKED_DIR / country_dirs[0].name / f"{mode}_candidates.parquet"
            if requested_country else config.BLOCKED_DIR / f"{mode}_candidates.parquet"
        )
        self._atomic_write_parquet(final_cand_df, out_parquet)
        artifact_name = f"{mode}_candidates_{country_dirs[0].name}" if requested_country else f"{mode}_candidates"
        self.state_manager.record_artifact(artifact_name, out_parquet, len(final_cand_df), meta=country_stats)

        if mode == "test":
            self._export_official_candidate_pairs_tsv(final_cand_df)

        recall_est = {"overall": 0.0, "layer2": 0.0, "layer2_3": 0.0, "layer2_3_4": 0.0}
        if mode == "val":
            recall_est = self._estimate_validation_recall(final_cand_df, evaluated_val_ids)
        diagnostics = self._combine_country_diagnostics(country_stats)

        elapsed = time.perf_counter() - t_start
        total_s1_entities = sum(int(stats.get("s1_entities", 0)) for stats in country_stats.values())
        self.state_manager.mark_completed(stage_name, meta={
            "total_candidates": len(final_cand_df),
            "time_sec": elapsed,
            "recall_estimate": recall_est["overall"],
            "per_layer_recall": recall_est,
            "zero_candidate_entities": zero_candidate_entities,
            "country_stats": country_stats,
            "diagnostics": diagnostics
        })

        logger.info(
            "[%s Blocking Complete] Total Candidates: %d | Time: %.2fs | Recall Est: %.2f%% | Peak RAM: %.1f MB",
            mode.upper(), len(final_cand_df), elapsed, recall_est["overall"] * 100.0, self.peak_memory_mb
        )

        return {
            "mode": mode,
            "total_candidates": len(final_cand_df),
            "recall_estimate": recall_est["overall"],
            "per_layer_recall": recall_est,
            "zero_candidate_entities": zero_candidate_entities,
            "avg_candidates_per_entity": len(final_cand_df) / max(1, total_s1_entities),
            "time_sec": elapsed,
            "peak_ram_mb": self.peak_memory_mb,
            "country_stats": country_stats,
            "diagnostics": diagnostics
        }

    @staticmethod
    def _combine_country_diagnostics(country_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        diagnostics = [stats["diagnostics"] for stats in country_stats.values() if "diagnostics" in stats]
        if not diagnostics:
            return {}
        combined = {
            "query_entities": sum(item["query_entities"] for item in diagnostics),
            "queries_with_ground_truth_row": sum(item["queries_with_ground_truth_row"] for item in diagnostics),
            "truth_pairs": sum(item["truth_pairs"] for item in diagnostics),
            "target_present_truth_pairs": sum(item["target_present_truth_pairs"] for item in diagnostics),
            "missing_target_truth_pairs": sum(item["missing_target_truth_pairs"] for item in diagnostics),
            "survival": {},
        }
        total = combined["truth_pairs"]
        present = combined["target_present_truth_pairs"]
        combined["target_ingestion_coverage"] = present / total if total else 1.0
        for stage in diagnostics[0]["survival"]:
            found = sum(item["survival"][stage]["found_truth_pairs"] for item in diagnostics)
            combined["survival"][stage] = {
                "found_truth_pairs": found,
                "recall_of_all_truth": found / total if total else 1.0,
                "recall_of_ingested_truth": found / present if present else 1.0,
            }
        if len(diagnostics) == 1 and "ranking_audit" in diagnostics[0]:
            combined["ranking_audit"] = diagnostics[0]["ranking_audit"]
        return add_diagnostic_losses(combined)

    @staticmethod
    def _load_validation_truth(query_ids: Set[str]) -> Dict[str, Set[str]]:
        truth = {}
        query_id_list = sorted(query_ids)
        for path in config.VAL_SPLIT_DIR.glob("val_ground_truth_part_*.parquet"):
            frame = pl.read_parquet(path).filter(pl.col("source1_entity_id").is_in(query_id_list))
            for s1_id, matches in zip(frame["source1_entity_id"], frame["matched_entity_ids"]):
                truth[s1_id] = {match for match in (matches or "").split(",") if match}
        return truth

    def _export_official_candidate_pairs_tsv(self, cand_df: pl.DataFrame):
        out_tsv = config.OUTPUT_DIR / "candidate_pairs.tsv"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        grouped = cand_df.group_by("source1_entity_id").agg(
            pl.col("candidate_entity_id").implode().map_elements(lambda ids: ",".join(ids), return_dtype=pl.String).alias("candidate_entity_ids")
        )
        
        test_s1_files = list(config.CLEANED_DIR.rglob("test_source1_part_*.parquet"))
        if test_s1_files:
            all_s1_df = pl.concat([pl.read_parquet(f).select("entity_id") for f in test_s1_files]).rename({"entity_id": "source1_entity_id"})
            merged = all_s1_df.join(grouped, on="source1_entity_id", how="left").with_columns(
                pl.col("candidate_entity_ids").fill_null("")
            )
        else:
            merged = grouped

        tmp_tsv = out_tsv.with_name("candidate_pairs.tsv.tmp")
        with open(tmp_tsv, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1, c_ids in zip(merged["source1_entity_id"], merged["candidate_entity_ids"]):
                f.write(f"{s1}\t{c_ids}\n")
        os.replace(tmp_tsv, out_tsv)
        logger.info("Exported official candidate pairs to %s (%d rows)", out_tsv, len(merged))

    def _estimate_validation_recall(self, val_cand_df: pl.DataFrame, query_ids: Optional[Set[str]] = None) -> Dict[str, float]:
        val_gt_files = list(config.VAL_SPLIT_DIR.glob("val_ground_truth_part_*.parquet"))
        if not val_gt_files:
            return {"overall": 0.0, "layer2": 0.0, "layer2_3": 0.0, "layer2_3_4": 0.0}
        
        val_gt = pl.concat([pl.read_parquet(f) for f in val_gt_files])
        if query_ids is not None:
            val_gt = val_gt.filter(pl.col("source1_entity_id").is_in(sorted(query_ids)))
        
        pairs_set = set()
        for s1, mids in zip(val_gt["source1_entity_id"], val_gt["matched_entity_ids"]):
            if mids and mids.strip():
                for m in mids.split(","):
                    pairs_set.add((s1, m))
                    
        if not pairs_set:
            return {"overall": 1.0, "layer2": 1.0, "layer2_3": 1.0, "layer2_3_4": 1.0}

        def recall(mask):
            subset = val_cand_df.filter(mask)
            found = len(pairs_set & set(zip(subset["source1_entity_id"], subset["candidate_entity_id"])))
            return found / len(pairs_set)

        l2 = pl.col("l2")
        l3 = pl.col("l3")
        l4 = pl.col("l4")
        return {
            "overall": recall(pl.lit(True)),
            "layer2": recall(l2),
            "layer2_3": recall(l2 | l3),
            "layer2_3_4": recall(l2 | l3 | l4),
        }
