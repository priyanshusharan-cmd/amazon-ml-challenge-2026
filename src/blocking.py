import gc
import multiprocessing as mp
import logging
import os
import re
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
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


def score_tfidf_query_sparse(index, features, weights):
    """Score only targets reached by this query's postings, without O(n_targets) scratch."""
    posting_ids, posting_scores = [], []
    for feature, weight in zip(features, weights):
        start, end = index.indptr[feature:feature + 2]
        posting_ids.append(index.indices[start:end])
        posting_scores.append(index.data[start:end] * weight)
    if not posting_ids:
        return np.empty(0, dtype=index.indices.dtype), np.empty(0, dtype=np.float64)
    all_ids = np.concatenate(posting_ids)
    all_scores = np.concatenate(posting_scores)
    unique_ids, inverse = np.unique(all_ids, return_inverse=True)
    scores = np.bincount(inverse, weights=all_scores, minlength=len(unique_ids))
    return unique_ids, scores


_GLOBAL_TARGET_INDEX = None
_GLOBAL_QUERY_INDEX = None
_GLOBAL_TRUE_TARGET_INDICES = None
_GLOBAL_S1_NAMES = None
_GLOBAL_TARGET_NAMES = None
_GLOBAL_N_TARGET = 0
_GLOBAL_INTERNAL_TOP_K = config.LAYER4_INTERNAL_TOP_K
_GLOBAL_MIN_SIM = config.TFIDF_MIN_SIMILARITY

_WORKER_SCORES = None
_WORKER_TOUCHED = None
_GLOBAL_DEFER_FALLBACK = False

# Leave room for the desktop and kernel before allocating Layer 4 workers.
LAYER4_RESERVED_RAM_BYTES = 3 * 1024 ** 3
LAYER4_RUNTIME_SAFETY_FLOOR_BYTES = int(1.5 * 1024 ** 3)
# Conservative private-memory allowance. Fork workers share imported modules
# and sparse indexes through COW; spawned workers need a larger process budget.
LAYER4_FORK_WORKER_BASE_RAM_BYTES = 64 * 1024 ** 2
LAYER4_SPAWN_WORKER_BASE_RAM_BYTES = 768 * 1024 ** 2
LAYER4_CANDIDATE_RESULT_BYTES = 96
LAYER4_MIN_QUERY_SCRATCH_BYTES = 16 * 1024 ** 2
LAYER4_CANDIDATE_MAP_BYTES = 236
LAYER4_MAX_QUERIES_PER_TASK = 64


def _layer4_batch_size(n_queries: int, workers: int) -> int:
    # Keep each worker's serialized candidate result bounded. The previous
    # minimum of 1,000 queries could make every concurrent worker return a
    # very large Python object graph at once.
    target_batch = max(1, (n_queries + workers * 8 - 1) // (workers * 8))
    return min(LAYER4_MAX_QUERIES_PER_TASK, target_batch)


def _bounded_ordered_pool_results(pool, tasks, max_pending: int):
    """Yield ordered results while keeping submitted work strictly bounded."""
    task_iter = iter(tasks)
    pending = deque()
    for _ in range(max_pending):
        try:
            pending.append(pool.apply_async(_layer4_worker_task, (next(task_iter),)))
        except StopIteration:
            break
    while pending:
        result = pending.popleft()
        yield result.get()
        try:
            pending.append(pool.apply_async(_layer4_worker_task, (next(task_iter),)))
        except StopIteration:
            pass


def select_layer4_worker_count(
    n_queries: int,
    n_targets: int,
    indices_dtype: np.dtype,
    requested_workers: Optional[int] = None,
    *,
    cpu_count: Optional[int] = None,
    available_ram_bytes: Optional[int] = None,
    total_ram_bytes: Optional[int] = None,
    internal_top_k: int = config.LAYER4_INTERNAL_TOP_K,
    worker_base_ram_bytes: int = LAYER4_FORK_WORKER_BASE_RAM_BYTES,
    max_query_postings: int = 0,
) -> Tuple[int, Dict[str, int]]:
    """Select a CPU- and RAM-safe Layer 4 pool size and return its audit data."""
    memory = psutil.virtual_memory()
    total_ram = int(total_ram_bytes if total_ram_bytes is not None else memory.total)
    available_ram = int(available_ram_bytes if available_ram_bytes is not None else memory.available)
    cpu_limit = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    if requested_workers is not None:
        cpu_limit = min(cpu_limit, max(1, int(requested_workers)))
    cpu_limit = min(cpu_limit, max(1, int(n_queries)))

    reserved_ram = LAYER4_RESERVED_RAM_BYTES
    usable_ram = max(0, available_ram - reserved_ram)

    is_fork = worker_base_ram_bytes == LAYER4_FORK_WORKER_BASE_RAM_BYTES
    if is_fork and total_ram <= 17 * 1024 ** 3 and usable_ram < 8 * 1024 ** 3:
        # On ~16 GB systems running fork, adaptively cap workers to 12 when safe
        # (gives ~10 workers at typical available RAM ~5.5 GiB, and up to 10-12 when safe).
        cpu_limit = min(cpu_limit, 12)

    # Scratch accounts ONLY for worker-private query allocations (never the shared TF-IDF matrix).
    # Under fork, the TF-IDF matrix is in the parent process and shared via COW.
    if is_fork:
        scratch_per_worker = 150 * 1024 ** 2
    else:
        # Spawn mode (Windows mmap path)
        scratch_per_worker = max(
            LAYER4_MIN_QUERY_SCRATCH_BYTES,
            int(max_query_postings) * 32 if max_query_postings else 32 * 1024 ** 2,
        )

    # Include two outstanding result batches per worker plus the parent batch
    # map. These bounds assume every query reaches the full internal top-K.
    selected = 1
    estimated_per_worker = worker_base_ram_bytes + scratch_per_worker
    for workers in range(cpu_limit, 0, -1):
        batch_size = _layer4_batch_size(int(n_queries), workers)
        result_bytes = batch_size * int(internal_top_k) * LAYER4_CANDIDATE_RESULT_BYTES
        estimate = worker_base_ram_bytes + scratch_per_worker + result_bytes
        outstanding_bytes = workers * result_bytes * 2
        parent_batch_bytes = min(int(n_queries), workers * LAYER4_MAX_QUERIES_PER_TASK) * int(internal_top_k) * LAYER4_CANDIDATE_MAP_BYTES
        total_estimate = workers * (worker_base_ram_bytes + scratch_per_worker) + outstanding_bytes + parent_batch_bytes
        if total_estimate <= usable_ram:
            selected = workers
            estimated_per_worker = estimate
            break
        estimated_per_worker = estimate

    if available_ram <= reserved_ram + estimated_per_worker:
        logger.warning(
            "Layer 4 available RAM is below the 3 GiB reserve plus one worker estimate; "
            "using a single worker. Consider freeing memory before continuing."
        )

    return selected, {
        "total_ram_bytes": total_ram,
        "available_ram_bytes": available_ram,
        "reserved_ram_bytes": reserved_ram,
        "estimated_worker_ram_bytes": estimated_per_worker,
        "estimated_total_worker_ram_bytes": total_estimate,
        "scratch_worker_ram_bytes": scratch_per_worker,
        "cpu_limit": cpu_limit,
    }


def _init_layer4_worker_scratch(n_target: int, indices_dtype: np.dtype):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    # Avoid allocating arrays proportional to the full 4M target universe in
    # every process. Query scoring uses compact posting-local arrays instead.
    gc.disable()  # Worker task objects are acyclic; keep inherited fork pages clean.


def _init_layer4_spawn_worker(target_paths, query_paths, target_shape, query_shape,
                              n_target, indices_dtype, internal_top_k, min_sim,
                              true_target_indices):
    """Attach read-only matrix arrays once in each Windows-spawned worker."""
    global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES
    global _GLOBAL_N_TARGET, _GLOBAL_INTERNAL_TOP_K, _GLOBAL_MIN_SIM
    global _GLOBAL_DEFER_FALLBACK
    target_arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in target_paths]
    query_arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in query_paths]
    _GLOBAL_TARGET_INDEX = SimpleNamespace(
        data=target_arrays[0], indices=target_arrays[1], indptr=target_arrays[2], shape=target_shape
    )
    _GLOBAL_QUERY_INDEX = SimpleNamespace(
        data=query_arrays[0], indices=query_arrays[1], indptr=query_arrays[2], shape=query_shape
    )
    _GLOBAL_TRUE_TARGET_INDICES = true_target_indices
    _GLOBAL_N_TARGET = n_target
    _GLOBAL_INTERNAL_TOP_K = internal_top_k
    _GLOBAL_MIN_SIM = min_sim
    _GLOBAL_DEFER_FALLBACK = True
    _init_layer4_worker_scratch(n_target, indices_dtype)


def _layer4_worker_task(batch_range: Tuple[int, int]):
    start_idx, end_idx = batch_range
    target_index = _GLOBAL_TARGET_INDEX
    query_index = _GLOBAL_QUERY_INDEX
    true_target_indices = _GLOBAL_TRUE_TARGET_INDICES
    internal_top_k = _GLOBAL_INTERNAL_TOP_K
    min_sim = _GLOBAL_MIN_SIM
    s1_names = _GLOBAL_S1_NAMES
    target_names = _GLOBAL_TARGET_NAMES

    query_indptr = query_index.indptr
    query_features = query_index.indices
    query_weights = query_index.data

    diag_counts = {
        "after_layer4_before_threshold": 0,
        "after_layer4_before_top_k": 0,
        "after_layer4_top_k": 0,
    }
    fallback_queries = []

    has_truth = true_target_indices is not None
    batch_results = []

    for s1_idx in range(start_idx, end_idx):
        q_start, q_end = query_indptr[s1_idx:s1_idx + 2]
        if q_start == q_end:
            continue

        reached_ids, reached_scores = score_tfidf_query_sparse(
            target_index, query_features[q_start:q_end], query_weights[q_start:q_end]
        )

        if has_truth:
            truth_indices = true_target_indices[s1_idx]
            for t_idx in truth_indices:
                pos = int(np.searchsorted(reached_ids, t_idx))
                if pos < len(reached_ids) and reached_ids[pos] == t_idx:
                    diag_counts["after_layer4_before_threshold"] += 1
                    if reached_scores[pos] >= min_sim:
                        diag_counts["after_layer4_before_top_k"] += 1

        mask = reached_scores >= min_sim
        candidate_ids = reached_ids[mask]
        similarities = reached_scores[mask]

        if len(candidate_ids) < getattr(config, "LAYER4_FALLBACK_THRESHOLD", 0) and _GLOBAL_DEFER_FALLBACK:
            fallback_queries.append(s1_idx)
        elif len(candidate_ids) < getattr(config, "LAYER4_FALLBACK_THRESHOLD", 0):
            from rapidfuzz import process, fuzz
            query_name = s1_names[s1_idx]
            if query_name and len(query_name) > 3:
                fallback_results = process.extract(
                    query_name, 
                    target_names, 
                    scorer=fuzz.token_set_ratio, 
                    limit=getattr(config, "LAYER4_FALLBACK_TOP_K", 10)
                )
                fallback_indices = np.array([match[2] for match in fallback_results if match[1] >= 60.0], dtype=np.int32)
                if len(fallback_indices) > 0:
                    merged = dict(zip(map(int, candidate_ids), map(float, similarities)))
                    for t_idx in fallback_indices:
                        merged.setdefault(int(t_idx), min_sim + 0.01)
                    candidate_ids = np.asarray(sorted(merged), dtype=target_index.indices.dtype)
                    similarities = np.asarray([merged[int(t)] for t in candidate_ids])

        if not len(candidate_ids):
            continue

        if len(similarities) > internal_top_k:
            top = np.argpartition(similarities, -internal_top_k)[-internal_top_k:]
            candidate_ids = candidate_ids[top]
            similarities = similarities[top]

        if has_truth:
            diag_counts["after_layer4_top_k"] += len(
                true_target_indices[s1_idx] & set(map(int, candidate_ids))
            )

        cands = [
            (int(t_idx), float(sim))
            for t_idx, sim in zip(candidate_ids, similarities)
        ]
        batch_results.append((s1_idx, cands))

    return batch_results, diag_counts, fallback_queries


def add_diagnostic_losses(diagnostics, top_k=config.TOP_K):
    """Add explicit loss buckets derived from the survival counters."""
    survival = diagnostics["survival"]
    found = lambda stage: survival[stage]["found_truth_pairs"]
    diagnostics["losses"] = {
        "ingestion_missing_truth_pairs": diagnostics["missing_target_truth_pairs"],
        "layer4_no_shared_gram_pairs": diagnostics["target_present_truth_pairs"] - found("after_layer4_before_threshold"),
        "layer4_threshold_loss_pairs": found("after_layer4_before_threshold") - found("after_layer4_before_top_k"),
        "layer4_top_k_loss_pairs": found("after_layer4_before_top_k") - found("after_layer4_top_k"),
        "combined_pool_missing_pairs": diagnostics["target_present_truth_pairs"] - found("after_combined_pool"),
        "layer5_ranking_loss_pairs": found("after_combined_pool") - found(f"after_final_top{top_k}"),
    }
    return diagnostics


class MultiLayerBlocker:
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        top_k: int = config.TOP_K,
        layer4_internal_top_k: int = config.LAYER4_INTERNAL_TOP_K,
        num_workers: Optional[int] = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if layer4_internal_top_k < top_k:
            raise ValueError("layer4_internal_top_k must be at least top_k")
        self.top_k = top_k
        self.layer4_internal_top_k = layer4_internal_top_k
        self.num_workers = num_workers or os.cpu_count() or 4
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

    def _sample_process_tree_memory(self) -> Tuple[float, float]:
        """Return process-tree USS and system available RAM in MiB."""
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except psutil.Error:
            pass
        uss_bytes = 0
        parent_rss = 0
        for proc in processes:
            try:
                info = proc.memory_full_info()
                uss_bytes += getattr(info, "uss", info.rss)
                if proc.pid == self.process.pid:
                    parent_rss = info.rss
            except psutil.Error:
                continue
        self.peak_memory_mb = max(self.peak_memory_mb, parent_rss / (1024 ** 2))
        return uss_bytes / (1024 ** 2), psutil.virtual_memory().available / (1024 ** 2)

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
        true_matches: Optional[Dict[str, Set[str]]] = None,
        output_parts_dir: Optional[Path] = None,
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Block a country using bounded query batches and a full-country TF-IDF index."""
        t0 = time.perf_counter()
        required = {"name_clean", "address_clean", "legal_suffix", "has_address"}
        if not required.issubset(s1_df.columns):
            s1_df = self.normalizer.normalize_polars_df(s1_df)
        if not required.issubset(target_df.columns):
            target_df = self.normalizer.normalize_polars_df(target_df)

        n_s1, n_target = len(s1_df), len(target_df)
        if not n_s1 or not n_target:
            empty = pl.DataFrame({
                "source1_entity_id": pl.Series([], dtype=pl.String),
                "candidate_entity_id": pl.Series([], dtype=pl.String),
                "heuristic_score": pl.Series([], dtype=pl.Float32),
                "layers_matched": pl.Series([], dtype=pl.Int8),
                "l2": pl.Series([], dtype=pl.Boolean),
                "l3": pl.Series([], dtype=pl.Boolean),
                "l4": pl.Series([], dtype=pl.Boolean),
            })
            stats = {"country": country, "s1_entities": n_s1, "target_entities": n_target,
                     "candidates_generated": 0, "avg_candidates_per_entity": 0.0,
                     "queries_with_candidates": 0, "time_sec": 0.0,
                     "peak_ram_mb": self.peak_memory_mb}
            if true_matches is not None:
                truth_pairs = sum(len(true_matches.get(eid, set())) for eid in s1_df["entity_id"])
                stages = ("after_layer2", "after_layer3", "after_layer4_before_threshold",
                          "after_layer4_before_top_k", "after_layer4_top_k", "after_combined_pool",
                          f"after_final_top{self.top_k}")
                diagnostics = {
                    "query_entities": n_s1,
                    "queries_with_ground_truth_row": sum(eid in true_matches for eid in s1_df["entity_id"]),
                    "truth_pairs": truth_pairs, "target_present_truth_pairs": 0,
                    "missing_target_truth_pairs": truth_pairs,
                    "target_ingestion_coverage": 0.0 if truth_pairs else 1.0,
                    "survival": {
                        stage: {"found_truth_pairs": 0,
                                "recall_of_all_truth": 0.0 if truth_pairs else 1.0,
                                "recall_of_ingested_truth": 1.0}
                        for stage in stages
                    },
                }
                stats["diagnostics"] = add_diagnostic_losses(diagnostics, self.top_k)
            return empty, stats

        s1_ids = s1_df["entity_id"].to_list()
        s1_names = s1_df["name_clean"].to_list()
        s1_addrs = s1_df["address_clean"].to_list()
        s1_suffixes = s1_df["legal_suffix"].to_list()
        target_ids = target_df["entity_id"].to_list()
        target_names = target_df["name_clean"].to_list()
        target_addrs = target_df["address_clean"].to_list()
        target_suffixes = target_df["legal_suffix"].to_list()

        true_target_indices = []
        total_truth_pairs = target_present_pairs = 0
        if true_matches is not None:
            target_id_to_idx = {entity_id: idx for idx, entity_id in enumerate(target_ids)}
            for s1_id in s1_ids:
                truth_ids = true_matches.get(s1_id, set())
                total_truth_pairs += len(truth_ids)
                indices = {target_id_to_idx[t] for t in truth_ids if t in target_id_to_idx}
                target_present_pairs += len(indices)
                true_target_indices.append(indices)
            del target_id_to_idx

        diag = {k: 0 for k in (
            "after_layer2", "after_layer3", "after_layer4_before_threshold",
            "after_layer4_before_top_k", "after_layer4_top_k", "after_combined_pool",
            f"after_final_top{self.top_k}",
        )}

        # Build reusable target indexes once; query-side candidate maps exist
        # only for one bounded batch at a time.
        t_l2 = time.perf_counter()
        token_doc_freq = defaultdict(int)
        for name in target_names:
            if name:
                for tok in set(name.split()):
                    if len(tok) >= config.MIN_TOKEN_LEN and tok not in GLOBAL_STOP_WORDS:
                        token_doc_freq[tok] += 1
        max_freq = max(1, int(n_target * config.MAX_TOKEN_DOC_FREQ))
        valid_rare_tokens = {tok for tok, freq in token_doc_freq.items() if freq <= max_freq}
        target_token_index = defaultdict(list)
        for t_idx, name in enumerate(target_names):
            if name:
                for tok in set(name.split()):
                    if tok in valid_rare_tokens:
                        target_token_index[tok].append(t_idx)
        del token_doc_freq, valid_rare_tokens
        logger.info("[%s | Layer 2] Rare token index completed in %.2fs", country, time.perf_counter() - t_l2)

        t_l3 = time.perf_counter()
        target_num_index = defaultdict(list)
        for t_idx, (name, addr) in enumerate(zip(target_names, target_addrs)):
            for num in set(re.findall(r'\b\d+\b', f"{name} {addr}")):
                if len(num) >= 2:
                    target_num_index[num].append(t_idx)
        logger.info("[%s | Layer 3] Numeric anchor index completed in %.2fs", country, time.perf_counter() - t_l3)

        t_l4 = time.perf_counter()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=config.TFIDF_NGRAM_RANGE,
            max_features=config.TFIDF_MAX_FEATURES, sublinear_tf=True,
        )
        all_names_for_vocab = target_names + [n for n in s1_names if n]
        vectorizer.fit(all_names_for_vocab)
        X_target_index = vectorizer.transform(target_names).tocsc()
        X_target_index.sum_duplicates()
        X_s1 = vectorizer.transform(s1_names).tocsr()
        del vectorizer, all_names_for_vocab
        gc.collect()

        posting_lengths = np.diff(X_target_index.indptr)
        sample_queries = np.linspace(0, n_s1 - 1, num=min(n_s1, 512), dtype=np.int64)
        max_query_postings = 0
        for query_idx in sample_queries:
            start, end = X_s1.indptr[query_idx:query_idx + 2]
            if start != end:
                max_query_postings = max(
                    max_query_postings,
                    int(posting_lengths[X_s1.indices[start:end]].sum()),
                )
        del posting_lengths, sample_queries
        gc.collect()

        is_windows = sys.platform == "win32"
        parallel_mode = "spawn+mmap" if is_windows else "fork"
        num_workers, memory_plan = select_layer4_worker_count(
            n_s1, n_target, X_target_index.indices.dtype,
            requested_workers=self.num_workers,
            internal_top_k=self.layer4_internal_top_k,
            max_query_postings=max_query_postings,
            worker_base_ram_bytes=(LAYER4_SPAWN_WORKER_BASE_RAM_BYTES if is_windows
                                   else LAYER4_FORK_WORKER_BASE_RAM_BYTES),
        )
        if not is_windows:
            num_workers = min(max(1, int(self.num_workers)), max(1, n_s1))
            forced_msg = " (FORCED: adaptive RAM cap disabled)"
        else:
            forced_msg = ""
        logger.info(
            "[%s | Layer 4] RAM plan: total=%.2f GiB, available=%.2f GiB, reserved=%.2f GiB, "
            "estimated/worker=%.3f GiB, estimated workers+buffers=%.2f GiB, "
            "workers=%d/%d, mode=%s%s, query_batch=%d",
            country, memory_plan["total_ram_bytes"] / 1024**3,
            memory_plan["available_ram_bytes"] / 1024**3,
            memory_plan["reserved_ram_bytes"] / 1024**3,
            memory_plan["estimated_worker_ram_bytes"] / 1024**3,
            memory_plan["estimated_total_worker_ram_bytes"] / 1024**3,
            num_workers, memory_plan["cpu_limit"], parallel_mode, forced_msg, LAYER4_MAX_QUERIES_PER_TASK,
        )

        global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES
        global _GLOBAL_N_TARGET, _GLOBAL_INTERNAL_TOP_K, _GLOBAL_MIN_SIM
        global _GLOBAL_S1_NAMES, _GLOBAL_TARGET_NAMES, _GLOBAL_DEFER_FALLBACK
        _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX = X_target_index, X_s1
        _GLOBAL_TRUE_TARGET_INDICES = true_target_indices if true_matches is not None else None
        _GLOBAL_N_TARGET = n_target
        _GLOBAL_INTERNAL_TOP_K = self.layer4_internal_top_k
        _GLOBAL_MIN_SIM = config.TFIDF_MIN_SIMILARITY
        _GLOBAL_S1_NAMES, _GLOBAL_TARGET_NAMES = s1_names, target_names
        _GLOBAL_DEFER_FALLBACK = True

        # Final output is at most TOP_K per query. Flush each ranked batch to
        # disk so Python lists/DataFrames from earlier batches are released.
        output_temp = None if output_parts_dir is not None else tempfile.TemporaryDirectory(
            prefix="amazon_ml_blocked_parts_"
        )
        output_part_dir = Path(output_parts_dir) if output_parts_dir is not None else Path(output_temp.name)
        output_part_dir.mkdir(parents=True, exist_ok=True)
        for stale_part in output_part_dir.glob("part_*.parquet"):
            stale_part.unlink()
        output_part_count = 0
        total_output_rows = 0
        queries_with_candidates = 0
        peak_tree_uss_mb = 0.0
        min_available_ram_mb = float("inf")
        ranking_audit = {"lost_true": [], "boundary_displacer": [], "paired": []}
        pool = None
        scratch_context = None
        try:
            if is_windows:
                scratch_context = tempfile.TemporaryDirectory(prefix="amazon_ml_layer4_")
                scratch_dir = Path(scratch_context.name)
                target_paths, query_paths = [], []
                for label, matrix, paths in (("target", X_target_index, target_paths), ("query", X_s1, query_paths)):
                    for field in ("data", "indices", "indptr"):
                        path = scratch_dir / f"{label}_{field}.npy"
                        np.save(path, getattr(matrix, field), allow_pickle=False)
                        paths.append(str(path))
                ctx = mp.get_context("spawn")
            else:
                try:
                    ctx = mp.get_context("fork")
                except Exception as exc:
                    raise RuntimeError("Linux/macOS fork Copy-on-Write mode is required.") from exc

            def make_pool(worker_count):
                if is_windows:
                    return ctx.Pool(
                        processes=worker_count, initializer=_init_layer4_spawn_worker,
                        initargs=(target_paths, query_paths, X_target_index.shape, X_s1.shape,
                                  n_target, X_target_index.indices.dtype, self.layer4_internal_top_k,
                                  config.TFIDF_MIN_SIMILARITY,
                                  true_target_indices if true_matches is not None else None),
                    )
                return ctx.Pool(
                    processes=worker_count, initializer=_init_layer4_worker_scratch,
                    initargs=(n_target, X_target_index.indices.dtype),
                )

            pool = make_pool(num_workers)
            q_start = 0
            while q_start < n_s1:
                q_end = min(q_start + num_workers * LAYER4_MAX_QUERIES_PER_TASK, n_s1)
                candidates = defaultdict(lambda: defaultdict(lambda: {
                    "tfidf": 0.0, "token_hit": 0, "num_hit": 0,
                }))
                # Layers 2 and 3 for this batch.
                for s1_idx in range(q_start, q_end):
                    name = s1_names[s1_idx]
                    if name:
                        for tok in set(name.split()):
                            for t_idx in target_token_index.get(tok, ())[:100]:
                                candidates[s1_idx][t_idx]["token_hit"] += 1
                if true_matches is not None:
                    diag["after_layer2"] += sum(
                        len(true_target_indices[i] & set(candidates.get(i, {})))
                        for i in range(q_start, q_end)
                    )
                for s1_idx in range(q_start, q_end):
                    name, addr = s1_names[s1_idx], s1_addrs[s1_idx]
                    for num in set(re.findall(r'\b\d+\b', f"{name} {addr}")):
                        if len(num) >= 2:
                            for t_idx in target_num_index.get(num, ())[:50]:
                                candidates[s1_idx][t_idx]["num_hit"] += 1
                if true_matches is not None:
                    diag["after_layer3"] += sum(
                        len(true_target_indices[i] & set(candidates.get(i, {})))
                        for i in range(q_start, q_end)
                    )

                tasks = [(i, min(i + LAYER4_MAX_QUERIES_PER_TASK, q_end))
                         for i in range(q_start, q_end, LAYER4_MAX_QUERIES_PER_TASK)]
                for batch_candidates, batch_diag, fallback_queries in _bounded_ordered_pool_results(
                    pool, tasks, max_pending=max(1, num_workers * 2)
                ):
                    tree_uss, available_mb = self._sample_process_tree_memory()
                    peak_tree_uss_mb = max(peak_tree_uss_mb, tree_uss)
                    min_available_ram_mb = min(min_available_ram_mb, available_mb)
                    if true_matches is not None:
                        for key in ("after_layer4_before_threshold", "after_layer4_before_top_k", "after_layer4_top_k"):
                            diag[key] += batch_diag[key]
                    by_query = dict(batch_candidates)
                    fallback_set = set(fallback_queries)
                    for s1_idx in sorted(set(by_query) | fallback_set):
                        tfidf_cands = dict(by_query.get(s1_idx, []))
                        base_ids = set(tfidf_cands)
                        if s1_idx in fallback_set:
                            query_name = s1_names[s1_idx]
                            if query_name and len(query_name) > 3:
                                from rapidfuzz import process, fuzz
                                matches = process.extract(
                                    query_name, target_names, scorer=fuzz.token_set_ratio,
                                    score_cutoff=60.0,
                                    limit=getattr(config, "LAYER4_FALLBACK_TOP_K", 10),
                                )
                                fallback_ids = {int(m[2]) for m in matches if m[1] >= 60.0}
                                for t_idx in sorted(fallback_ids):
                                    tfidf_cands.setdefault(t_idx, config.TFIDF_MIN_SIMILARITY + 0.01)
                                if true_matches is not None:
                                    diag["after_layer4_top_k"] += len(
                                        true_target_indices[s1_idx] & (fallback_ids - base_ids)
                                    )
                                tfidf_cands = dict(sorted(tfidf_cands.items()))
                        if len(tfidf_cands) > self.layer4_internal_top_k:
                            ids = sorted(tfidf_cands)
                            vals = np.asarray([tfidf_cands[t] for t in ids])
                            keep = np.argpartition(vals, -self.layer4_internal_top_k)[-self.layer4_internal_top_k:]
                            tfidf_cands = {ids[i]: float(vals[i]) for i in keep}
                        for t_idx, score in tfidf_cands.items():
                            candidates[s1_idx][t_idx]["tfidf"] = max(
                                candidates[s1_idx][t_idx]["tfidf"], score
                            )

                if true_matches is not None:
                    diag["after_combined_pool"] += sum(
                        len(true_target_indices[i] & set(candidates.get(i, {})))
                        for i in range(q_start, q_end)
                    )

                out = {"source1_entity_id": [], "candidate_entity_id": [], "heuristic_score": [],
                       "layers_matched": [], "l2": [], "l3": [], "l4": []}
                for s1_idx in range(q_start, q_end):
                    cand_dict = candidates.get(s1_idx, {})
                    if not cand_dict:
                        continue
                    s1_address_tokens = set((s1_addrs[s1_idx] or "").split())
                    scored = []
                    for t_idx, hits in cand_dict.items():
                        address_jaccard = self._address_jaccard(
                            s1_address_tokens, set((target_addrs[t_idx] or "").split())
                        )
                        score = (2.0 * hits["tfidf"] + min(hits["token_hit"], 3)
                                 + 1.5 * min(hits["num_hit"], 2)
                                 + config.LAYER5_ADDRESS_JACCARD_WEIGHT * address_jaccard)
                        layers = int(hits["tfidf"] > 0) + int(hits["token_hit"] > 0) + int(hits["num_hit"] > 0)
                        scored.append((t_idx, score, layers))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    score_by_target = {t: score for t, score, _ in scored}
                    selected = scored[:self.top_k]
                    if selected:
                        queries_with_candidates += 1
                    if true_matches is not None:
                        truths = true_target_indices[s1_idx]
                        selected_ids = {t for t, _, _ in selected}
                        diag[f"after_final_top{self.top_k}"] += len(truths & selected_ids)
                        lost = (truths & set(cand_dict)) - selected_ids
                        false_selected = [item for item in selected if item[0] not in truths]
                        if lost and false_selected:
                            boundary = min(false_selected, key=lambda item: item[1])
                            boundary_metrics = self._ranking_metrics(
                                s1_idx, boundary[0], boundary[1], cand_dict[boundary[0]],
                                s1_names, s1_addrs, s1_suffixes, target_names, target_addrs, target_suffixes,
                            )
                            for lost_idx in lost:
                                lost_metrics = self._ranking_metrics(
                                    s1_idx, lost_idx, score_by_target[lost_idx], cand_dict[lost_idx],
                                    s1_names, s1_addrs, s1_suffixes, target_names, target_addrs, target_suffixes,
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
                    for t_idx, score, layers in selected:
                        out["source1_entity_id"].append(s1_ids[s1_idx])
                        out["candidate_entity_id"].append(target_ids[t_idx])
                        out["heuristic_score"].append(score)
                        out["layers_matched"].append(layers)
                        hits = cand_dict[t_idx]
                        out["l2"].append(hits["token_hit"] > 0)
                        out["l3"].append(hits["num_hit"] > 0)
                        out["l4"].append(hits["tfidf"] > 0)
                if out["source1_entity_id"]:
                    batch_frame = pl.DataFrame({
                        "source1_entity_id": pl.Series(out["source1_entity_id"], dtype=pl.String),
                        "candidate_entity_id": pl.Series(out["candidate_entity_id"], dtype=pl.String),
                        "heuristic_score": pl.Series(out["heuristic_score"], dtype=pl.Float32),
                        "layers_matched": pl.Series(out["layers_matched"], dtype=pl.Int8),
                        "l2": pl.Series(out["l2"], dtype=pl.Boolean),
                        "l3": pl.Series(out["l3"], dtype=pl.Boolean),
                        "l4": pl.Series(out["l4"], dtype=pl.Boolean),
                    })
                    batch_frame.write_parquet(output_part_dir / f"part_{output_part_count:06d}.parquet",
                                              compression="snappy")
                    output_part_count += 1
                    total_output_rows += len(batch_frame)
                    del batch_frame
                del candidates, out
                gc.collect()
                q_start = q_end
                if is_windows and q_start < n_s1:
                    current_available = int(psutil.virtual_memory().available)
                    # When workers are already running, their private memory is already resident.
                    # Only re-evaluate pool downscaling if available RAM falls below the runtime safety floor.
                    if current_available < LAYER4_RUNTIME_SAFETY_FLOOR_BYTES:
                        releasable = num_workers * memory_plan["estimated_worker_ram_bytes"]
                        effective_available = current_available + releasable
                        safer_workers, _ = select_layer4_worker_count(
                            n_s1 - q_start, n_target, X_target_index.indices.dtype,
                            requested_workers=num_workers,
                            internal_top_k=self.layer4_internal_top_k,
                            max_query_postings=max_query_postings,
                            available_ram_bytes=effective_available,
                            total_ram_bytes=memory_plan["total_ram_bytes"],
                            worker_base_ram_bytes=(LAYER4_SPAWN_WORKER_BASE_RAM_BYTES if is_windows
                                                   else LAYER4_FORK_WORKER_BASE_RAM_BYTES),
                        )
                        if safer_workers < num_workers:
                            logger.warning(
                                "[%s | Layer 4] Available RAM fell (%.2f GiB); scaling workers down from %d to %d",
                                country, current_available / 1024**3, num_workers, safer_workers,
                            )
                            pool.close()
                            pool.join()
                            num_workers = safer_workers
                            pool = make_pool(num_workers)
        except Exception:
            if output_temp is not None:
                output_temp.cleanup()
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            if scratch_context is not None:
                scratch_context.cleanup()
            _GLOBAL_TARGET_INDEX = _GLOBAL_QUERY_INDEX = _GLOBAL_TRUE_TARGET_INDICES = None
            _GLOBAL_S1_NAMES = _GLOBAL_TARGET_NAMES = None
            _GLOBAL_DEFER_FALLBACK = False
            del X_target_index, X_s1
            gc.collect()

        empty_result = pl.DataFrame({
            "source1_entity_id": pl.Series([], dtype=pl.String),
            "candidate_entity_id": pl.Series([], dtype=pl.String),
            "heuristic_score": pl.Series([], dtype=pl.Float32),
            "layers_matched": pl.Series([], dtype=pl.Int8),
            "l2": pl.Series([], dtype=pl.Boolean), "l3": pl.Series([], dtype=pl.Boolean),
            "l4": pl.Series([], dtype=pl.Boolean),
        })
        result = (pl.scan_parquet(str(output_part_dir / "part_*.parquet"))
                  .collect(engine="streaming") if output_part_count and output_temp is not None else empty_result)
        if output_temp is not None:
            output_temp.cleanup()
        del target_token_index, target_num_index
        elapsed = time.perf_counter() - t0
        _, peak = self._get_memory_mb()
        stats = {
            "country": country, "s1_entities": n_s1, "target_entities": n_target,
            "candidates_generated": total_output_rows,
            "avg_candidates_per_entity": total_output_rows / max(1, n_s1),
            "queries_with_candidates": queries_with_candidates,
            "time_sec": elapsed, "peak_ram_mb": peak,
            "peak_process_tree_uss_mb": peak_tree_uss_mb,
            "minimum_available_ram_mb": min_available_ram_mb if min_available_ram_mb != float("inf") else None,
            "selected_workers": num_workers,
            "estimated_max_query_postings": max_query_postings,
        }
        if true_matches is not None:
            diagnostics = {
                "query_entities": n_s1,
                "queries_with_ground_truth_row": sum(s in true_matches for s in s1_ids),
                "truth_pairs": total_truth_pairs,
                "target_present_truth_pairs": target_present_pairs,
                "missing_target_truth_pairs": total_truth_pairs - target_present_pairs,
                "target_ingestion_coverage": target_present_pairs / total_truth_pairs if total_truth_pairs else 1.0,
                "survival": {},
            }
            for stage, found in diag.items():
                diagnostics["survival"][stage] = {
                    "found_truth_pairs": int(found),
                    "recall_of_all_truth": found / total_truth_pairs if total_truth_pairs else 1.0,
                    "recall_of_ingested_truth": found / target_present_pairs if target_present_pairs else 1.0,
                }
            stats["diagnostics"] = add_diagnostic_losses(diagnostics, self.top_k)
            stats["diagnostics"]["ranking_audit"] = self._summarize_ranking_audit(ranking_audit)
        logger.info("[%s | Layer 5] Batch blocking complete: %d final candidates in %.2fs", country, total_output_rows, elapsed)
        return result, stats

    @staticmethod
    def _address_jaccard(left: Set[str], right: Set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    @staticmethod
    def _ranking_metrics(
        s1_idx, target_idx, score, hits,
        s1_names, s1_addrs, s1_suffixes,
        target_names, target_addrs, target_suffixes
    ) -> Dict[str, float]:
        s1_name = s1_names[s1_idx] or ""
        target_name = target_names[target_idx] or ""
        s1_name_tokens = set(s1_name.split())
        target_name_tokens = set(target_name.split())
        s1_addr_tokens = set((s1_addrs[s1_idx] or "").split())
        target_addr_tokens = set((target_addrs[target_idx] or "").split())
        address_jaccard = MultiLayerBlocker._address_jaccard(s1_addr_tokens, target_addr_tokens)
        name_jaccard = MultiLayerBlocker._address_jaccard(s1_name_tokens, target_name_tokens)
        s1_suffix = s1_suffixes[s1_idx] or "none"
        target_suffix = target_suffixes[target_idx] or "none"
        return {
            "score": float(score),
            "tfidf": float(hits["tfidf"]),
            "token_hits": float(hits["token_hit"]),
            "numeric_hits": float(hits["num_hit"]),
            "exact_name": float(bool(s1_name) and s1_name == target_name),
            "name_token_jaccard": float(name_jaccard),
            "address_jaccard": float(address_jaccard),
            "strong_address": float(address_jaccard >= 0.5),
            "address_last_token_match": float(
                bool(s1_addr_tokens) and bool(target_addr_tokens)
                and (s1_addrs[s1_idx] or "").split()[-1] == (target_addrs[target_idx] or "").split()[-1]
            ),
            "legal_suffix_match": float(
                s1_suffix != "none" and target_suffix != "none" and s1_suffix == target_suffix
            ),
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

        country_stats = {}
        evaluated_val_ids = set()
        zero_candidate_entities = 0
        total_candidates = 0
        batch_parts_root = config.BLOCKED_DIR / ".batch_parts" / mode
        if batch_parts_root.exists():
            for stale_part in batch_parts_root.glob("*/*.parquet"):
                stale_part.unlink()

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
            country_parts = batch_parts_root / country
            cand_df, stats = self.block_country_partition(
                country, s1_df, target_df, true_matches=true_matches,
                output_parts_dir=country_parts,
            )
            total_candidates += stats["candidates_generated"]
            zero_candidate_entities += len(s1_df) - int(stats["queries_with_candidates"] or 0)
            if mode == "val":
                evaluated_val_ids.update(s1_df["entity_id"].to_list())
            country_stats[country] = stats

            del s1_df, target_df
            gc.collect()

        empty_candidate_df = pl.DataFrame({
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
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        part_files = (
            sorted((batch_parts_root / country_dirs[0].name).glob("part_*.parquet"))
            if requested_country else sorted(batch_parts_root.glob("*/*.parquet"))
        )
        if part_files:
            tmp_out = out_parquet.with_name(f"{out_parquet.name}.tmp")
            pl.concat([pl.scan_parquet(str(path)) for path in part_files], how="vertical").sink_parquet(
                tmp_out, compression="snappy", maintain_order=True
            )
            os.replace(tmp_out, out_parquet)
            for part_file in part_files:
                part_file.unlink()
        else:
            self._atomic_write_parquet(empty_candidate_df, out_parquet)
        artifact_name = f"{mode}_candidates_{country_dirs[0].name}" if requested_country else f"{mode}_candidates"
        self.state_manager.record_artifact(artifact_name, out_parquet, total_candidates, meta=country_stats)

        if mode == "test":
            self._export_official_candidate_pairs_tsv(out_parquet)

        recall_est = {"overall": 0.0, "layer2": 0.0, "layer2_3": 0.0, "layer2_3_4": 0.0}
        if mode == "val":
            recall_est = self._estimate_validation_recall(out_parquet, evaluated_val_ids)
        diagnostics = self._combine_country_diagnostics(country_stats)

        elapsed = time.perf_counter() - t_start
        total_s1_entities = sum(int(stats.get("s1_entities", 0)) for stats in country_stats.values())
        self.state_manager.mark_completed(stage_name, meta={
            "total_candidates": total_candidates,
            "time_sec": elapsed,
            "recall_estimate": recall_est["overall"],
            "per_layer_recall": recall_est,
            "zero_candidate_entities": zero_candidate_entities,
            "country_stats": country_stats,
            "diagnostics": diagnostics
        })

        logger.info(
            "[%s Blocking Complete] Total Candidates: %d | Time: %.2fs | Recall Est: %.2f%% | Peak RAM: %.1f MB",
            mode.upper(), total_candidates, elapsed, recall_est["overall"] * 100.0, self.peak_memory_mb
        )

        return {
            "mode": mode,
            "total_candidates": total_candidates,
            "recall_estimate": recall_est["overall"],
            "per_layer_recall": recall_est,
            "zero_candidate_entities": zero_candidate_entities,
            "avg_candidates_per_entity": total_candidates / max(1, total_s1_entities),
            "time_sec": elapsed,
            "peak_ram_mb": self.peak_memory_mb,
            "country_stats": country_stats,
            "diagnostics": diagnostics
        }

    def _combine_country_diagnostics(self, country_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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
        return add_diagnostic_losses(combined, self.top_k)

    @staticmethod
    def _load_validation_truth(query_ids: Set[str]) -> Dict[str, Set[str]]:
        truth = {}
        query_id_list = sorted(query_ids)
        for path in config.VAL_SPLIT_DIR.glob("val_ground_truth_part_*.parquet"):
            frame = pl.read_parquet(path).filter(pl.col("source1_entity_id").is_in(query_id_list))
            for s1_id, matches in zip(frame["source1_entity_id"], frame["matched_entity_ids"]):
                truth[s1_id] = {match for match in (matches or "").split(",") if match}
        return truth

    def _export_official_candidate_pairs_tsv(self, cand_df: Any):
        out_tsv = config.OUTPUT_DIR / "candidate_pairs.tsv"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cand_lf = pl.scan_parquet(cand_df) if isinstance(cand_df, (str, Path)) else cand_df.lazy()
        grouped = cand_lf.group_by("source1_entity_id", maintain_order=True).agg(
            pl.col("candidate_entity_id").implode().list.join(",").alias("candidate_entity_ids")
        )

        test_s1_files = list(config.CLEANED_DIR.rglob("test_source1_part_*.parquet"))
        if test_s1_files:
            all_s1_lf = pl.concat([pl.scan_parquet(f).select("entity_id") for f in test_s1_files]).rename({"entity_id": "source1_entity_id"})
            merged = all_s1_lf.join(grouped, on="source1_entity_id", how="left").with_columns(
                pl.col("candidate_entity_ids").fill_null("")
            )
        else:
            merged = grouped

        tmp_tsv = out_tsv.with_name("candidate_pairs.tsv.tmp")
        merged.sink_csv(tmp_tsv, separator="\t")
        os.replace(tmp_tsv, out_tsv)
        logger.info("Exported official candidate pairs to %s", out_tsv)

    def _estimate_validation_recall(self, val_cand_df: Any, query_ids: Optional[Set[str]] = None) -> Dict[str, float]:
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

        truth_df = pl.DataFrame({
            "source1_entity_id": [pair[0] for pair in pairs_set],
            "candidate_entity_id": [pair[1] for pair in pairs_set],
        })
        cand_lf = pl.scan_parquet(val_cand_df) if isinstance(val_cand_df, (str, Path)) else val_cand_df.lazy()
        found = (
            cand_lf.join(truth_df.lazy(), on=["source1_entity_id", "candidate_entity_id"], how="inner")
            .select(
                pl.len().alias("overall"),
                pl.col("l2").cast(pl.UInt64).sum().alias("layer2"),
                (pl.col("l2") | pl.col("l3")).cast(pl.UInt64).sum().alias("layer2_3"),
                (pl.col("l2") | pl.col("l3") | pl.col("l4")).cast(pl.UInt64).sum().alias("layer2_3_4"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
        return {key: value / len(pairs_set) for key, value in found.items()}
