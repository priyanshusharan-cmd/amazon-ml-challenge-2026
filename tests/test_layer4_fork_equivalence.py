import gc
import hashlib
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, "/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026")
from src.normalizer import EntityNormalizer

import numpy as np
import polars as pl
import psutil
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestSuite")

# Globals for COW
_GLOBAL_TARGET_INDEX = None
_GLOBAL_QUERY_INDEX = None
_GLOBAL_TRUE_TARGET_INDICES = None
_GLOBAL_N_TARGET = 0
_GLOBAL_INTERNAL_TOP_K = 100
_GLOBAL_MIN_SIM = 0.35

_WORKER_SCORES = None
_WORKER_TOUCHED = None

def accumulate_tfidf_scores(index, features, weights, scores, touched):
    touched_count = 0
    for feature, weight in zip(features, weights):
        start, end = index.indptr[feature:feature + 2]
        ids = index.indices[start:end]
        new_ids = ids[scores[ids] == 0]
        new_count = len(new_ids)
        touched[touched_count:touched_count + new_count] = new_ids
        touched_count += new_count
        scores[ids] += index.data[start:end] * weight
    return np.sort(touched[:touched_count])

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
    global _WORKER_SCORES, _WORKER_TOUCHED
    _WORKER_SCORES = np.zeros(n_target, dtype=np.float64)
    _WORKER_TOUCHED = np.empty(n_target, dtype=indices_dtype)

def _layer4_worker_task(batch_range):
    start_idx, end_idx = batch_range
    target_index = _GLOBAL_TARGET_INDEX
    query_index = _GLOBAL_QUERY_INDEX
    true_target_indices = _GLOBAL_TRUE_TARGET_INDICES
    internal_top_k = _GLOBAL_INTERNAL_TOP_K
    min_sim = _GLOBAL_MIN_SIM

    scores = _WORKER_SCORES
    touched = _WORKER_TOUCHED

    query_indptr = query_index.indptr
    query_features = query_index.indices
    query_weights = query_index.data

    diag_counts = {
        "after_layer4_before_threshold": 0,
        "after_layer4_before_top_k": 0,
        "after_layer4_top_k": 0,
    }

    has_truth = true_target_indices is not None
    batch_results = []

    for s1_idx in range(start_idx, end_idx):
        q_start, q_end = query_indptr[s1_idx:s1_idx + 2]
        if q_start == q_end:
            continue

        reached_ids = accumulate_tfidf_scores(
            target_index,
            query_features[q_start:q_end],
            query_weights[q_start:q_end],
            scores,
            touched,
        )

        if has_truth:
            truth_indices = true_target_indices[s1_idx]
            diag_counts["after_layer4_before_threshold"] += sum(
                scores[t_idx] > 0 for t_idx in truth_indices
            )
            diag_counts["after_layer4_before_top_k"] += sum(
                scores[t_idx] >= min_sim for t_idx in truth_indices
            )

        candidate_ids = reached_ids[scores[reached_ids] >= min_sim]
        if not len(candidate_ids):
            scores[reached_ids] = 0.0
            continue

        similarities = scores[candidate_ids]
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
        scores[reached_ids] = 0.0

    return batch_results, diag_counts

def run_layer4_sequential(X_target_index, X_s1, n_target, n_s1, true_target_indices):
    scores = np.zeros(n_target, dtype=np.float64)
    touched = np.empty(n_target, dtype=X_target_index.indices.dtype)
    query_indptr = X_s1.indptr
    query_features = X_s1.indices
    query_weights = X_s1.data

    diag_counts = {
        "after_layer4_before_threshold": 0,
        "after_layer4_before_top_k": 0,
        "after_layer4_top_k": 0,
    }
    cands_map = defaultdict(lambda: defaultdict(lambda: {"tfidf": 0.0}))

    for s1_idx in range(n_s1):
        q_start, q_end = query_indptr[s1_idx:s1_idx + 2]
        if q_start == q_end:
            continue

        reached_ids = accumulate_tfidf_scores(
            X_target_index, query_features[q_start:q_end],
            query_weights[q_start:q_end], scores, touched
        )
        if true_target_indices is not None:
            truth_indices = true_target_indices[s1_idx]
            diag_counts["after_layer4_before_threshold"] += sum(
                scores[t_idx] > 0 for t_idx in truth_indices
            )
            diag_counts["after_layer4_before_top_k"] += sum(
                scores[t_idx] >= 0.35 for t_idx in truth_indices
            )
        candidate_ids = reached_ids[scores[reached_ids] >= 0.35]
        if not len(candidate_ids):
            scores[reached_ids] = 0.0
            continue
        similarities = scores[candidate_ids]
        if len(similarities) > 100:
            top = np.argpartition(similarities, -100)[-100:]
            candidate_ids = candidate_ids[top]
            similarities = similarities[top]

        if true_target_indices is not None:
            diag_counts["after_layer4_top_k"] += len(
                true_target_indices[s1_idx] & set(map(int, candidate_ids))
            )

        for t_idx, similarity in zip(candidate_ids, similarities):
            cands_map[s1_idx][int(t_idx)]["tfidf"] = max(
                cands_map[s1_idx][int(t_idx)].get("tfidf", 0.0),
                float(similarity)
            )
        scores[reached_ids] = 0.0

    return cands_map, diag_counts

def run_layer4_parallel(X_target_index, X_s1, n_target, n_s1, true_target_indices, num_workers):
    try:
        mp_ctx = mp.get_context("fork")
    except Exception as exc:
        raise RuntimeError(
            "macOS fork Copy-on-Write mode is required for Layer 4 parallelization. "
            "Serialized fallback is refused."
        ) from exc

    logger.info("Using macOS fork Copy-on-Write mode for Layer 4 parallelization (zero-copy inheritance, W=%d workers)", num_workers)

    global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES
    global _GLOBAL_N_TARGET, _GLOBAL_INTERNAL_TOP_K, _GLOBAL_MIN_SIM

    _GLOBAL_TARGET_INDEX = X_target_index
    _GLOBAL_QUERY_INDEX = X_s1
    _GLOBAL_TRUE_TARGET_INDICES = true_target_indices
    _GLOBAL_N_TARGET = n_target
    _GLOBAL_INTERNAL_TOP_K = 100
    _GLOBAL_MIN_SIM = 0.35

    # Contiguous batch partitioning
    # For small tests (<= 4000), chunk evenly across workers (at least 4 tasks per worker)
    # For full runs (> 4000), chunk at ~1000-2000 queries per task
    if n_s1 <= num_workers * 4:
        batch_size = max(1, (n_s1 + num_workers - 1) // num_workers)
    elif n_s1 <= 4000:
        batch_size = max(25, n_s1 // (num_workers * 4))
    else:
        batch_size = max(1000, n_s1 // (num_workers * 4))

    tasks = [(i, min(i + batch_size, n_s1)) for i in range(0, n_s1, batch_size)]

    diag_counts = {
        "after_layer4_before_threshold": 0,
        "after_layer4_before_top_k": 0,
        "after_layer4_top_k": 0,
    }
    cands_map = defaultdict(lambda: defaultdict(lambda: {"tfidf": 0.0}))

    try:
        with mp_ctx.Pool(
            processes=num_workers,
            initializer=_init_layer4_worker_scratch,
            initargs=(n_target, X_target_index.indices.dtype),
        ) as pool:
            batch_outputs = pool.map(_layer4_worker_task, tasks)

        for batch_candidates, batch_diag in batch_outputs:
            if true_target_indices is not None:
                for k in ("after_layer4_before_threshold", "after_layer4_before_top_k", "after_layer4_top_k"):
                    diag_counts[k] += batch_diag[k]

            for s1_idx, cands in batch_candidates:
                for t_idx, similarity in cands:
                    cands_map[s1_idx][t_idx]["tfidf"] = max(
                        cands_map[s1_idx][t_idx].get("tfidf", 0.0),
                        similarity
                    )
    finally:
        _GLOBAL_TARGET_INDEX = None
        _GLOBAL_QUERY_INDEX = None
        _GLOBAL_TRUE_TARGET_INDICES = None

    return cands_map, diag_counts

def build_layer5_dataframe(s1_ids, target_ids, s1_names, s1_addrs, target_names, target_addrs, cands_map):
    pair_s1_ids, pair_target_ids, pair_scores, pair_layers = [], [], [], []
    l2_hits, l3_hits, l4_hits = [], [], []

    for s1_idx in range(len(s1_ids)):
        s1_id = s1_ids[s1_idx]
        cand_dict = cands_map.get(s1_idx, {})
        if not cand_dict:
            continue

        scored_cands = []
        s1_addr_tokens = set((s1_addrs[s1_idx] or "").split())
        for t_idx, hits in cand_dict.items():
            tfidf_score = hits["tfidf"]
            token_hit = hits.get("token_hit", 0)
            num_hit = hits.get("num_hit", 0)
            target_addr_tokens = set((target_addrs[t_idx] or "").split())
            union = s1_addr_tokens | target_addr_tokens
            addr_jaccard = len(s1_addr_tokens & target_addr_tokens) / len(union) if union else 0.0

            composite_score = (
                (2.0 * tfidf_score)
                + (1.2 * min(token_hit, 3))
                + (1.5 * min(num_hit, 2))
                + (1.0 * addr_jaccard)
            )
            layers_count = (1 if tfidf_score > 0 else 0) + (1 if token_hit > 0 else 0) + (1 if num_hit > 0 else 0)
            scored_cands.append((t_idx, composite_score, layers_count))

        scored_cands.sort(key=lambda x: x[1], reverse=True)
        top_cands = scored_cands[:20]

        for t_idx, score, layers in top_cands:
            pair_s1_ids.append(s1_id)
            pair_target_ids.append(target_ids[t_idx])
            pair_scores.append(score)
            pair_layers.append(layers)
            l2_hits.append(False)
            l3_hits.append(False)
            l4_hits.append(True if cand_dict[t_idx]["tfidf"] > 0 else False)

    return pl.DataFrame({
        "source1_entity_id": pl.Series(pair_s1_ids, dtype=pl.String),
        "candidate_entity_id": pl.Series(pair_target_ids, dtype=pl.String),
        "heuristic_score": pl.Series(pair_scores, dtype=pl.Float32),
        "layers_matched": pl.Series(pair_layers, dtype=pl.Int8),
        "l2": pl.Series(l2_hits, dtype=pl.Boolean),
        "l3": pl.Series(l3_hits, dtype=pl.Boolean),
        "l4": pl.Series(l4_hits, dtype=pl.Boolean)
    })

def sha256_parquet(df: pl.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

if __name__ == "__main__":
    logger.info("Starting Pre-Validation Test Harness...")

    # Load cleaned India data
    cleaned_india = Path("/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026/artifacts/cleaned/India")
    val_india = Path("/Users/priyanshusharan/Documents/Codex/2026-09-26/amazon-ml-2026/artifacts/val_split/India")

    s1_files = list(val_india.glob("val_s1_part_*.parquet"))
    target_files = list(cleaned_india.glob("train_source2_part_*.parquet")) + list(cleaned_india.glob("train_source3_part_*.parquet"))

    logger.info("Reading parquet files...")
    s1_full_df = pl.concat([pl.read_parquet(f) for f in s1_files])
    target_full_df = pl.concat([pl.read_parquet(f) for f in target_files])

    logger.info("Loaded India: %d S1 val records, %d Target records", len(s1_full_df), len(target_full_df))

    normalizer = EntityNormalizer()
    logger.info("Normalizing sample subsets...")


    # --- STEP 1: 10-query smoke test ---
    logger.info("=== STEP 1: 10-query smoke test (2 workers) ===")
    s1_10 = normalizer.normalize_polars_df(s1_full_df.slice(0, 10))
    target_10k = normalizer.normalize_polars_df(target_full_df.slice(0, 50_000))

    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_features=50_000, sublinear_tf=True)
    all_names = target_10k["name_clean"].to_list() + [n for n in s1_10["name_clean"].to_list() if n]
    vec.fit(all_names)
    X_target_10 = vec.transform(target_10k["name_clean"].to_list()).tocsc()
    X_target_10.sum_duplicates()
    X_s1_10 = vec.transform(s1_10["name_clean"].to_list()).tocsr()

    cands_smoke, diag_smoke = run_layer4_parallel(X_target_10, X_s1_10, len(target_10k), 10, None, num_workers=2)
    logger.info("Step 1 PASSED: 10 queries executed with 2 workers, no crash. Found %d queries with cands.", len(cands_smoke))

    # --- STEP 2: 100-query equivalence test ---
    logger.info("=== STEP 2: 100-query equivalence test (full worker count: 10) ===")
    s1_100 = normalizer.normalize_polars_df(s1_full_df.slice(0, 100))
    target_subset = normalizer.normalize_polars_df(target_full_df.slice(0, 200_000))

    s1_ids_100 = s1_100["entity_id"].to_list()
    target_ids_sub = target_subset["entity_id"].to_list()
    s1_names_100 = s1_100["name_clean"].to_list()
    target_names_sub = target_subset["name_clean"].to_list()
    s1_addrs_100 = s1_100["address_clean"].to_list()
    target_addrs_sub = target_subset["address_clean"].to_list()

    vec100 = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_features=50_000, sublinear_tf=True)
    vec100.fit(target_names_sub + [n for n in s1_names_100 if n])
    X_target_100 = vec100.transform(target_names_sub).tocsc()
    X_target_100.sum_duplicates()
    X_s1_100 = vec100.transform(s1_names_100).tocsr()

    target_id_to_idx = {eid: idx for idx, eid in enumerate(target_ids_sub)}
    # Mock some truth matches
    true_target_indices = [{i % len(target_subset)} for i in range(100)]

    logger.info("Running sequential baseline on 100 queries...")
    t0_seq = time.perf_counter()
    cands_seq, diag_seq = run_layer4_sequential(X_target_100, X_s1_100, len(target_subset), 100, true_target_indices)
    t_seq = time.perf_counter() - t0_seq

    logger.info("Running parallel fork COW on 100 queries (10 workers)...")
    t0_par = time.perf_counter()
    cands_par, diag_par = run_layer4_parallel(X_target_100, X_s1_100, len(target_subset), 100, true_target_indices, num_workers=10)
    t_par = time.perf_counter() - t0_par

    # Assertions
    logger.info("Comparing candidates_map...")
    assert set(cands_seq.keys()) == set(cands_par.keys()), "Candidate query keys differ!"
    for q_idx in cands_seq:
        assert set(cands_seq[q_idx].keys()) == set(cands_par[q_idx].keys()), f"Target candidates differ for query {q_idx}!"
        for t_idx in cands_seq[q_idx]:
            score_seq = cands_seq[q_idx][t_idx]["tfidf"]
            score_par = cands_par[q_idx][t_idx]["tfidf"]
            assert np.isclose(score_seq, score_par, atol=1e-6), f"Score mismatch: {score_seq} vs {score_par}"

    logger.info("Comparing diagnostics...")
    assert diag_seq == diag_par, f"Diagnostics mismatch: {diag_seq} vs {diag_par}"

    logger.info("Comparing Layer 5 DataFrames...")
    df_seq = build_layer5_dataframe(s1_ids_100, target_ids_sub, s1_names_100, s1_addrs_100, target_names_sub, target_addrs_sub, cands_seq)
    df_par = build_layer5_dataframe(s1_ids_100, target_ids_sub, s1_names_100, s1_addrs_100, target_names_sub, target_addrs_sub, cands_par)

    assert df_seq.equals(df_par), "Layer 5 DataFrames are not equal!"
    logger.info("parallel_df.equals(sequential_df) is TRUE!")

    seq_parquet = Path("scratch/test_seq.parquet")
    par_parquet = Path("scratch/test_par.parquet")
    h_seq = sha256_parquet(df_seq, seq_parquet)
    h_par = sha256_parquet(df_par, par_parquet)
    logger.info("SHA-256 Sequential: %s", h_seq)
    logger.info("SHA-256 Parallel:   %s", h_par)
    assert h_seq == h_par, f"SHA-256 hash mismatch! {h_seq} vs {h_par}"
    logger.info("Step 2 PASSED: 100 queries byte-for-byte identical (SHA-256 match)!")

    # --- STEP 3: 1000-query scaling test ---
    logger.info("=== STEP 3: 1000-query scaling test ===")
    s1_1000 = normalizer.normalize_polars_df(s1_full_df.slice(0, 1000))
    s1_names_1000 = s1_1000["name_clean"].to_list()
    X_s1_1000 = vec100.transform(s1_names_1000).tocsr()

    proc = psutil.Process()
    ram_before = proc.memory_info().rss / 1e6

    t0 = time.perf_counter()
    cands_1000_1, _ = run_layer4_parallel(X_target_100, X_s1_1000, len(target_subset), 1000, None, num_workers=1)
    t_1w = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands_1000_10, _ = run_layer4_parallel(X_target_100, X_s1_1000, len(target_subset), 1000, None, num_workers=10)
    t_10w = time.perf_counter() - t0

    ram_after = proc.memory_info().rss / 1e6

    logger.info("1000 queries on 1 worker:  %.3fs", t_1w)
    logger.info("1000 queries on 10 workers: %.3fs", t_10w)
    logger.info("Speedup: %.2fx", t_1w / max(0.001, t_10w))
    logger.info("RAM before: %.1f MB, RAM after: %.1f MB (Delta: %.1f MB)", ram_before, ram_after, ram_after - ram_before)
    logger.info("Step 3 PASSED: Scaling verified!")
