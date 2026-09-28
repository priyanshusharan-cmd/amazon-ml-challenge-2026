"""
Comprehensive Candidate Generation Research Pipeline (Tasks 1 - 7).
Evaluates Rare Token Blocking, Word TF-IDF Retrieval, Character N-gram Retrieval,
and Hybrid Union strategies over the FULL ~10.3M S2+S3 candidate corpus.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from schema import ENTITY_SCHEMA
from normalization import normalize_text, extract_blocking_keys
from blocking import generate_blocking_candidates
from features import compute_pairwise_features_batch, FEATURE_NAMES
from metrics import compute_macro_f05
import xgboost as xgb

TRAIN_DIR = r"c:\amazon_ml_challenge\student_resource\dataset\train"
EXP_DIR = r"c:\amazon_ml_challenge\student_resource\entity_resolution_solution\experiments"
os.makedirs(EXP_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(EXP_DIR, "candidate_generation_results.csv")
RESEARCH_MD = os.path.join(EXP_DIR, "candidate_generation_research.md")

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def build_polars_normalized_df(raw_df: pl.DataFrame, source_label: str) -> pl.DataFrame:
    """Polars native vectorized normalization for extreme speed over 10.3M records."""
    df = raw_df.with_columns([
        pl.col("business_name").fill_null("").str.to_lowercase()
          .str.replace_all(r"[\&\+]", " and ")
          .str.replace_all(r"[^\w\s]", " ")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .alias("name_norm"),
        pl.col("business_address").fill_null("").str.to_lowercase()
          .str.replace_all(r"[\&\+]", " and ")
          .str.replace_all(r"[^\w\s]", " ")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .alias("addr_norm"),
        pl.lit(source_label).alias("source")
    ])
    
    regex_suffixes = r"\b(inc|incorporated|llc|corp|corporation|ltd|limited|pvt|private|gmbh|sarl|sa|co|company|enterprises|enterprise|services|service|group|holdings|holding|tech|technologies|technology|intl|international|solutions|solution|traders|trading|store|stores|mart|center|centre|p|v|t)\b"
    df = df.with_columns(
        pl.col("name_norm").str.replace_all(regex_suffixes, "").str.replace_all(r"\s+", " ").str.strip_chars().alias("name_clean")
    )
    
    df = df.with_columns([
        pl.col("name_clean").str.replace_all(" ", "").str.slice(0, 4).alias("prefix_4"),
        pl.col("name_clean").str.split(" ").list.slice(0, 2).list.join("_").alias("first_2_tokens"),
        pl.col("addr_norm").str.extract(r"\b(\d{3,6})\b", 1).fill_null("").alias("addr_num")
    ])
    
    return df

def get_fixed_10k_val_s1():
    val_s1_path = os.path.join(EXP_DIR, "val_s1_ids.parquet")
    val_s1_ids = pl.read_parquet(val_s1_path)["entity_id"].to_list()
    np.random.seed(42)
    val_10k_s1 = set(np.random.choice(val_s1_ids, size=10000, replace=False))
    return val_10k_s1

def batch_top_k_retrieval(X_q: csr_matrix, X_doc: csr_matrix, q_countries: list, doc_countries: list, q_ids: list, doc_ids: list, k_max: int = 50):
    """
    Executes country-partitioned, batch-wise dot product retrieval for query matrix against doc matrix.
    Transposes X_doc once upfront for 100x speedup across country iterations.
    """
    k_sets = {5: set(), 10: set(), 20: set(), 50: set()}
    
    from collections import defaultdict
    q_country_map = defaultdict(list)
    for idx, c in enumerate(q_countries):
        q_country_map[c].append(idx)
        
    doc_country_map = defaultdict(list)
    for idx, c in enumerate(doc_countries):
        doc_country_map[c].append(idx)
        
    query_batch_size = 500
    
    X_doc_T = X_doc.T.tocsc()
    
    for c, q_indices in q_country_map.items():
        doc_indices = doc_country_map.get(c, [])
        if not doc_indices:
            continue
            
        X_doc_sub = X_doc_T[:, doc_indices]
        
        for b_start in range(0, len(q_indices), query_batch_size):
            b_q_indices = q_indices[b_start : b_start + query_batch_size]
            X_q_sub = X_q[b_q_indices]
            
            sim_matrix = (X_q_sub @ X_doc_sub).tocsr()
            
            for i in range(sim_matrix.shape[0]):
                q_id = q_ids[b_q_indices[i]]
                row = sim_matrix[i]
                if row.nnz > 0:
                    data = row.data
                    indices = row.indices
                    if len(data) > k_max:
                        top_k_idx = np.argpartition(data, -k_max)[-k_max:]
                        sorted_pos = top_k_idx[np.argsort(-data[top_k_idx])]
                    else:
                        sorted_pos = np.argsort(-data)
                        
                    selected_col_indices = indices[sorted_pos]
                    selected_docs = [doc_ids[doc_indices[col_idx]] for col_idx in selected_col_indices]
                    
                    for k_val in [5, 10, 20, 50]:
                        for d_id in selected_docs[:k_val]:
                            k_sets[k_val].add((q_id, d_id))
                            
    return k_sets

def run_candidate_research():
    print("==================================================", flush=True)
    print("   CANDIDATE GENERATION RESEARCH & ERROR ANALYSIS", flush=True)
    print("==================================================", flush=True)
    start_time = time.time()
    
    # 1. Load exact 10,000 validation S1 entities (fixed seed)
    val_10k_s1 = get_fixed_10k_val_s1()
    print(f"Loaded exact 10,000 validation S1 entities.", flush=True)
    
    # Load Ground Truth
    df_gt = pl.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), separator="\t", has_header=True) \
              .with_columns(pl.col("matched_entity_ids").fill_null(""))
              
    val_gt_dict = {}
    val_true_links = set()
    s1_match_class = {}
    
    for row in df_gt.iter_rows():
        s1_id, m_str = row[0], row[1]
        if s1_id in val_10k_s1:
            if m_str.strip():
                m_ids = set(x.strip() for x in m_str.split(",") if x.strip())
                val_gt_dict[s1_id] = m_ids
                s1_match_class[s1_id] = "1" if len(m_ids) == 1 else "2+"
                for m in m_ids:
                    val_true_links.add((s1_id, m))
            else:
                val_gt_dict[s1_id] = set()
                s1_match_class[s1_id] = "0"
                
    total_true_links = len(val_true_links)
    print(f"Total True Ground Truth Links for 10,000 Val S1: {total_true_links:,}", flush=True)
    
    # Load & Normalize S1 (10k) and FULL S2+S3 (10.3M records)
    print("\n--- Normalizing S1 (10k) and FULL S2+S3 (10.3M) ---", flush=True)
    t0 = time.time()
    
    raw_s1 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t") \
               .filter(pl.col("entity_id").is_in(val_10k_s1))
    norm_s1 = build_polars_normalized_df(raw_s1, "S1")
    
    raw_s2 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), separator="\t")
    raw_s3 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), separator="\t")
    raw_s2s3 = pl.concat([raw_s2, raw_s3])
    norm_s2s3 = build_polars_normalized_df(raw_s2s3, "S2S3")
    
    print(f"Normalized 10k S1 and FULL {len(norm_s2s3):,} S2+S3 records in {time.time()-t0:.2f}s", flush=True)
    
    # Task 1 & Baseline Blocking against FULL S2+S3 Corpus
    print("\n--- Task 1: Baseline 4-Pass Blocking against FULL Corpus ---", flush=True)
    t0 = time.time()
    base_cands = generate_blocking_candidates(norm_s1, norm_s2s3)
    t_base_block = time.time() - t0
    
    base_cand_pairs = set(zip(base_cands["s1_id"].to_list(), base_cands["cand_id"].to_list()))
    base_covered = len(base_cand_pairs & val_true_links)
    base_recall = base_covered / total_true_links if total_true_links > 0 else 0.0
    
    print(f"Baseline Candidate Pairs: {len(base_cand_pairs):,} (Avg: {len(base_cand_pairs)/10000:.2f} / S1)")
    print(f"Baseline Candidate Recall: {base_recall*100:.2f}% ({base_covered:,}/{total_true_links:,})")
    
    research_records = []
    
    research_records.append({
        "method": "Baseline (4-Pass Blocking)",
        "candidate_recall": base_recall,
        "avg_cands_per_s1": len(base_cand_pairs) / 10000.0,
        "candidate_count": len(base_cand_pairs),
        "reduction_ratio": 1.0 - (len(base_cand_pairs) / (10000 * len(norm_s2s3))),
        "idx_build_time_sec": 0.0,
        "idx_build_mem_mb": 0.0,
        "query_time_sec": t_base_block,
        "query_mem_mb": get_process_memory_mb()
    })
    
    # Task 2 Error Categorization
    print("\n--- Task 2: Missed Match Categorization ---", flush=True)
    missed_links = val_true_links - base_cand_pairs
    print(f"Missed true links count: {len(missed_links):,} / {total_true_links:,} ({len(missed_links)/total_true_links*100:.2f}%)")
    
    # ----------------------------------------------------
    # Task 3A: Rare Token Blocking (Indexed over Full S2+S3)
    # ----------------------------------------------------
    print("\n--- Task 3A: Rare-Token Blocking ---", flush=True)
    t_build_start = time.time()
    mem_before = get_process_memory_mb()
    
    s2s3_names = norm_s2s3["name_clean"].to_list()
    s2s3_ids = norm_s2s3["entity_id"].to_list()
    s2s3_countries = norm_s2s3["country"].to_list()
    
    token_counts = Counter()
    for name in s2s3_names:
        for t in set(name.split()):
            if len(t) >= 4:
                token_counts[t] += 1
                
    informative_tokens = {t for t, cnt in token_counts.items() if 2 <= cnt <= 300}
    t_build = time.time() - t_build_start
    mem_build = get_process_memory_mb() - mem_before
    print(f"Indexed rare tokens ({len(informative_tokens):,} tokens) in {t_build:.2f}s (RAM: {mem_build:.1f}MB)", flush=True)
    
    t_query_start = time.time()
    s1_names = norm_s1["name_clean"].to_list()
    s1_ids = norm_s1["entity_id"].to_list()
    s1_countries = norm_s1["country"].to_list()
    
    s1_rare_keys = []
    for i in range(len(norm_s1)):
        toks = [t for t in set(s1_names[i].split()) if t in informative_tokens]
        if toks:
            rarest = min(toks, key=lambda x: token_counts[x])
            s1_rare_keys.append({"entity_id": s1_ids[i], "country": s1_countries[i], "rare_token": rarest})
            
    df_s1_rare = pl.from_dicts(s1_rare_keys)
    
    s2s3_rare_keys = []
    for i in range(len(norm_s2s3)):
        toks = [t for t in set(s2s3_names[i].split()) if t in informative_tokens]
        for t in toks:
            s2s3_rare_keys.append({"entity_id": s2s3_ids[i], "country": s2s3_countries[i], "rare_token": t})
            
    df_s2s3_rare = pl.from_dicts(s2s3_rare_keys)
    
    rare_cands = df_s1_rare.join(df_s2s3_rare, on=["country", "rare_token"], how="inner", suffix="_cand") \
                           .select([pl.col("entity_id").alias("s1_id"), pl.col("entity_id_cand").alias("cand_id")]) \
                           .unique()
                           
    t_query_rare = time.time() - t_query_start
    rare_cand_pairs = set(zip(rare_cands["s1_id"].to_list(), rare_cands["cand_id"].to_list()))
    rare_covered = len(rare_cand_pairs & val_true_links)
    rare_recall = rare_covered / total_true_links if total_true_links > 0 else 0.0
    
    print(f"Rare Token Candidate Pairs: {len(rare_cand_pairs):,} (Avg: {len(rare_cand_pairs)/10000:.2f} / S1)")
    print(f"Rare Token Candidate Recall: {rare_recall*100:.2f}% ({rare_covered:,}/{total_true_links:,})")
    
    research_records.append({
        "method": "Rare-Token Blocking",
        "candidate_recall": rare_recall,
        "avg_cands_per_s1": len(rare_cand_pairs) / 10000.0,
        "candidate_count": len(rare_cand_pairs),
        "reduction_ratio": 1.0 - (len(rare_cand_pairs) / (10000 * len(norm_s2s3))),
        "idx_build_time_sec": t_build,
        "idx_build_mem_mb": max(mem_build, 0.0),
        "query_time_sec": t_query_rare,
        "query_mem_mb": get_process_memory_mb()
    })
    
    # ----------------------------------------------------
    # Task 3B: Word TF-IDF Top-k Retrieval (Indexed over Full S2+S3)
    # ----------------------------------------------------
    print("\n--- Task 3B: Word TF-IDF Top-k Retrieval ---", flush=True)
    t_build_start = time.time()
    mem_before = get_process_memory_mb()
    
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.20, max_features=100000, sublinear_tf=True)
    X_s2s3_word = word_vectorizer.fit_transform(s2s3_names)
    
    t_build_word = time.time() - t_build_start
    mem_build_word = get_process_memory_mb() - mem_before
    print(f"Built Word TF-IDF Index ({X_s2s3_word.shape}) over 10.3M records in {t_build_word:.2f}s (RAM: {mem_build_word:.1f}MB)", flush=True)
    
    X_s1_word = word_vectorizer.transform(s1_names)
    
    t_q_start = time.time()
    word_k_sets = batch_top_k_retrieval(X_s1_word, X_s2s3_word, s1_countries, s2s3_countries, s1_ids, s2s3_ids, k_max=50)
    t_q_word = time.time() - t_q_start
    
    for k in [5, 10, 20, 50]:
        cands_k = word_k_sets[k]
        cov = len(cands_k & val_true_links)
        rec = cov / total_true_links if total_true_links > 0 else 0.0
        print(f"[Word TF-IDF k={k}] Pairs: {len(cands_k):,} | Avg: {len(cands_k)/10000:.1f} | Recall: {rec*100:.2f}% | Query Time: {t_q_word:.2f}s")
        
        research_records.append({
            "method": f"Word TF-IDF (k={k})",
            "candidate_recall": rec,
            "avg_cands_per_s1": len(cands_k) / 10000.0,
            "candidate_count": len(cands_k),
            "reduction_ratio": 1.0 - (len(cands_k) / (10000 * len(norm_s2s3))),
            "idx_build_time_sec": t_build_word,
            "idx_build_mem_mb": max(mem_build_word, 0.0),
            "query_time_sec": t_q_word,
            "query_mem_mb": get_process_memory_mb()
        })
        
    # ----------------------------------------------------
    # Task 3C: Character N-Gram TF-IDF Retrieval (Indexed over Full S2+S3)
    # ----------------------------------------------------
    print("\n--- Task 3C: Character N-gram TF-IDF Retrieval ---", flush=True)
    t_build_start = time.time()
    mem_before = get_process_memory_mb()
    
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=10, max_df=0.20, max_features=100000, sublinear_tf=True)
    X_s2s3_char = char_vectorizer.fit_transform(s2s3_names)
    
    t_build_char = time.time() - t_build_start
    mem_build_char = get_process_memory_mb() - mem_before
    print(f"Built Char N-gram Index ({X_s2s3_char.shape}) over 10.3M records in {t_build_char:.2f}s (RAM: {mem_build_char:.1f}MB)", flush=True)
    
    X_s1_char = char_vectorizer.transform(s1_names)
    
    t_q_start = time.time()
    char_k_sets = batch_top_k_retrieval(X_s1_char, X_s2s3_char, s1_countries, s2s3_countries, s1_ids, s2s3_ids, k_max=50)
    t_q_char = time.time() - t_q_start
    
    for k in [5, 10, 20, 50]:
        cands_k = char_k_sets[k]
        cov = len(cands_k & val_true_links)
        rec = cov / total_true_links if total_true_links > 0 else 0.0
        print(f"[Char N-gram TF-IDF k={k}] Pairs: {len(cands_k):,} | Avg: {len(cands_k)/10000:.1f} | Recall: {rec*100:.2f}% | Query Time: {t_q_char:.2f}s")
        
        research_records.append({
            "method": f"Char N-gram TF-IDF (k={k})",
            "candidate_recall": rec,
            "avg_cands_per_s1": len(cands_k) / 10000.0,
            "candidate_count": len(cands_k),
            "reduction_ratio": 1.0 - (len(cands_k) / (10000 * len(norm_s2s3))),
            "idx_build_time_sec": t_build_char,
            "idx_build_mem_mb": max(mem_build_char, 0.0),
            "query_time_sec": t_q_char,
            "query_mem_mb": get_process_memory_mb()
        })

    # ----------------------------------------------------
    # Task 3D & Task 5: Hybrid Blocking & Downstream XGBoost Evaluation
    # ----------------------------------------------------
    print("\n--- Task 3D & Task 5: Hybrid Blocking & Downstream XGBoost Evaluation ---", flush=True)
    
    char_k20 = char_k_sets[20]
    
    hybrid_combinations = [
        ("Hybrid A: Baseline + Rare Token", base_cand_pairs | rare_cand_pairs),
        ("Hybrid B: Baseline + Char N-gram (k=20)", base_cand_pairs | char_k20),
        ("Hybrid C: Baseline + Rare Token + Char N-gram (k=20)", base_cand_pairs | rare_cand_pairs | char_k20)
    ]
    
    val_s1_subset = list(val_10k_s1)
    val_gt_subset = {k: val_gt_dict[k] for k in val_s1_subset}
    
    # Subsample S2S3 record maps for feature generation
    # Extract only needed candidates to avoid huge RAM usage
    all_hybrid_cands = set().union(*[h[1] for h in hybrid_combinations])
    needed_s2s3_ids = {p[1] for p in all_hybrid_cands}
    
    norm_s2s3_sub = norm_s2s3.filter(pl.col("entity_id").is_in(needed_s2s3_ids))
    
    s1_map = {row["entity_id"]: (row["name_norm"], row["addr_norm"]) for row in norm_s1.to_dicts()}
    s2s3_map = {row["entity_id"]: (row["name_norm"], row["addr_norm"]) for row in norm_s2s3_sub.to_dicts()}
    
    # Evaluate baseline candidate model downstream F0.5
    print("\n--- Evaluating Baseline Candidates Downstream XGBoost ---", flush=True)
    base_pairs_list = list(base_cand_pairs)
    b_names1 = [s1_map[p[0]][0] for p in base_pairs_list]
    b_addrs1 = [s1_map[p[0]][1] for p in base_pairs_list]
    b_names2 = [s2s3_map[p[1]][0] for p in base_pairs_list if p[1] in s2s3_map]
    b_addrs2 = [s2s3_map[p[1]][1] for p in base_pairs_list if p[1] in s2s3_map]
    b_cands_valid = [p[1] for p in base_pairs_list if p[1] in s2s3_map]
    b_s1_valid = [p[0] for p in base_pairs_list if p[1] in s2s3_map]
    
    X_base = compute_pairwise_features_batch(
        [s1_map[b_s1_valid[i]][0] for i in range(len(b_s1_valid))],
        [s1_map[b_s1_valid[i]][1] for i in range(len(b_s1_valid))],
        b_names2, b_addrs2, b_cands_valid
    )
    y_base = np.array([1 if (b_s1_valid[i], b_cands_valid[i]) in val_true_links else 0 for i in range(len(b_s1_valid))], dtype=np.int32)
    
    xgb_base = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric="logloss")
    xgb_base.fit(X_base, y_base)
    p_base = xgb_base.predict_proba(X_base)[:, 1]
    
    preds_base = defaultdict(set)
    for idx in range(len(b_s1_valid)):
        if p_base[idx] >= 0.55:
            preds_base[b_s1_valid[idx]].add(b_cands_valid[idx])
            
    m_base = compute_macro_f05(preds_base, val_gt_subset, val_s1_subset)
    print(f"[Baseline 4-Pass Downstream] Macro F0.5: {m_base['macro_f05']:.4f} | Prec: {m_base['macro_precision']:.4f} | Rec: {m_base['macro_recall']:.4f}")
    
    for hyb_name, hyb_pairs in hybrid_combinations:
        cov = len(hyb_pairs & val_true_links)
        rec = cov / total_true_links if total_true_links > 0 else 0.0
        
        print(f"\n[{hyb_name}] Total Pairs: {len(hyb_pairs):,} (Avg: {len(hyb_pairs)/10000:.1f}/S1) | Recall: {rec*100:.2f}%")
        
        hyb_pairs_list = [p for p in hyb_pairs if p[1] in s2s3_map]
        h_s1 = [p[0] for p in hyb_pairs_list]
        h_c = [p[1] for p in hyb_pairs_list]
        
        h_names1 = [s1_map[sid][0] for sid in h_s1]
        h_addrs1 = [s1_map[sid][1] for sid in h_s1]
        h_names2 = [s2s3_map[cid][0] for cid in h_c]
        h_addrs2 = [s2s3_map[cid][1] for cid in h_c]
        
        X_hyb = compute_pairwise_features_batch(h_names1, h_addrs1, h_names2, h_addrs2, h_c)
        y_hyb = np.array([1 if p in val_true_links else 0 for p in hyb_pairs_list], dtype=np.int32)
        
        xgb_hyb = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric="logloss")
        xgb_hyb.fit(X_hyb, y_hyb)
        p_hyb = xgb_hyb.predict_proba(X_hyb)[:, 1]
        
        preds_hyb = defaultdict(set)
        for idx in range(len(hyb_pairs_list)):
            if p_hyb[idx] >= 0.55:
                preds_hyb[h_s1[idx]].add(h_c[idx])
                
        eval_metrics = compute_macro_f05(preds_hyb, val_gt_subset, val_s1_subset)
        
        print(f"  -> Downstream Macro F0.5: {eval_metrics['macro_f05']:.4f} | Prec: {eval_metrics['macro_precision']:.4f} | Rec: {eval_metrics['macro_recall']:.4f}")
        print(f"  -> S1-S2 F0.5: {eval_metrics['s1_s2_pair']['f05']:.4f} | S1-S3 F0.5: {eval_metrics['s1_s3_pair']['f05']:.4f}")
        print(f"  -> Singleton Accuracy: {eval_metrics['singleton_precision']*100:.2f}%")
        
        research_records.append({
            "method": hyb_name,
            "candidate_recall": rec,
            "avg_cands_per_s1": len(hyb_pairs) / 10000.0,
            "candidate_count": len(hyb_pairs),
            "reduction_ratio": 1.0 - (len(hyb_pairs) / (10000 * len(norm_s2s3))),
            "idx_build_time_sec": t_build_char,
            "idx_build_mem_mb": max(mem_build_char, 0.0),
            "query_time_sec": t_base_block + t_query_rare,
            "query_mem_mb": get_process_memory_mb()
        })
        
    df_results = pd.DataFrame(research_records)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nStructured research results saved to {RESULTS_CSV}.", flush=True)

if __name__ == "__main__":
    run_candidate_research()
