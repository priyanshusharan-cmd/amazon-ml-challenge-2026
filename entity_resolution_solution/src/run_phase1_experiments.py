"""
Phase 1 Experimentation Pipeline.
Runs normalization, stratified sampling, multi-pass blocking, feature generation,
model training (Logistic Regression, Random Forest, LightGBM, XGBoost),
threshold tuning, per-pair breakdown, and results logging.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from schema import ENTITY_SCHEMA
from normalization import normalize_text, extract_blocking_keys, get_tokens
from validation_split import create_stratified_split
from blocking import generate_blocking_candidates
from features import compute_pairwise_features_batch, FEATURE_NAMES
from metrics import compute_macro_f05

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

TRAIN_DIR = r"c:\amazon_ml_challenge\student_resource\dataset\train"
EXP_DIR = r"c:\amazon_ml_challenge\student_resource\entity_resolution_solution\experiments"
os.makedirs(EXP_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(EXP_DIR, "results.csv")

def build_normalized_df(raw_df: pl.DataFrame, source_label: str) -> pl.DataFrame:
    """Applies clean normalization to raw Polars DataFrame and extracts clean keys."""
    pdf = raw_df.to_pandas()
    
    names_norm = [normalize_text(n, remove_suffixes=False) for n in pdf["business_name"]]
    names_clean = [normalize_text(n, remove_suffixes=True) for n in pdf["business_name"]]
    addrs_norm = [normalize_text(a, remove_suffixes=False) if a else "" for a in pdf["business_address"]]
    
    prefixes = []
    first_2_list = []
    addr_num_list = []
    
    for i in range(len(pdf)):
        keys = extract_blocking_keys(names_norm[i], names_clean[i], addrs_norm[i])
        prefixes.append(keys["prefix_4"])
        first_2_list.append(keys["first_2_tokens"])
        addr_num_list.append(keys["addr_num"])
        
    pdf["name_norm"] = names_norm
    pdf["name_clean"] = names_clean
    pdf["addr_norm"] = addrs_norm
    pdf["prefix_4"] = prefixes
    pdf["first_2_tokens"] = first_2_list
    pdf["addr_num"] = addr_num_list
    pdf["source"] = source_label
    
    return pl.from_pandas(pdf)

def run_phase1():
    print("==================================================", flush=True)
    print("      PHASE 1: SAMPLE EXPERIMENTATION PIPELINE", flush=True)
    print("==================================================", flush=True)
    start_total = time.time()
    
    # 1. Stratified Validation Split
    print("\n--- STEP 1: Validation Split ---", flush=True)
    train_s1_path, val_s1_path = create_stratified_split(
        s1_path=os.path.join(TRAIN_DIR, "train_source1.tsv"),
        gt_path=os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        output_dir=EXP_DIR,
        val_size=0.2,
        random_seed=42
    )
    
    train_s1_ids = pl.read_parquet(train_s1_path)["entity_id"].to_list()
    val_s1_ids = pl.read_parquet(val_s1_path)["entity_id"].to_list()
    
    # Sample size: 40,000 Train S1, 10,000 Val S1
    np.random.seed(42)
    sample_train_s1 = set(np.random.choice(train_s1_ids, size=40000, replace=False))
    sample_val_s1 = set(np.random.choice(val_s1_ids, size=10000, replace=False))
    all_sample_s1 = sample_train_s1 | sample_val_s1
    
    print(f"Sample size: {len(sample_train_s1):,} Train S1, {len(sample_val_s1):,} Val S1", flush=True)
    
    # 2. Load Ground Truth for sample
    print("\n--- STEP 2: Loading Ground Truth ---", flush=True)
    df_gt = pl.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), separator="\t", has_header=True) \
              .with_columns(pl.col("matched_entity_ids").fill_null(""))
              
    gt_dict = {}
    true_matches_set = set()
    sample_gt_s2s3_ids = set()
    
    for row in df_gt.iter_rows():
        s1_id, m_str = row[0], row[1]
        if s1_id in all_sample_s1:
            if m_str.strip():
                m_ids = set(x.strip() for x in m_str.split(",") if x.strip())
                gt_dict[s1_id] = m_ids
                for m in m_ids:
                    true_matches_set.add((s1_id, m))
                    sample_gt_s2s3_ids.add(m)
            else:
                gt_dict[s1_id] = set()
                
    total_true_links = len(true_matches_set)
    print(f"Sample Ground Truth Links: {total_true_links:,}", flush=True)
    
    # 3. Load & Normalize S1, S2, S3 for Sample
    print("\n--- STEP 3: Normalizing Sample Records ---", flush=True)
    t0 = time.time()
    
    raw_s1 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t") \
               .filter(pl.col("entity_id").is_in(all_sample_s1))
               
    norm_s1 = build_normalized_df(raw_s1, "S1")
    
    raw_s2 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), separator="\t")
    raw_s3 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), separator="\t")
    
    s2_matched_df = raw_s2.filter(pl.col("entity_id").is_in(sample_gt_s2s3_ids))
    s3_matched_df = raw_s3.filter(pl.col("entity_id").is_in(sample_gt_s2s3_ids))
    
    s2_rand_df = raw_s2.filter(~pl.col("entity_id").is_in(sample_gt_s2s3_ids)).sample(n=100000, seed=42)
    s3_rand_df = raw_s3.filter(~pl.col("entity_id").is_in(sample_gt_s2s3_ids)).sample(n=100000, seed=42)
    
    raw_s2s3 = pl.concat([s2_matched_df, s3_matched_df, s2_rand_df, s3_rand_df])
    norm_s2s3 = build_normalized_df(raw_s2s3, "S2S3")
    
    print(f"Normalized {len(norm_s1):,} S1 records and {len(norm_s2s3):,} S2/S3 records in {time.time()-t0:.2f}s", flush=True)
    
    # 4. Multi-Pass Blocking
    print("\n--- STEP 4: Candidate Generation / Blocking ---", flush=True)
    t0 = time.time()
    cands_df = generate_blocking_candidates(norm_s1, norm_s2s3)
    blocking_time = time.time() - t0
    
    total_candidate_pairs = len(cands_df)
    
    # Measure Candidate Recall
    cand_pairs_set = set(zip(cands_df["s1_id"].to_list(), cands_df["cand_id"].to_list()))
    retrieved_true_links = len(cand_pairs_set & true_matches_set)
    candidate_recall = retrieved_true_links / total_true_links if total_true_links > 0 else 0.0
    
    total_possible = len(norm_s1) * len(norm_s2s3)
    reduction_ratio = 1.0 - (total_candidate_pairs / total_possible)
    
    print(f"Blocking Runtime: {blocking_time:.2f}s")
    print(f"Candidate Pairs: {total_candidate_pairs:,}")
    print(f"Retrieved True Matches: {retrieved_true_links:,} / {total_true_links:,}")
    print(f"Candidate Recall: {candidate_recall * 100:.2f}%")
    print(f"Reduction Ratio: {reduction_ratio * 100:.6f}%")
    
    # 5. Pairwise Feature Extraction
    print("\n--- STEP 5: Feature Extraction ---", flush=True)
    t0 = time.time()
    
    s1_map = {row["entity_id"]: (row["name_norm"], row["addr_norm"]) for row in norm_s1.to_dicts()}
    s2s3_map = {row["entity_id"]: (row["name_norm"], row["addr_norm"]) for row in norm_s2s3.to_dicts()}
    
    s1_ids_list = cands_df["s1_id"].to_list()
    cand_ids_list = cands_df["cand_id"].to_list()
    
    names1 = [s1_map[sid][0] for sid in s1_ids_list]
    addrs1 = [s1_map[sid][1] for sid in s1_ids_list]
    names2 = [s2s3_map[cid][0] for cid in cand_ids_list]
    addrs2 = [s2s3_map[cid][1] for cid in cand_ids_list]
    
    X = compute_pairwise_features_batch(names1, addrs1, names2, addrs2, cand_ids_list)
    y = np.array([1 if (s1_ids_list[i], cand_ids_list[i]) in true_matches_set else 0 for i in range(len(s1_ids_list))], dtype=np.int32)
    
    feature_time = time.time() - t0
    print(f"Feature matrix shape: {X.shape}, positive labels: {np.sum(y):,} ({np.mean(y)*100:.2f}%) extracted in {feature_time:.2f}s", flush=True)
    
    # Train / Val Pair Split
    is_val_pair = np.array([s1_ids_list[i] in sample_val_s1 for i in range(len(s1_ids_list))])
    is_train_pair = ~is_val_pair
    
    X_train, y_train = X[is_train_pair], y[is_train_pair]
    X_val, y_val = X[is_val_pair], y[is_val_pair]
    
    val_s1_pairs = [(s1_ids_list[i], cand_ids_list[i]) for i in range(len(s1_ids_list)) if is_val_pair[i]]
    val_s1_subset = list(sample_val_s1)
    val_gt_subset = {k: gt_dict[k] for k in val_s1_subset}
    
    # 6. Model Training & Comparison
    print("\n--- STEP 6: Model Training & Threshold Optimization ---", flush=True)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    exp_results = []
    
    for m_name, model in models.items():
        print(f"\nTraining {m_name}...", flush=True)
        t_tr = time.time()
        try:
            model.fit(X_train, y_train)
            tr_time = time.time() - t_tr
            
            t_inf = time.time()
            val_probs = model.predict_proba(X_val)[:, 1]
            inf_time = time.time() - t_inf
            
            # Grid search probability thresholds
            best_thresh = 0.5
            best_f05 = -1.0
            best_metrics = {}
            
            for thresh in np.arange(0.50, 0.96, 0.05):
                preds_dict = defaultdict(set)
                for idx, (s1_id, cand_id) in enumerate(val_s1_pairs):
                    if val_probs[idx] >= thresh:
                        preds_dict[s1_id].add(cand_id)
                        
                m_res = compute_macro_f05(preds_dict, val_gt_subset, val_s1_subset)
                if m_res["macro_f05"] > best_f05:
                    best_f05 = m_res["macro_f05"]
                    best_thresh = float(thresh)
                    best_metrics = m_res
                    
            print(f"[{m_name}] Best Val Threshold: {best_thresh:.2f} | Macro F0.5: {best_metrics['macro_f05']:.4f} | Prec: {best_metrics['macro_precision']:.4f} | Rec: {best_metrics['macro_recall']:.4f}")
            print(f"  -> S1-S2 Pair F0.5: {best_metrics['s1_s2_pair']['f05']:.4f} (Prec: {best_metrics['s1_s2_pair']['precision']:.4f}, Rec: {best_metrics['s1_s2_pair']['recall']:.4f})")
            print(f"  -> S1-S3 Pair F0.5: {best_metrics['s1_s3_pair']['f05']:.4f} (Prec: {best_metrics['s1_s3_pair']['precision']:.4f}, Rec: {best_metrics['s1_s3_pair']['recall']:.4f})")
            print(f"  -> Singletons: {best_metrics['singleton_count']} entities, Acc: {best_metrics['singleton_precision']*100:.2f}%")
            
            row_exp = {
                "experiment_id": f"phase1_{m_name.lower()}",
                "blocking_strategy": "multi_pass_v2",
                "feature_set": f"{len(FEATURE_NAMES)}_features",
                "model": m_name,
                "threshold": best_thresh,
                "candidate_count": total_candidate_pairs,
                "candidate_recall": candidate_recall,
                "precision": best_metrics["macro_precision"],
                "recall": best_metrics["macro_recall"],
                "macro_f05": best_metrics["macro_f05"],
                "s1_s2_f05": best_metrics["s1_s2_pair"]["f05"],
                "s1_s3_f05": best_metrics["s1_s3_pair"]["f05"],
                "singleton_prec": best_metrics["singleton_precision"],
                "runtime_sec": tr_time + inf_time,
                "notes": f"Phase 1 sample, {len(sample_val_s1)} val S1 entities"
            }
            exp_results.append(row_exp)
        except Exception as e:
            print(f"[{m_name}] skipped due to environment incompatibility: {e}")
        
    df_exp = pd.DataFrame(exp_results)
    df_exp.to_csv(RESULTS_CSV, index=False)
    
    total_elapsed = time.time() - start_total
    print(f"\nPhase 1 Experimentation Complete in {total_elapsed:.2f}s. Results logged to {RESULTS_CSV}.", flush=True)
    
    return {
        "candidate_recall": candidate_recall,
        "total_candidate_pairs": total_candidate_pairs,
        "reduction_ratio": reduction_ratio,
        "exp_results": exp_results
    }

if __name__ == "__main__":
    run_phase1()
