import os
import gc
import time
import sqlite3
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

def compute_f05_single(pred_set, true_set):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def evaluate_predictions(pred_dict, gt_map, total_entities):
    score_sum = 0.0
    for s1_id, true_set in gt_map.items():
        preds = pred_dict.get(s1_id, set())
        score_sum += compute_f05_single(preds, true_set)
    return score_sum / total_entities

print("Loading validation ground truth...")
val_gt_path = 'output/val_gt_split.tsv'
gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
total_val_entities = len(gt_df)
gt_map = {}
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        gt_map[sid] = set(x.strip() for x in str(m).split(',') if x.strip())
    else:
        gt_map[sid] = set()

print(f"Total validation entities: {total_val_entities:,}")

print("Loading validation probabilities...")
val_probs = np.load('output/val_probs_lgb_v3.npy')

val_csv_path = 'output/full_val_features_v3.csv'
print(f"Streaming through {val_csv_path}...")

cands_by_s1 = {}
chunk_size = 500000
idx = 0

for chunk in pd.read_csv(val_csv_path, chunksize=chunk_size, usecols=['source1_entity_id', 'candidate_entity_id'], dtype=str):
    n = len(chunk)
    chunk_p = val_probs[idx:idx+n]
    mask = chunk_p >= 0.40
    if np.any(mask):
        sub_s1 = chunk['source1_entity_id'].to_numpy()[mask]
        sub_c = chunk['candidate_entity_id'].to_numpy()[mask]
        sub_p = chunk_p[mask]
        for s1, c, p in zip(sub_s1, sub_c, sub_p):
            if s1 not in cands_by_s1:
                cands_by_s1[s1] = []
            cands_by_s1[s1].append((c, float(p)))
    idx += n

print(f"Queries with candidates >= 0.40: {len(cands_by_s1):,}")

# Baseline score at threshold 0.64 with singleton cutoff 0.75
baseline_preds = {}
for s1, cands in cands_by_s1.items():
    max_p = max(p for c, p in cands)
    if max_p >= 0.75:
        p_set = set(c for c, p in cands if p >= 0.64)
        if p_set:
            baseline_preds[s1] = p_set

base_score = evaluate_predictions(baseline_preds, gt_map, total_val_entities)
print(f"\n>>> Baseline Validation Macro F0.5 (T=0.64, cutoff=0.75): {base_score:.5f} <<<")

# Test Transitive / Multi-Match Propagation:
# In ER: If a query has multiple strong matches (p >= 0.85) from S2 and S3,
# borderline candidates (p in [0.55, 0.64)) from the same cluster are true matches!
for border_t in [0.55, 0.58, 0.60, 0.62]:
    trans_preds = {}
    rescued_count = 0
    for s1, cands in cands_by_s1.items():
        max_p = max(p for c, p in cands)
        if max_p >= 0.85:
            # Query has very strong anchor match
            p_set = set(c for c, p in cands if p >= border_t)
            if p_set:
                trans_preds[s1] = p_set
                rescued_count += len(p_set) - sum(1 for c, p in cands if p >= 0.64)
        elif max_p >= 0.75:
            p_set = set(c for c, p in cands if p >= 0.64)
            if p_set:
                trans_preds[s1] = p_set
    t_score = evaluate_predictions(trans_preds, gt_map, total_val_entities)
    print(f"Borderline threshold {border_t:.2f} for anchor queries (p>=0.85): Macro F0.5 = {t_score:.5f} (Rescued {rescued_count:,} matches)")
