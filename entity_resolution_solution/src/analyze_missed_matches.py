"""
Task 1 & Task 2: Baseline Candidate Error & Missed Matches Analysis.
Analyzes why the baseline 4-pass blocking misses true matches on the exact 10,000 validation S1 entities.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from normalization import normalize_text, extract_blocking_keys
from blocking import generate_blocking_candidates

TRAIN_DIR = r"c:\amazon_ml_challenge\student_resource\dataset\train"
EXP_DIR = r"c:\amazon_ml_challenge\student_resource\entity_resolution_solution\experiments"

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

def get_fixed_10k_val_s1():
    val_s1_path = os.path.join(EXP_DIR, "val_s1_ids.parquet")
    val_s1_ids = pl.read_parquet(val_s1_path)["entity_id"].to_list()
    np.random.seed(42)
    val_10k_s1 = set(np.random.choice(val_s1_ids, size=10000, replace=False))
    return val_10k_s1

def analyze_baseline_misses():
    print("==================================================", flush=True)
    print("   TASK 1 & 2: MISSED MATCH ERROR ANALYSIS", flush=True)
    print("==================================================", flush=True)
    
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
    print(f"Total True Links for 10,000 Val S1: {total_true_links:,}", flush=True)
    
    # Load & Normalize 10k S1 and FULL S2+S3 Corpus
    print("\n--- Loading & Normalizing S1 (10k) and FULL S2+S3 (10.3M) ---", flush=True)
    t0 = time.time()
    
    raw_s1 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t") \
               .filter(pl.col("entity_id").is_in(val_10k_s1))
    norm_s1 = build_normalized_df(raw_s1, "S1")
    
    raw_s2 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), separator="\t")
    raw_s3 = pl.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), separator="\t")
    raw_s2s3 = pl.concat([raw_s2, raw_s3])
    norm_s2s3 = build_normalized_df(raw_s2s3, "S2S3")
    print(f"Normalized 10k S1 and 10.3M S2+S3 in {time.time()-t0:.2f}s", flush=True)
    
    # Run Baseline 4-Pass Blocking over FULL S2+S3 corpus
    print("\n--- Running Baseline 4-Pass Blocking against FULL Corpus ---", flush=True)
    t0 = time.time()
    base_cands = generate_blocking_candidates(norm_s1, norm_s2s3)
    t_blocking = time.time() - t0
    
    base_cand_pairs = set(zip(base_cands["s1_id"].to_list(), base_cands["cand_id"].to_list()))
    
    covered_links = val_true_links & base_cand_pairs
    missed_links = val_true_links - base_cand_pairs
    
    cand_recall = len(covered_links) / total_true_links if total_true_links > 0 else 0.0
    
    print(f"Baseline Blocking Execution Time: {t_blocking:.2f}s")
    print(f"Total Candidate Pairs Generated: {len(base_cand_pairs):,}")
    print(f"Avg Candidates per S1: {len(base_cand_pairs)/10000:.2f}")
    print(f"Covered True Links: {len(covered_links):,} / {total_true_links:,}")
    print(f"Missed True Links: {len(missed_links):,} / {total_true_links:,}")
    print(f"Baseline Candidate Recall: {cand_recall*100:.2f}%")
    
    # Task 1 Breakdown
    # 1. By S2 vs S3
    s2_total = len([x for x in val_true_links if x[1].startswith("S2-")])
    s2_covered = len([x for x in covered_links if x[1].startswith("S2-")])
    s3_total = len([x for x in val_true_links if x[1].startswith("S3-")])
    s3_covered = len([x for x in covered_links if x[1].startswith("S3-")])
    
    print("\nBreakdown by Source:")
    print(f"  - S1-S2: Covered {s2_covered:,} / {s2_total:,} (Recall: {s2_covered/s2_total*100:.2f}%)")
    print(f"  - S1-S3: Covered {s3_covered:,} / {s3_total:,} (Recall: {s3_covered/s3_total*100:.2f}%)")
    
    # 2. By Country
    s1_country_map = {row["entity_id"]: row["country"] for row in norm_s1.to_dicts()}
    us_total = len([x for x in val_true_links if s1_country_map.get(x[0]) == "US"])
    us_covered = len([x for x in covered_links if s1_country_map.get(x[0]) == "US"])
    in_total = len([x for x in val_true_links if s1_country_map.get(x[0]) == "India"])
    in_covered = len([x for x in covered_links if s1_country_map.get(x[0]) == "India"])
    
    print("\nBreakdown by Country:")
    print(f"  - US: Covered {us_covered:,} / {us_total:,} (Recall: {us_covered/us_total*100:.2f}%)")
    print(f"  - India: Covered {in_covered:,} / {in_total:,} (Recall: {in_covered/in_total*100:.2f}%)")
    
    # 3. By S1 Match Class
    class_totals = Counter()
    class_covered = Counter()
    for s1_id, cand_id in val_true_links:
        m_cls = s1_match_class[s1_id]
        class_totals[m_cls] += 1
        if (s1_id, cand_id) in covered_links:
            class_covered[m_cls] += 1
            
    print("\nBreakdown by Match Class:")
    for m_cls in sorted(class_totals.keys()):
        tot = class_totals[m_cls]
        cov = class_covered[m_cls]
        print(f"  - Class '{m_cls}': Covered {cov:,} / {tot:,} (Recall: {cov/tot*100:.2f}%)")
        
    # Task 2: Error Inspection of Missed Matches
    print("\n--- TASK 2: Qualitative Error Analysis of Missed Matches ---", flush=True)
    
    s1_record_map = {row["entity_id"]: row for row in norm_s1.to_dicts()}
    
    # Filter S2S3 for missed target entity records
    missed_target_ids = {x[1] for x in missed_links}
    raw_missed_s2s3 = raw_s2s3.filter(pl.col("entity_id").is_in(missed_target_ids))
    norm_missed_s2s3 = build_normalized_df(raw_missed_s2s3, "S2S3")
    s2s3_missed_map = {row["entity_id"]: row for row in norm_missed_s2s3.to_dicts()}
    
    error_categories = Counter()
    sample_errors = []
    
    for s1_id, cand_id in list(missed_links)[:500]: # Sample 500 missed links
        rec1 = s1_map_info = s1_record_map[s1_id]
        rec2 = s2s3_missed_map.get(cand_id)
        if not rec2:
            continue
            
        n1, n2 = rec1["name_norm"], rec2["name_norm"]
        c1, c2 = rec1["name_clean"], rec2["name_clean"]
        a1, a2 = rec1["addr_norm"], rec2["addr_norm"]
        
        t1, t2 = set(n1.split()), set(n2.split())
        token_overlap = len(t1 & t2)
        
        # Categorize error
        category = "other"
        if not a1 or not a2:
            category = "missing address/name"
        elif c1 == c2:
            category = "legal suffix variation"
        elif set(c1.split()) == set(c2.split()):
            category = "token reordering"
        elif token_overlap == 0:
            category = "completely different-looking names but related address"
        elif token_overlap > 0:
            category = "spelling variation / typo"
        else:
            category = "abbreviation"
            
        error_categories[category] += 1
        
        if len(sample_errors) < 10:
            sample_errors.append({
                "s1_id": s1_id,
                "cand_id": cand_id,
                "country": rec1["country"],
                "s1_name": rec1["business_name"],
                "cand_name": rec2["business_name"],
                "s1_addr": rec1["business_address"],
                "cand_addr": rec2["business_address"],
                "category": category
            })
            
    print("Categorized Missed Match Error Breakdown (Sample size: 500):")
    for cat, cnt in error_categories.most_common():
        pct = cnt / sum(error_categories.values()) * 100
        print(f"  - {cat}: {cnt} ({pct:.1f}%)")
        
    # Save error analysis json
    with open(os.path.join(EXP_DIR, "missed_matches_analysis.json"), "w") as f:
        json.dump({
            "total_true_links": total_true_links,
            "covered_links": len(covered_links),
            "missed_links": len(missed_links),
            "candidate_recall": cand_recall,
            "s2_recall": s2_covered / s2_total,
            "s3_recall": s3_covered / s3_total,
            "us_recall": us_covered / us_total,
            "india_recall": in_covered / in_total,
            "error_categories": dict(error_categories),
            "sample_errors": sample_errors
        }, f, indent=2)
        
    print("\nTask 1 & Task 2 complete. Summary saved to experiments/missed_matches_analysis.json.", flush=True)

if __name__ == "__main__":
    analyze_baseline_misses()
