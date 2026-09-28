import os
import sys
import polars as pl
import json
import time

TRAIN_DIR = r"c:\amazon_ml_challenge\student_resource\dataset\train"
TEST_DIR = r"c:\amazon_ml_challenge\student_resource\dataset\test"
OUTPUT_DIR = r"c:\amazon_ml_challenge\student_resource\entity_resolution_solution\experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_file(filepath):
    print(f"\n--- Inspecting {os.path.basename(filepath)} ---", flush=True)
    t0 = time.time()
    
    # Read lazily or eager with polars
    df = pl.read_csv(filepath, separator="\t", has_header=True, schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}, ignore_errors=True)
    
    total_rows = len(df)
    missing_counts = {}
    for col in df.columns:
        null_or_empty = df.filter(pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")).height
        missing_counts[col] = null_or_empty
        
    country_counts = {}
    if "country" in df.columns:
        counts = df.group_by("country").len().sort("len", descending=True)
        for row in counts.iter_rows():
            c_val = row[0] if row[0] is not None else "<NULL>"
            country_counts[c_val] = row[1]
            
    name_stats = {}
    if "business_name" in df.columns:
        name_lens = df.select(pl.col("business_name").str.len_chars().fill_null(0).alias("name_len"))["name_len"]
        name_stats = {
            "min": int(name_lens.min()),
            "max": int(name_lens.max()),
            "median": float(name_lens.median()),
            "mean": float(name_lens.mean()),
            "p95": float(name_lens.quantile(0.95))
        }
        
    addr_stats = {}
    if "business_address" in df.columns:
        addr_lens = df.select(pl.col("business_address").str.len_chars().fill_null(0).alias("addr_len"))["addr_len"]
        addr_stats = {
            "min": int(addr_lens.min()),
            "max": int(addr_lens.max()),
            "median": float(addr_lens.median()),
            "mean": float(addr_lens.mean()),
            "p95": float(addr_lens.quantile(0.95))
        }

    # Count duplicate names (case-insensitive)
    dup_names = 0
    if "business_name" in df.columns:
        dup_names = df.filter(pl.col("business_name").is_not_null() & (pl.col("business_name").str.strip_chars() != "")) \
                      .select(pl.col("business_name").str.to_lowercase().str.strip_chars()) \
                      .group_by("business_name").len() \
                      .filter(pl.col("len") > 1).height
                      
    t1 = time.time()
    print(f"Read & Processed {total_rows:,} rows in {t1 - t0:.2f}s", flush=True)
    print(f"Missing Counts: {missing_counts}", flush=True)
    print(f"Country Counts: {country_counts}", flush=True)
    print(f"Name Length Stats: {name_stats}", flush=True)
    print(f"Address Length Stats: {addr_stats}", flush=True)
    print(f"Unique Duplicate Business Names (case-insensitive): {dup_names:,}", flush=True)
    
    return {
        "filename": os.path.basename(filepath),
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "country_counts": country_counts,
        "name_stats": name_stats,
        "addr_stats": addr_stats,
        "dup_names_count": dup_names
    }

def inspect_ground_truth(filepath):
    print(f"\n--- Inspecting Ground Truth {os.path.basename(filepath)} ---", flush=True)
    t0 = time.time()
    
    df = pl.read_csv(filepath, separator="\t", has_header=True, schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}, ignore_errors=True)
    
    total_s1 = len(df)
    
    # Fill nulls with empty string
    df = df.with_columns(pl.col("matched_entity_ids").fill_null(""))
    
    # Calculate match count per S1
    # Count commas if string is non-empty, else 0
    df = df.with_columns(
        pl.when(pl.col("matched_entity_ids").str.strip_chars() == "")
        .then(0)
        .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
        .alias("match_count")
    )
    
    match_dist = df.group_by("match_count").len().sort("match_count").to_dicts()
    
    singletons = df.filter(pl.col("match_count") == 0).height
    single_match = df.filter(pl.col("match_count") == 1).height
    multi_match = df.filter(pl.col("match_count") >= 2).height
    
    # Count S2 vs S3 matches
    # Explode matched_entity_ids
    exploded = df.filter(pl.col("matched_entity_ids") != "") \
                 .select(pl.col("matched_entity_ids").str.split(by=",").alias("m_id")) \
                 .explode("m_id") \
                 .with_columns(pl.col("m_id").str.strip_chars())
                 
    s2_matches = exploded.filter(pl.col("m_id").str.starts_with("S2-")).height
    s3_matches = exploded.filter(pl.col("m_id").str.starts_with("S3-")).height
    
    t1 = time.time()
    print(f"Processed Ground Truth {total_s1:,} rows in {t1 - t0:.2f}s", flush=True)
    print(f"Singletons (0 matches): {singletons:,} ({singletons/total_s1*100:.2f}%)", flush=True)
    print(f"Single match (1 match): {single_match:,} ({single_match/total_s1*100:.2f}%)", flush=True)
    print(f"Multi matches (2+ matches): {multi_match:,} ({multi_match/total_s1*100:.2f}%)", flush=True)
    print(f"Total S2 matched records: {s2_matches:,}", flush=True)
    print(f"Total S3 matched records: {s3_matches:,}", flush=True)
    print(f"Match count distribution: {match_dist}", flush=True)
    
    return {
        "total_s1": total_s1,
        "singletons": singletons,
        "single_match": single_match,
        "multi_match": multi_match,
        "s2_matches": s2_matches,
        "s3_matches": s3_matches,
        "match_count_dist": match_dist
    }

if __name__ == "__main__":
    results = {}
    files_to_inspect = [
        os.path.join(TRAIN_DIR, "train_source1.tsv"),
        os.path.join(TRAIN_DIR, "train_source2.tsv"),
        os.path.join(TRAIN_DIR, "train_source3.tsv"),
        os.path.join(TEST_DIR, "test_source1.tsv"),
        os.path.join(TEST_DIR, "test_source2.tsv"),
        os.path.join(TEST_DIR, "test_source3.tsv"),
    ]
    
    for fpath in files_to_inspect:
        if os.path.exists(fpath):
            results[os.path.basename(fpath)] = inspect_file(fpath)
            
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    if os.path.exists(gt_path):
        results["ground_truth"] = inspect_ground_truth(gt_path)
        
    with open(os.path.join(OUTPUT_DIR, "data_inspection_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nData inspection complete. Summary saved to experiments/data_inspection_summary.json.", flush=True)
