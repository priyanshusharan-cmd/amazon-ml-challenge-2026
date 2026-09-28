"""
Multi-Pass Blocking Engine for High Candidate Recall.
Combines exact name, clean name, sorted token pair, prefix+address number,
and rare token index blocking passes.
"""

import polars as pl
from collections import Counter

def generate_blocking_candidates(df_s1: pl.DataFrame, df_s2s3: pl.DataFrame) -> pl.DataFrame:
    """
    Executes multi-pass blocking between S1 records and S2/S3 records.
    Returns deduplicated (s1_id, cand_id) pairs.
    """
    print("Executing multi-pass blocking...", flush=True)
    candidate_dfs = []
    
    # Pass 1: Exact Country + Exact Full Name
    cands_p1 = df_s1.join(
        df_s2s3,
        on=["country", "name_norm"],
        how="inner",
        suffix="_cand"
    ).select([
        pl.col("entity_id").alias("s1_id"),
        pl.col("entity_id_cand").alias("cand_id")
    ])
    candidate_dfs.append(cands_p1)
    
    # Pass 2: Exact Country + Exact Clean Name (no legal suffixes)
    s1_clean = df_s1.filter((pl.col("name_clean") != "") & (pl.col("name_clean").is_not_null()))
    s2s3_clean = df_s2s3.filter((pl.col("name_clean") != "") & (pl.col("name_clean").is_not_null()))
    
    cands_p2 = s1_clean.join(
        s2s3_clean,
        on=["country", "name_clean"],
        how="inner",
        suffix="_cand"
    ).select([
        pl.col("entity_id").alias("s1_id"),
        pl.col("entity_id_cand").alias("cand_id")
    ])
    candidate_dfs.append(cands_p2)
    
    # Pass 3: Exact Country + Sorted First 2 Tokens of Clean Name
    s1_f2 = df_s1.filter((pl.col("first_2_tokens") != "") & (pl.col("first_2_tokens").is_not_null()))
    s2s3_f2 = df_s2s3.filter((pl.col("first_2_tokens") != "") & (pl.col("first_2_tokens").is_not_null()))
    
    cands_p3 = s1_f2.join(
        s2s3_f2,
        on=["country", "first_2_tokens"],
        how="inner",
        suffix="_cand"
    ).select([
        pl.col("entity_id").alias("s1_id"),
        pl.col("entity_id_cand").alias("cand_id")
    ])
    candidate_dfs.append(cands_p3)
    
    # Pass 4: Country + Prefix 4 + Address Number (when address number present)
    s1_p4 = df_s1.filter((pl.col("prefix_4") != "") & (pl.col("addr_num") != ""))
    s2s3_p4 = df_s2s3.filter((pl.col("prefix_4") != "") & (pl.col("addr_num") != ""))
    
    cands_p4 = s1_p4.join(
        s2s3_p4,
        on=["country", "prefix_4", "addr_num"],
        how="inner",
        suffix="_cand"
    ).select([
        pl.col("entity_id").alias("s1_id"),
        pl.col("entity_id_cand").alias("cand_id")
    ])
    candidate_dfs.append(cands_p4)

    # Combine all candidate passes
    all_cands = pl.concat(candidate_dfs).unique()
    print(f"Total deduplicated candidate pairs generated across 4 passes: {len(all_cands):,}", flush=True)
    return all_cands
