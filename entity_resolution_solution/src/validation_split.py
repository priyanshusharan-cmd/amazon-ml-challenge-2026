"""
Validation Split Module.
Creates a stratified train/validation split at the Source-1 entity level.
Stratifies on (country, match_class) where match_class is ('0', '1', '2+').
"""

import os
import polars as pl
from sklearn.model_selection import train_test_split

def create_stratified_split(
    s1_path: str,
    gt_path: str,
    output_dir: str,
    val_size: float = 0.2,
    random_seed: int = 42
):
    print(f"Creating stratified validation split (val_size={val_size}, seed={random_seed})...", flush=True)
    
    # 1. Load S1 metadata (entity_id, country)
    df_s1 = pl.read_csv(s1_path, separator="\t", has_header=True, schema_overrides={"entity_id": pl.Utf8, "country": pl.Utf8}) \
              .select(["entity_id", "country"])
              
    # 2. Load GT metadata (source1_entity_id, matched_entity_ids)
    df_gt = pl.read_csv(gt_path, separator="\t", has_header=True, schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}) \
              .with_columns(pl.col("matched_entity_ids").fill_null(""))
              
    # Compute match_class
    df_gt = df_gt.with_columns(
        pl.when(pl.col("matched_entity_ids").str.strip_chars() == "")
        .then(pl.lit("0"))
        .when(pl.col("matched_entity_ids").str.count_matches(",") == 0)
        .then(pl.lit("1"))
        .otherwise(pl.lit("2+"))
        .alias("match_class")
    )
    
    # Join S1 and GT
    df_meta = df_s1.join(df_gt.select(["source1_entity_id", "match_class"]), left_on="entity_id", right_on="source1_entity_id", how="inner")
    
    # Stratification key
    df_meta = df_meta.with_columns(
        (pl.col("country") + "_" + pl.col("match_class")).alias("strat_key")
    )
    
    pdf = df_meta.to_pandas()
    
    train_df, val_df = train_test_split(
        pdf,
        test_size=val_size,
        random_state=random_seed,
        stratify=pdf["strat_key"]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_s1 = pl.from_pandas(train_df[["entity_id", "country", "match_class", "strat_key"]])
    val_s1 = pl.from_pandas(val_df[["entity_id", "country", "match_class", "strat_key"]])
    
    train_path = os.path.join(output_dir, "train_s1_ids.parquet")
    val_path = os.path.join(output_dir, "val_s1_ids.parquet")
    
    train_s1.write_parquet(train_path)
    val_s1.write_parquet(val_path)
    
    print(f"Train S1 Entities: {len(train_s1):,}")
    print(f"Validation S1 Entities: {len(val_s1):,}")
    dist_dict = val_s1.group_by("strat_key").len().sort("strat_key").to_dicts()
    print(f"Validation Stratification Distribution: {dist_dict}")
    
    return train_path, val_path

if __name__ == "__main__":
    TRAIN_DIR = r"c:\amazon_ml_challenge\student_resource\dataset\train"
    OUT_DIR = r"c:\amazon_ml_challenge\student_resource\entity_resolution_solution\experiments"
    
    create_stratified_split(
        s1_path=os.path.join(TRAIN_DIR, "train_source1.tsv"),
        gt_path=os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        output_dir=OUT_DIR,
        val_size=0.2,
        random_seed=42
    )
