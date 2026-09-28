import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_splits(data_dir, output_dir, val_size=0.2, random_state=42):
    print("Loading datasets...")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    
    gt_df = pd.read_csv(gt_path, sep="\t")
    s1_df = pd.read_csv(s1_path, sep="\t")
    
    # Merge to get country for stratification
    merged = pd.merge(gt_df, s1_df[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id', how='left')
    
    # Determine if it has matches
    merged['has_match'] = merged['matched_entity_ids'].notna() & (merged['matched_entity_ids'] != '')
    
    # Create stratification column
    merged['stratify_col'] = merged['country'].astype(str) + "_" + merged['has_match'].astype(str)
    
    print(f"Total entities: {len(merged)}")
    print("Stratification distribution:")
    print(merged['stratify_col'].value_counts())
    
    # Stratified split
    train_split, val_split = train_test_split(
        merged, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=merged['stratify_col']
    )
    
    print(f"\nTrain size: {len(train_split)}, Val size: {len(val_split)}")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep only the original columns
    train_gt = train_split[['source1_entity_id', 'matched_entity_ids']]
    val_gt = val_split[['source1_entity_id', 'matched_entity_ids']]
    
    train_gt.to_csv(os.path.join(output_dir, "train_gt_split.tsv"), sep="\t", index=False)
    val_gt.to_csv(os.path.join(output_dir, "val_gt_split.tsv"), sep="\t", index=False)
    print(f"Saved splits to {output_dir}")

if __name__ == "__main__":
    # Run from the root of student_resource
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir = os.path.join(base_dir, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
    output_dir = os.path.join(base_dir, "6ab10eb3b23ba_student_resource", "student_resource", "dataset", "splits")
    create_splits(data_dir, output_dir)
