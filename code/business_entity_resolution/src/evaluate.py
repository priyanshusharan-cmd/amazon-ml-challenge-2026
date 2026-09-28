import pandas as pd

def compute_f05(pred_ids, true_ids):
    pred_set = set(pred_ids) if pred_ids else set()
    true_set = set(true_ids) if true_ids else set()
    
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
        
    if len(pred_set) == 0:
        return 0.0
        
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0 and recall == 0:
        return 0.0
        
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def evaluate(pred_df, gt_df):
    """
    Evaluates predictions against ground truth.
    Both dataframes must have 'source1_entity_id' and 'matched_entity_ids'.
    """
    merged = pd.merge(gt_df, pred_df, on='source1_entity_id', how='left', suffixes=('_true', '_pred'))
    
    scores = []
    for _, row in merged.iterrows():
        true_val = row.get('matched_entity_ids_true', '')
        pred_val = row.get('matched_entity_ids_pred', '')
        
        true_ids = str(true_val).split(',') if pd.notna(true_val) and str(true_val).strip() else []
        pred_ids = str(pred_val).split(',') if pd.notna(pred_val) and str(pred_val).strip() else []
        
        score = compute_f05(pred_ids, true_ids)
        scores.append(score)
        
    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    return macro_f05

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate F0.5 macro score.")
    parser.add_argument("--preds", required=True, help="Path to predictions TSV")
    parser.add_argument("--gt", required=True, help="Path to ground truth TSV")
    args = parser.parse_args()
    
    pred_df = pd.read_csv(args.preds, sep="\t")
    gt_df = pd.read_csv(args.gt, sep="\t")
    
    score = evaluate(pred_df, gt_df)
    print(f"Macro F0.5 Score: {score:.5f}")
