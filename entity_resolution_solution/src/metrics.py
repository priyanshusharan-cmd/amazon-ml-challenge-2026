"""
Evaluation Metrics Engine for Macro F0.5 and Entity Resolution Statistics.
Evaluates per-S1 entity precision, recall, and F0.5 according to challenge rules.
"""

from collections import defaultdict
import numpy as np

def compute_macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    all_s1_ids: list[str]
) -> dict:
    """
    Computes challenge evaluation metrics:
      - Macro F0.5
      - Macro Precision
      - Macro Recall
      - Singleton Precision / Accuracy
      - Breakdown by S1-S2 vs S1-S3 matches
    """
    f05_scores = []
    precisions = []
    recalls = []
    
    singleton_count = 0
    singleton_correct = 0
    
    # Per-pair statistics
    s2_tp, s2_fp, s2_fn = 0, 0, 0
    s3_tp, s3_fp, s3_fn = 0, 0, 0
    
    total_preds = 0
    
    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        total_preds += len(pred_set)
        
        # Breakdown S2 / S3
        t_s2 = {x for x in true_set if x.startswith("S2-")}
        p_s2 = {x for x in pred_set if x.startswith("S2-")}
        s2_tp += len(p_s2 & t_s2)
        s2_fp += len(p_s2 - t_s2)
        s2_fn += len(t_s2 - p_s2)
        
        t_s3 = {x for x in true_set if x.startswith("S3-")}
        p_s3 = {x for x in pred_set if x.startswith("S3-")}
        s3_tp += len(p_s3 & t_s3)
        s3_fp += len(p_s3 - t_s3)
        s3_fn += len(t_s3 - p_s3)
        
        # Singleton handling
        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0) # Or 0.0 depending on convention, but F0.5 is 0.0
        else: # Non-singleton
            if len(pred_set) == 0:
                f05_scores.append(0.0)
                precisions.append(1.0)
                recalls.append(0.0)
            else:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                precisions.append(p)
                recalls.append(r)
                
                denom = (0.25 * p + r)
                if denom > 0:
                    f05 = (1.25 * p * r) / denom
                else:
                    f05 = 0.0
                f05_scores.append(f05)
                
    # Per-pair macro / aggregate metrics
    s2_prec = s2_tp / (s2_tp + s2_fp) if (s2_tp + s2_fp) > 0 else 0.0
    s2_rec = s2_tp / (s2_tp + s2_fn) if (s2_tp + s2_fn) > 0 else 0.0
    s2_f05 = (1.25 * s2_prec * s2_rec) / (0.25 * s2_prec + s2_rec) if (0.25 * s2_prec + s2_rec) > 0 else 0.0

    s3_prec = s3_tp / (s3_tp + s3_fp) if (s3_tp + s3_fp) > 0 else 0.0
    s3_rec = s3_tp / (s3_tp + s3_fn) if (s3_tp + s3_fn) > 0 else 0.0
    s3_f05 = (1.25 * s3_prec * s3_rec) / (0.25 * s3_prec + s3_rec) if (0.25 * s3_prec + s3_rec) > 0 else 0.0

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_count": singleton_count,
        "singleton_precision": float(singleton_correct / singleton_count) if singleton_count > 0 else 0.0,
        "total_predicted_matches": total_preds,
        "s1_s2_pair": {
            "precision": s2_prec,
            "recall": s2_rec,
            "f05": s2_f05,
            "tp": s2_tp, "fp": s2_fp, "fn": s2_fn
        },
        "s1_s3_pair": {
            "precision": s3_prec,
            "recall": s3_rec,
            "f05": s3_f05,
            "tp": s3_tp, "fp": s3_fp, "fn": s3_fn
        }
    }
