import os
import gc
import json
import time
import psutil
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

FEATURE_COLS = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_compact_match', 'name_acronym_match', 'is_addr_missing',
    'street_num_match', 'street_name_sim', 'city_state_sim',
    'addr_token_sort', 'digits_match', 'country_match',
    'is_dba_pattern', 'source_origin'
]

COL_DTYPES = {
    'name_ratio': np.float32,
    'name_token_sort': np.float32,
    'name_token_set': np.float32,
    'name_partial': np.float32,
    'name_compact_match': np.float32,
    'name_acronym_match': np.float32,
    'is_addr_missing': np.float32,
    'street_num_match': np.float32,
    'street_name_sim': np.float32,
    'city_state_sim': np.float32,
    'addr_token_sort': np.float32,
    'digits_match': np.float32,
    'country_match': np.float32,
    'is_dba_pattern': np.float32,
    'source_origin': np.float32,
    'source1_entity_id': 'str',
    'candidate_entity_id': 'str',
    'label': np.int8
}

def log_memory(label=""):
    mem = psutil.virtual_memory()
    print(f"  [MEM {label}] Used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}%)", flush=True)

def compute_f05_single(pred_set, true_set):
    """Computes entity-level F0.5 according to the competition specification."""
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


def scan_optimal_threshold(val_csv_path, val_probs, val_gt_path, model_label="Model"):
    """
    Scans probability thresholds from 0.50 to 0.98 with exact macro-averaged F0.5
    over all entities in the validation set, including true singletons.
    Streams val_csv to preserve memory.
    """
    print(f"\n--- EXACT MACRO-AVERAGED F0.5 THRESHOLD SCANNING ({model_label.upper()}) ---")
    log_memory(f"Threshold Scan {model_label}")
    
    gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
    all_s1_ids = gt_df['source1_entity_id'].tolist()
    total_val_entities = len(all_s1_ids)
    
    gt_map = {}
    singleton_entities = set()
    for _, row in gt_df.iterrows():
        sid = row['source1_entity_id']
        matches_val = row.get('matched_entity_ids', '')
        if pd.notna(matches_val) and str(matches_val).strip() and str(matches_val).strip() != 'nan':
            gt_map[sid] = set(m.strip() for m in str(matches_val).split(',') if m.strip())
        else:
            gt_map[sid] = set()
            singleton_entities.add(sid)
            
    val_probs_arr = np.array(val_probs, dtype=np.float32)
    cand_by_s1 = {}
    chunk_size = 500000
    idx = 0
    
    for chunk in pd.read_csv(val_csv_path, chunksize=chunk_size, usecols=['source1_entity_id', 'candidate_entity_id'], dtype=str):
        n = len(chunk)
        chunk_probs = val_probs_arr[idx:idx+n]
        mask = chunk_probs >= 0.40
        if np.any(mask):
            filtered_s1 = chunk['source1_entity_id'].to_numpy()[mask]
            filtered_cand = chunk['candidate_entity_id'].to_numpy()[mask]
            filtered_p = chunk_probs[mask]
            for s1_id, cand_id, p in zip(filtered_s1, filtered_cand, filtered_p):
                if s1_id not in cand_by_s1:
                    cand_by_s1[s1_id] = []
                cand_by_s1[s1_id].append((cand_id, float(p)))
        idx += n
        
    del val_probs_arr
    gc.collect()
    
    best_threshold = 0.50
    best_macro_f05 = -1.0
    thresholds = np.arange(0.50, 0.98, 0.01)
    results = []
    
    print(f"Scanning decision thresholds for {model_label} (Total candidates >= 0.40: {sum(len(v) for v in cand_by_s1.values()):,})...")
    print(f"{'Threshold':>10} | {'Macro F0.5':>12} | {'Entities w/ Preds':>18} | {'Singletons Kept':>16}")
    print("-" * 65)
    
    for t in thresholds:
        t_val = round(float(t), 2)
        score_sum = 0.0
        entities_with_preds = 0
        singletons_survived = 0
        
        for s1_id, cands in cand_by_s1.items():
            pred_set = set(cid for cid, p in cands if p >= t_val)
            if pred_set:
                true_set = gt_map.get(s1_id, set())
                score = compute_f05_single(pred_set, true_set)
                score_sum += score
                entities_with_preds += 1
            else:
                if s1_id in singleton_entities:
                    score_sum += 1.0
                    singletons_survived += 1
                    
        singletons_no_cands = len(singleton_entities - set(cand_by_s1.keys()))
        score_sum += singletons_no_cands
        singletons_survived += singletons_no_cands
        
        macro_f05 = score_sum / total_val_entities
        results.append((t_val, macro_f05, entities_with_preds, singletons_survived))
        
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = t_val
            
        if int(round(t_val * 100)) % 5 == 0 or t_val in [0.60, 0.68, 0.75, 0.80, 0.85, 0.90]:
            marker = "  <<< MAX" if t_val == best_threshold else ""
            print(f"{t_val:>10.2f} | {macro_f05:>12.5f} | {entities_with_preds:>18,} | {singletons_survived:>16,}{marker}")
            
    print("-" * 65)
    print(f"\n>>> [{model_label.upper()}] OPTIMAL THRESHOLD: {best_threshold:.2f} with Macro F0.5 = {best_macro_f05:.5f} <<<")
    return best_threshold, best_macro_f05, results


def load_dataset(csv_path, chunk_size=500000):
    log_memory(f"Loading {os.path.basename(csv_path)}")
    X_chunks = []
    y_chunks = []
    
    total_rows = 0
    t0 = time.time()
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=FEATURE_COLS + ['label'], dtype=COL_DTYPES):
        total_rows += len(chunk)
        X_chunks.append(chunk[FEATURE_COLS].to_numpy(dtype=np.float32))
        y_chunks.append(chunk['label'].to_numpy(dtype=np.int8))
            
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    del X_chunks, y_chunks
    gc.collect()
    
    elapsed = time.time() - t0
    print(f"Loaded {total_rows:,} rows from {csv_path} in {elapsed:.1f}s ({X.nbytes / (1024**2):.1f} MB matrix).")
    log_memory("After array consolidation")
    
    return X, y


def train_and_evaluate(train_csv, val_csv, val_gt_path, out_dir):
    log_memory("Pipeline Start")
    
    out_lgb_model = os.path.join(out_dir, "lgbm_model_v3.txt")
    out_xgb_model = os.path.join(out_dir, "xgb_model_v3.json")
    out_config = os.path.join(out_dir, "threshold_config_v3.json")
    num_cores = min(6, os.cpu_count())
    val_probs_lgb_path = os.path.join(out_dir, "val_probs_lgb_v3.npy")
    val_probs_xgb_path = os.path.join(out_dir, "val_probs_xgb_v3.npy")
    
    # -------------------------------------------------------------
    # MODEL 1: High-Capacity LightGBM Booster
    # -------------------------------------------------------------
    if os.path.exists(out_lgb_model) and os.path.exists(val_probs_lgb_path):
        print(f"\n[LightGBM] Found existing trained model ({out_lgb_model}) and predictions ({val_probs_lgb_path}). Loading...")
        val_probs_lgb = np.load(val_probs_lgb_path)
    else:
        print("\n" + "="*60)
        print("=== MODEL 1: Training High-Capacity LightGBM Booster (800 Trees) ===")
        print("="*60)
        print(f"\n=== Loading Training & Validation Data for LightGBM ===")
        X_train, y_train = load_dataset(train_csv)
        X_val, y_val = load_dataset(val_csv)
        
        train_data_lgb = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, free_raw_data=True)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data_lgb, free_raw_data=False)
        del X_train, y_train
        gc.collect()
        log_memory("After LGBM Dataset Construction")
        
        params_lgb = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 127,
            'max_depth': 10,
            'feature_fraction': 0.80,
            'bagging_fraction': 0.80,
            'bagging_freq': 1,
            'min_child_samples': 40,
            'verbose': -1,
            'n_jobs': num_cores
        }
        
        t0 = time.time()
        lgb_booster = lgb.train(
            params_lgb,
            train_data_lgb,
            num_boost_round=800,
            valid_sets=[train_data_lgb, val_data_lgb],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )
        print(f"LightGBM training finished in {time.time() - t0:.1f}s!")
        lgb_booster.save_model(out_lgb_model)
        print(f"Saved LightGBM model to: {out_lgb_model}")
        
        val_probs_lgb = lgb_booster.predict(X_val)
        np.save(val_probs_lgb_path, val_probs_lgb)
        print(f"Saved LightGBM validation predictions to: {val_probs_lgb_path}")
        
        del train_data_lgb, val_data_lgb, lgb_booster, X_val, y_val
        gc.collect()
        log_memory("After LGBM Cleanup")
    
    # -------------------------------------------------------------
    # MODEL 2: High-Capacity XGBoost Hist Booster
    # -------------------------------------------------------------
    if os.path.exists(out_xgb_model) and os.path.exists(val_probs_xgb_path):
        print(f"\n[XGBoost] Found existing trained model ({out_xgb_model}) and predictions ({val_probs_xgb_path}). Loading...")
        val_probs_xgb = np.load(val_probs_xgb_path)
    else:
        print("\n" + "="*60)
        print("=== MODEL 2: Training High-Capacity XGBoost Booster (800 Trees) ===")
        print("="*60)
        print(f"\n=== Loading Training & Validation Data for XGBoost ===")
        X_train, y_train = load_dataset(train_csv)
        X_val, y_val = load_dataset(val_csv)
        
        t0 = time.time()
        dtrain_xgb = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
        del X_train, y_train
        gc.collect()
        
        dval_xgb = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
        del X_val, y_val
        gc.collect()
        log_memory("After XGBoost DMatrix Construction")
        
        params_xgb = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 10,
            'learning_rate': 0.03,
            'subsample': 0.80,
            'colsample_bytree': 0.80,
            'min_child_weight': 40,
            'n_jobs': num_cores
        }
        
        xgb_booster = xgb.train(
            params_xgb,
            dtrain_xgb,
            num_boost_round=800,
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'val')],
            early_stopping_rounds=40,
            verbose_eval=50
        )
        print(f"XGBoost training finished in {time.time() - t0:.1f}s!")
        xgb_booster.save_model(out_xgb_model)
        print(f"Saved XGBoost model to: {out_xgb_model}")
        
        val_probs_xgb = xgb_booster.predict(dval_xgb)
        np.save(val_probs_xgb_path, val_probs_xgb)
        print(f"Saved XGBoost validation predictions to: {val_probs_xgb_path}")
        
        del dtrain_xgb, dval_xgb, xgb_booster
        gc.collect()
        log_memory("After XGBoost Cleanup")
    
    # -------------------------------------------------------------
    # EVALUATION & ENSEMBLE BLENDING
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("=== 3. Evaluating Individual Models & Ensemble Blend ===")
    print("="*60)
    
    # 1. Evaluate LightGBM alone
    t_lgb, f05_lgb, _ = scan_optimal_threshold(val_csv, val_probs_lgb, val_gt_path, model_label="LightGBM")
    
    # 2. Evaluate XGBoost alone
    t_xgb, f05_xgb, _ = scan_optimal_threshold(val_csv, val_probs_xgb, val_gt_path, model_label="XGBoost")
    
    # 3. Evaluate Ensemble Blend (50/50 blend)
    val_probs_ens = 0.5 * val_probs_lgb + 0.5 * val_probs_xgb
    t_ens, f05_ens, _ = scan_optimal_threshold(val_csv, val_probs_ens, val_gt_path, model_label="LightGBM+XGBoost Ensemble")
    
    print("\n" + "="*60)
    print("=== FINAL MODEL COMPARISON ===")
    print(f"  LightGBM Alone:             Macro F0.5 = {f05_lgb:.5f} (Threshold: {t_lgb:.2f})")
    print(f"  XGBoost Alone:              Macro F0.5 = {f05_xgb:.5f} (Threshold: {t_xgb:.2f})")
    print(f"  Ensemble (LGBM + XGBoost):  Macro F0.5 = {f05_ens:.5f} (Threshold: {t_ens:.2f})")
    print("="*60)
    
    # Select winning model configuration
    if f05_ens >= max(f05_lgb, f05_xgb):
        chosen_mode = "ensemble"
        chosen_thresh = t_ens
        chosen_f05 = f05_ens
    elif f05_xgb > f05_lgb:
        chosen_mode = "xgboost"
        chosen_thresh = t_xgb
        chosen_f05 = f05_xgb
    else:
        chosen_mode = "lgbm"
        chosen_thresh = t_lgb
        chosen_f05 = f05_lgb
        
    print(f"\n>>> Selected Winning Configuration: {chosen_mode.upper()} with Macro F0.5 = {chosen_f05:.5f} (Threshold: {chosen_thresh:.2f}) <<<")
    
    singleton_cutoff = max(0.75, round(chosen_thresh - 0.04, 2))
    config = {
        "model_mode": chosen_mode,
        "optimal_threshold": chosen_thresh,
        "singleton_cutoff": singleton_cutoff,
        "validation_macro_f05": chosen_f05,
        "lgbm_score": f05_lgb,
        "xgb_score": f05_xgb,
        "ensemble_score": f05_ens,
        "features": FEATURE_COLS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(out_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Saved complete config to: {out_config}")


if __name__ == "__main__":
    out_dir = r"../output"
    train_csv = os.path.join(out_dir, "full_train_features_v3.csv")
    val_csv = os.path.join(out_dir, "full_val_features_v3.csv")
    val_gt = os.path.join(out_dir, "val_gt_split.tsv")
    
    train_and_evaluate(train_csv, val_csv, val_gt, out_dir)
