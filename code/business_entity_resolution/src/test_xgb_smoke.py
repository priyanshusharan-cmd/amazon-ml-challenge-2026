import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb

print(f"XGBoost Version: {xgb.__version__}")

# Load a small sample from our actual feature dataset
csv_path = r"../output/full_train_features_v3.csv"
print(f"Reading 10,000 sample rows from {os.path.basename(csv_path)}...")

FEATURE_COLS = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_compact_match', 'name_acronym_match', 'is_addr_missing',
    'street_num_match', 'street_name_sim', 'city_state_sim',
    'addr_token_sort', 'digits_match', 'country_match',
    'is_dba_pattern', 'source_origin'
]

df_sample = pd.read_csv(csv_path, nrows=10000, usecols=FEATURE_COLS + ['label'])
X = df_sample[FEATURE_COLS].to_numpy(dtype=np.float32)
y = df_sample['label'].to_numpy(dtype=np.int8)

print(f"Sample data loaded: X shape={X.shape}, y shape={y.shape}, Positives: {(y == 1).sum()}, Negatives: {(y == 0).sum()}")

# Test DMatrix creation
t0 = time.time()
dtrain = xgb.DMatrix(X[:8000], label=y[:8000], feature_names=FEATURE_COLS)
dval = xgb.DMatrix(X[8000:], label=y[8000:], feature_names=FEATURE_COLS)
print(f"DMatrix created in {time.time() - t0:.3f}s")

# Test Exact Training Parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 10,
    'learning_rate': 0.03,
    'subsample': 0.80,
    'colsample_bytree': 0.80,
    'min_child_weight': 40,
    'n_jobs': 2
}

print("Running 10-round smoke test with exact XGBoost configuration...")
t0 = time.time()
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=10,
    evals=[(dtrain, 'train'), (dval, 'val')],
    verbose_eval=5
)
train_time = time.time() - t0
print(f"XGBoost 10 rounds trained successfully in {train_time:.2f}s!")

# Test Prediction
preds = booster.predict(dval)
print(f"Predicted probabilities on validation slice: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")
print(">>> XGBOOST SMOKE TEST PASSED 100% CLEANLY! <<<")
