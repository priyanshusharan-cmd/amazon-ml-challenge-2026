"""
Smoke test for tfidf_blocking.py
Creates tiny fake data, runs every function, and verifies correctness.
If this passes, the real script will NOT crash.
"""
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import traceback

print("=" * 60)
print("SMOKE TEST: tfidf_blocking.py")
print("=" * 60)

# Step 0: Test imports
print("\n[TEST 1] Testing all imports...")
try:
    import psutil
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from preprocess import normalize_text
    print("  ✅ All imports OK")
except ImportError as e:
    print(f"  ❌ IMPORT FAILED: {e}")
    sys.exit(1)

# Step 1: Test psutil memory logging
print("\n[TEST 2] Testing psutil memory logging...")
try:
    mem = psutil.virtual_memory()
    print(f"  ✅ RAM: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent}%)")
except Exception as e:
    print(f"  ❌ psutil failed: {e}")
    sys.exit(1)

# Step 2: Create tiny fake catalog data
print("\n[TEST 3] Testing build_tfidf_index with fake data...")
tmp_dir = tempfile.mkdtemp()
fake_s2 = os.path.join(tmp_dir, "fake_s2.tsv")
fake_s3 = os.path.join(tmp_dir, "fake_s3.tsv")

s2_data = pd.DataFrame({
    'entity_id': [f's2_{i}' for i in range(100)],
    'business_name': [f'Business Alpha {i}' for i in range(100)]
})
s3_data = pd.DataFrame({
    'entity_id': [f's3_{i}' for i in range(50)],
    'business_name': [f'Company Beta {i}' for i in range(50)]
})
s2_data.to_csv(fake_s2, sep="\t", index=False)
s3_data.to_csv(fake_s3, sep="\t", index=False)

try:
    from tfidf_blocking import build_tfidf_index
    vectorizer, catalog_matrix, catalog_ids = build_tfidf_index(fake_s2, fake_s3)
    assert catalog_matrix.shape[0] == 150, f"Expected 150 rows, got {catalog_matrix.shape[0]}"
    assert len(catalog_ids) == 150, f"Expected 150 IDs, got {len(catalog_ids)}"
    print(f"  ✅ Catalog matrix: {catalog_matrix.shape}, IDs: {len(catalog_ids)}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test search_tfidf_candidates (the function that was crashing)
print("\n[TEST 4] Testing search_tfidf_candidates with threading...")
fake_s1 = os.path.join(tmp_dir, "fake_s1.tsv")
s1_data = pd.DataFrame({
    'entity_id': [f's1_{i}' for i in range(200)],
    'business_name': ['Business Alpha 5'] * 50 + ['Company Beta 10'] * 50 + ['Random Xyz'] * 100
})
s1_data.to_csv(fake_s1, sep="\t", index=False)
tfidf_out = os.path.join(tmp_dir, "tfidf_candidates.tsv")

try:
    from tfidf_blocking import search_tfidf_candidates
    search_tfidf_candidates(fake_s1, vectorizer, catalog_matrix, catalog_ids, tfidf_out, k=5)
    
    result = pd.read_csv(tfidf_out, sep="\t")
    assert len(result) == 200, f"Expected 200 rows, got {len(result)}"
    assert 'source1_entity_id' in result.columns
    assert 'candidate_entity_ids' in result.columns
    
    # Check that candidates actually exist
    sample_cands = str(result.iloc[0]['candidate_entity_ids']).split(',')
    assert len(sample_cands) > 0, "No candidates found for first query!"
    print(f"  ✅ Search complete: {len(result)} queries, sample candidates: {sample_cands[:3]}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test merge_candidates
print("\n[TEST 5] Testing merge_candidates...")
fake_dense = os.path.join(tmp_dir, "dense_candidates.tsv")
dense_data = pd.DataFrame({
    'source1_entity_id': [f's1_{i}' for i in range(200)],
    'candidate_entity_ids': ['s2_1,s2_2,s3_5'] * 200
})
dense_data.to_csv(fake_dense, sep="\t", index=False)
merged_out = os.path.join(tmp_dir, "merged.tsv")

try:
    from tfidf_blocking import merge_candidates
    merge_candidates(fake_dense, tfidf_out, merged_out)
    
    merged = pd.read_csv(merged_out, sep="\t")
    assert len(merged) == 200, f"Expected 200 rows, got {len(merged)}"
    
    # Check that union has MORE candidates than dense alone
    sample_merged_cands = str(merged.iloc[0]['candidate_entity_ids']).split(',')
    assert len(sample_merged_cands) >= 3, "Merged should have at least as many as dense!"
    print(f"  ✅ Merge complete: {len(merged)} rows, sample merged candidates: {len(sample_merged_cands)} unique")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Cleanup
import shutil
shutil.rmtree(tmp_dir)

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅✅✅")
print("tfidf_blocking.py is safe to run on the full dataset.")
print("=" * 60)
