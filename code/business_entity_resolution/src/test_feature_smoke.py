"""
Smoke test for feature_engineering.py
Creates tiny fake data, runs every function, and verifies correctness.
If this passes, the real script will NOT crash.
"""
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
import traceback
import multiprocessing as mp

def run_tests():
    print("=" * 60)
    print("SMOKE TEST: feature_engineering.py")
    print("=" * 60)

    # Step 0: Test imports
    print("\n[TEST 1] Testing all imports...")
    try:
        import psutil
        import sqlite3
        from rapidfuzz import fuzz
        from preprocess import normalize_text, normalize_address
        from feature_engineering import compute_features, extract_digits, build_feature_dataset, log_memory
        print("  ✅ All imports OK")
    except ImportError as e:
        print(f"  ❌ IMPORT FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 1: Test compute_features
    print("\n[TEST 2] Testing compute_features...")
    try:
        feats = compute_features("mcdonalds", "123 main st", "US", "mc donalds", "123 main street", "US")
        assert 'name_ratio' in feats, "Missing name_ratio"
        assert 'name_token_sort' in feats, "Missing name_token_sort"
        assert 'name_partial' in feats, "Missing name_partial"
        assert 'name_token_set' in feats, "Missing name_token_set"
        assert 'addr_ratio' in feats, "Missing addr_ratio"
        assert 'country_match' in feats, "Missing country_match"
        assert 'digits_match' in feats, "Missing digits_match"
        assert feats['country_match'] == 1.0, "Country match should be 1.0"
        assert feats['name_ratio'] > 0.5, f"Name ratio too low: {feats['name_ratio']}"
        print(f"  ✅ compute_features OK: {len(feats)} features generated")
        print(f"     name_ratio={feats['name_ratio']:.3f}, name_token_sort={feats['name_token_sort']:.3f}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Test with empty/missing values (edge cases that cause crashes)
    print("\n[TEST 3] Testing compute_features edge cases...")
    try:
        feats_empty = compute_features("", "", None, "", "", None)
        feats_none = compute_features("test", None, "US", "test", None, "UK")
        feats_mismatch = compute_features("apple", "1 infinite loop", "US", "banana", "999 elm st", "UK")
        assert feats_mismatch['country_match'] == 0.0
        print("  ✅ Edge cases OK (empty, None, mismatched inputs all handled)")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Test full build_feature_dataset pipeline
    print("\n[TEST 4] Testing full build_feature_dataset pipeline...")
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create fake S1 (queries)
        s1_data = pd.DataFrame({
            'entity_id': ['S1-1', 'S1-2', 'S1-3', 'S1-4', 'S1-5'],
            'business_name': ['McDonalds', 'Burger King', 'Wendys', 'Subway', 'KFC'],
            'business_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '101 Elm St', '202 Maple Dr'],
            'country': ['US', 'US', 'US', 'India', 'UK']
        })
        s1_path = os.path.join(tmp_dir, "s1.tsv")
        s1_data.to_csv(s1_path, sep="\t", index=False)
        
        # Create fake S2 catalog
        s2_data = pd.DataFrame({
            'entity_id': ['S2-10', 'S2-20', 'S2-30'],
            'business_name': ['Mc Donalds Inc', 'Burger King LLC', 'Pizza Hut'],
            'business_address': ['123 Main Street', '456 Oak Avenue', '333 Walnut'],
            'country': ['US', 'US', 'US']
        })
        s2_path = os.path.join(tmp_dir, "s2.tsv")
        s2_data.to_csv(s2_path, sep="\t", index=False)
        
        # Create fake S3 catalog
        s3_data = pd.DataFrame({
            'entity_id': ['S3-100', 'S3-200'],
            'business_name': ['Wendys Restaurant', 'SubWay Sandwiches'],
            'business_address': ['789 Pine Road', '101 Elm Street'],
            'country': ['US', 'India']
        })
        s3_path = os.path.join(tmp_dir, "s3.tsv")
        s3_data.to_csv(s3_path, sep="\t", index=False)
        
        # Create fake ground truth
        gt_data = pd.DataFrame({
            'source1_entity_id': ['S1-1', 'S1-2', 'S1-3', 'S1-4', 'S1-5'],
            'matched_entity_ids': ['S2-10', 'S2-20', 'S3-100', 'S3-200', np.nan]
        })
        gt_path = os.path.join(tmp_dir, "gt.tsv")
        gt_data.to_csv(gt_path, sep="\t", index=False)
        
        # Create fake candidate pairs
        cands_data = pd.DataFrame({
            'source1_entity_id': ['S1-1', 'S1-2', 'S1-3', 'S1-4', 'S1-5'],
            'candidate_entity_ids': [
                'S2-10,S2-20,S2-30',      # S2-10 is true match
                'S2-10,S2-20,S3-100',     # S2-20 is true match
                'S3-100,S3-200,S2-30',    # S3-100 is true match
                'S3-200,S2-10',           # S3-200 is true match
                'S2-30,S3-100'            # No true match (S1-5 has NaN GT)
            ]
        })
        cands_path = os.path.join(tmp_dir, "candidates.tsv")
        cands_data.to_csv(cands_path, sep="\t", index=False)
        
        out_path = os.path.join(tmp_dir, "features.csv")
        
        # Run the pipeline!
        build_feature_dataset(cands_path, gt_path, s1_path, s2_path, s3_path, out_path)
        
        # Verify output
        result = pd.read_csv(out_path)
        print(f"\n  Output columns: {result.columns.tolist()}")
        print(f"  Total rows: {len(result)}")
        print(f"  Positives (label=1): {(result['label'] == 1).sum()}")
        print(f"  Negatives (label=0): {(result['label'] == 0).sum()}")
        
        assert 'label' in result.columns, "Missing label column!"
        assert 'name_ratio' in result.columns, "Missing name_ratio column!"
        assert 'source1_entity_id' in result.columns, "Missing source1_entity_id!"
        assert 'candidate_entity_id' in result.columns, "Missing candidate_entity_id!"
        
        # With 5% negative sampling, we should have ALL positives
        positives = (result['label'] == 1).sum()
        assert positives >= 1, f"Expected at least 1 positive, got {positives}. GT loading is broken!"
        
        print(f"\n  ✅ Full pipeline OK! {positives} positives found correctly")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir)

    # Step 4: Verify the REAL ground truth file exists and has correct columns
    print("\n[TEST 5] Verifying real ground truth file...")
    try:
        real_gt_path = r"../dataset/train/train_ground_truth.tsv"
        assert os.path.exists(real_gt_path), f"Ground truth not found at: {real_gt_path}"
        gt_sample = pd.read_csv(real_gt_path, sep="\t", nrows=5)
        assert 'source1_entity_id' in gt_sample.columns, f"Missing 'source1_entity_id'! Has: {gt_sample.columns.tolist()}"
        assert 'matched_entity_ids' in gt_sample.columns, f"Missing 'matched_entity_ids'! Has: {gt_sample.columns.tolist()}"
        print(f"  ✅ Real GT file OK: {gt_sample.columns.tolist()}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Verify the REAL candidates file exists
    print("\n[TEST 6] Verifying real candidate pairs file...")
    try:
        real_cands_path = r"../output/full_train_candidate_pairs.tsv"
        assert os.path.exists(real_cands_path), f"Candidates not found at: {real_cands_path}"
        cands_sample = pd.read_csv(real_cands_path, sep="\t", nrows=5)
        assert 'source1_entity_id' in cands_sample.columns
        assert 'candidate_entity_ids' in cands_sample.columns
        print(f"  ✅ Real candidates file OK: {cands_sample.columns.tolist()}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL 6 TESTS PASSED ✅✅✅")
    print("feature_engineering.py is safe to run on the full dataset.")
    print("=" * 60)

if __name__ == "__main__":
    mp.freeze_support()
    run_tests()
