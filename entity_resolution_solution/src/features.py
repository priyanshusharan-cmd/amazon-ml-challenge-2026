"""
Pairwise Feature Extraction Engine.
Generates similarity features for candidate pairs using RapidFuzz and set logic.
"""

import numpy as np
import polars as pl
from rapidfuzz import fuzz

def char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compute_pairwise_features_batch(
    names1: list[str],
    addrs1: list[str],
    names2: list[str],
    addrs2: list[str],
    cand_ids: list[str]
) -> np.ndarray:
    """
    Computes numerical feature matrix X for N candidate pairs.
    """
    N = len(names1)
    # 16 numerical features
    X = np.zeros((N, 16), dtype=np.float32)
    
    for i in range(N):
        n1, n2 = names1[i], names2[i]
        a1, a2 = addrs1[i], addrs2[i]
        cid = cand_ids[i]
        
        # Name features
        n1_len, n2_len = len(n1), len(n2)
        X[i, 0] = 1.0 if n1 == n2 else 0.0
        
        # Token Jaccard
        t1, t2 = set(n1.split()), set(n2.split())
        X[i, 1] = jaccard_similarity(t1, t2)
        X[i, 2] = len(t1 & t2)
        
        # Length ratios
        max_nlen = max(n1_len, n2_len)
        min_nlen = min(n1_len, n2_len)
        X[i, 3] = min_nlen / max_nlen if max_nlen > 0 else 0.0
        X[i, 4] = abs(n1_len - n2_len)
        
        # RapidFuzz string similarities
        X[i, 5] = fuzz.ratio(n1, n2) / 100.0
        X[i, 6] = fuzz.token_sort_ratio(n1, n2) / 100.0
        X[i, 7] = fuzz.token_set_ratio(n1, n2) / 100.0
        
        # Char 3-gram Jaccard
        X[i, 8] = jaccard_similarity(char_ngrams(n1, 3), char_ngrams(n2, 3))
        
        # Address missing indicators
        a1_empty = (len(a1) == 0)
        a2_empty = (len(a2) == 0)
        X[i, 9] = 1.0 if a1_empty else 0.0
        X[i, 10] = 1.0 if a2_empty else 0.0
        
        # Address features (if both present)
        if not a1_empty and not a2_empty:
            X[i, 11] = 1.0 if a1 == a2 else 0.0
            at1, at2 = set(a1.split()), set(a2.split())
            X[i, 12] = jaccard_similarity(at1, at2)
            X[i, 13] = fuzz.token_sort_ratio(a1, a2) / 100.0
            max_alen = max(len(a1), len(a2))
            X[i, 14] = min(len(a1), len(a2)) / max_alen if max_alen > 0 else 0.0
        else:
            X[i, 11] = 0.0
            X[i, 12] = 0.0
            X[i, 13] = 0.0
            X[i, 14] = 0.0
            
        # Source 3 indicator (S1-S3 pair vs S1-S2 pair)
        X[i, 15] = 1.0 if cid.startswith("S3-") else 0.0
        
    return X

FEATURE_NAMES = [
    "name_exact", "name_jaccard", "name_shared_tokens", "name_len_ratio", "name_len_diff",
    "name_fuzz_ratio", "name_token_sort_ratio", "name_token_set_ratio", "name_3gram_jaccard",
    "addr_missing_s1", "addr_missing_s2s3", "addr_exact", "addr_jaccard", "addr_token_sort_ratio",
    "addr_len_ratio", "is_s3"
]
