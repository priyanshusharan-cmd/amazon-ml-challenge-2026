import pandas as pd
import numpy as np
import os
import gc
import psutil
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

from preprocess import normalize_text


def log_memory(label=""):
    mem = psutil.virtual_memory()
    print(f"  [MEM {label}] Used: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent}%)")


def build_tfidf_index(s2_path, s3_path):
    print("Loading catalog names...")
    catalog_ids = []
    catalog_names = []
    
    for path in [s2_path, s3_path]:
        for chunk in tqdm(pd.read_csv(path, sep="\t", chunksize=200000, usecols=['entity_id', 'business_name']), desc=f"Reading {os.path.basename(path)}"):
            chunk['business_name'] = chunk['business_name'].fillna("").apply(normalize_text)
            catalog_ids.extend(chunk['entity_id'].tolist())
            catalog_names.extend(chunk['business_name'].tolist())
    
    log_memory("After loading catalog names")
    
    print("Fitting TF-IDF Vectorizer (char 3-grams)...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=100000, dtype=np.float32)
    catalog_matrix = vectorizer.fit_transform(catalog_names)
    
    print(f"Catalog Matrix shape: {catalog_matrix.shape}, Memory: {catalog_matrix.data.nbytes / (1024**2):.1f} MB")
    
    del catalog_names
    gc.collect()
    log_memory("After fitting TF-IDF")
    
    return vectorizer, catalog_matrix, np.array(catalog_ids)


def _process_sub_chunk(query_ids, query_names, vectorizer, cat_matrix_T_csr, catalog_ids, k):
    """Process a sub-chunk of queries. cat_matrix_T_csr is the PRE-TRANSPOSED, PRE-CONVERTED CSR matrix."""
    query_matrix = vectorizer.transform(query_names)
    
    # Sparse dot product: CSR x CSR = no format conversion = no memory spike
    sim_matrix = query_matrix.dot(cat_matrix_T_csr)
    
    results = []
    for row_idx in range(sim_matrix.shape[0]):
        q_id = query_ids[row_idx]
        row_data = sim_matrix.getrow(row_idx)
        
        if row_data.nnz > 0:
            num_to_take = min(k, row_data.nnz)
            top_indices = np.argpartition(row_data.data, -num_to_take)[-num_to_take:]
            top_indices = top_indices[np.argsort(-row_data.data[top_indices])]
            real_col_indices = row_data.indices[top_indices]
            valid_cands = [str(catalog_ids[idx]) for idx in real_col_indices]
        else:
            valid_cands = []
            
        results.append({
            'source1_entity_id': str(q_id),
            'candidate_entity_ids': ",".join(valid_cands)
        })
    return results


def search_tfidf_candidates(s1_path, vectorizer, catalog_matrix, catalog_ids, out_path, k=30):
    print("Pre-computing CSR transpose of catalog matrix (one-time cost, prevents per-thread memory explosion)...")
    cat_matrix_T_csr = catalog_matrix.T.tocsr()
    print(f"  Transposed matrix shape: {cat_matrix_T_csr.shape}, Memory: {cat_matrix_T_csr.data.nbytes / (1024**2):.1f} MB")
    log_memory("After transpose")
    
    # Free the original catalog_matrix since we only need the transposed version
    del catalog_matrix
    gc.collect()
    
    print("Streaming S1 queries and performing sparse dot product search in PARALLEL...")
    
    header_written = False
    num_threads = 6  # Conservative: avoids memory spikes from too many simultaneous results
    chunk_size = 50000
    
    for chunk in tqdm(pd.read_csv(s1_path, sep="\t", chunksize=chunk_size, usecols=['entity_id', 'business_name']), desc="TF-IDF Search"):
        query_ids = chunk['entity_id'].tolist()
        query_names = chunk['business_name'].fillna("").apply(normalize_text).tolist()
        
        # Split into sub-chunks for threading
        sub_size = len(query_ids) // num_threads + 1
        sub_chunks = []
        for i in range(num_threads):
            start = i * sub_size
            end = min((i + 1) * sub_size, len(query_ids))
            if start < len(query_ids):
                sub_chunks.append((query_ids[start:end], query_names[start:end]))
        
        # Use ThreadPoolExecutor directly (no joblib, no loky, no process forking)
        # ThreadPoolExecutor shares memory safely. scipy.sparse.dot releases the GIL.
        all_results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(_process_sub_chunk, q_ids, q_names, vectorizer, cat_matrix_T_csr, catalog_ids, k)
                for q_ids, q_names in sub_chunks
            ]
            for future in futures:
                all_results.extend(future.result())
        
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(out_path, sep="\t", index=False, mode='w' if not header_written else 'a', header=not header_written)
        header_written = True
        
        del all_results, res_df
        gc.collect()
    
    print(f"\nTF-IDF candidate search complete! Saved to: {out_path}")


def merge_candidates(dense_path, tfidf_path, merged_path):
    print("\nMerging dense and TF-IDF candidates...")
    
    # Count lines first for progress bar
    print("  Counting dense candidates...")
    dense_count = sum(1 for _ in open(dense_path, 'r', encoding='utf-8')) - 1  # minus header
    print(f"  Dense candidates: {dense_count:,} rows")
    
    # Load TF-IDF candidates into a dictionary for safe lookup (not relying on row alignment)
    print("  Loading TF-IDF candidates into lookup dict...")
    tfidf_lookup = {}
    for chunk in tqdm(pd.read_csv(tfidf_path, sep="\t", chunksize=200000), desc="  Reading TF-IDF"):
        for _, row in chunk.iterrows():
            s1_id = str(row['source1_entity_id'])
            cands = str(row['candidate_entity_ids']) if pd.notna(row['candidate_entity_ids']) else ""
            tfidf_lookup[s1_id] = set(cands.split(',')) if cands else set()
    
    print(f"  TF-IDF lookup: {len(tfidf_lookup):,} queries")
    log_memory("After loading TF-IDF lookup")
    
    # Stream dense candidates and merge with TF-IDF lookup
    header_written = False
    merged_count = 0
    
    for d_chunk in tqdm(pd.read_csv(dense_path, sep="\t", chunksize=100000), desc="Merging", total=dense_count // 100000 + 1):
        merged_results = []
        
        for _, d_row in d_chunk.iterrows():
            s1_id = str(d_row['source1_entity_id'])
            
            # Dense candidates
            d_cands_str = str(d_row['candidate_entity_ids']) if pd.notna(d_row['candidate_entity_ids']) else ""
            d_cands = set(d_cands_str.split(',')) if d_cands_str else set()
            
            # TF-IDF candidates (safe lookup, no crash if missing)
            t_cands = tfidf_lookup.get(s1_id, set())
            
            union_cands = d_cands.union(t_cands)
            union_cands.discard('')  # Remove empty strings safely
            
            merged_results.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ",".join(list(union_cands))
            })
        
        merged_count += len(merged_results)
        res_df = pd.DataFrame(merged_results)
        res_df.to_csv(merged_path, sep="\t", index=False, mode='w' if not header_written else 'a', header=not header_written)
        header_written = True
        
        del merged_results, res_df
        gc.collect()
    
    print(f"  Merged {merged_count:,} total query rows")
    
    # Also add any TF-IDF-only queries that weren't in the dense file
    dense_ids = set()
    for chunk in pd.read_csv(dense_path, sep="\t", chunksize=200000, usecols=['source1_entity_id']):
        dense_ids.update(chunk['source1_entity_id'].astype(str).tolist())
    
    tfidf_only = []
    for s1_id, t_cands in tfidf_lookup.items():
        if s1_id not in dense_ids:
            t_cands.discard('')
            tfidf_only.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ",".join(list(t_cands))
            })
    
    if tfidf_only:
        print(f"  Adding {len(tfidf_only):,} TF-IDF-only queries not found in dense candidates...")
        res_df = pd.DataFrame(tfidf_only)
        res_df.to_csv(merged_path, sep="\t", index=False, mode='a', header=False)
    
    del tfidf_lookup, dense_ids
    gc.collect()


if __name__ == "__main__":
    data_dir = r"../dataset"
    out_dir = r"../output"
    
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    
    dense_path = os.path.join(out_dir, "full_train_candidate_pairs.tsv")
    tfidf_path = os.path.join(out_dir, "tfidf_train_candidate_pairs.tsv")
    merged_path = os.path.join(out_dir, "merged_train_candidate_pairs.tsv")
    
    log_memory("Script start")
    
    # Sanity check: make sure dense candidates exist
    if not os.path.exists(dense_path):
        print(f"ERROR: Dense candidate file not found at {dense_path}")
        print("Run blocking.py first!")
        exit(1)
    
    if not os.path.exists(tfidf_path):
        vectorizer, catalog_matrix, catalog_ids = build_tfidf_index(s2_path, s3_path)
        search_tfidf_candidates(s1_path, vectorizer, catalog_matrix, catalog_ids, tfidf_path, k=30)
        del vectorizer, catalog_ids
        gc.collect()
    else:
        print(f"TF-IDF candidates already exist at {tfidf_path}, skipping search...")
    
    merge_candidates(dense_path, tfidf_path, merged_path)
    log_memory("Script end")
    print(f"\nDONE! Merged candidate file: {merged_path}")
