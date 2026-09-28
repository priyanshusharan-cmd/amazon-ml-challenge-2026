import pandas as pd
import numpy as np
import os
import glob
import gc
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from preprocess import normalize_text, normalize_address

def preprocess_chunk(df, prefix="passage: "):
    """
    Preprocesses a dataframe chunk by applying normalization to names and addresses.
    Prepend 'passage: ' or 'query: ' for multilingual-e5 models for optimal retrieval quality.
    """
    names = df['business_name'].tolist()
    addresses = df['business_address'].tolist()
    
    norm_names = [normalize_text(n) for n in names]
    norm_addrs = [normalize_address(a)[0] for a in addresses]
    
    text_features = [f"{prefix}{n} {a}".strip() for n, a in zip(norm_names, norm_addrs)]
    return text_features


def build_catalog_embeddings(catalog_sources, cache_dir, model, chunk_size=50000):
    """
    Encodes catalog files in chunks and immediately caches each chunk to disk.
    If a chunk was already computed, it skips it automatically.
    RAM usage stays minimal because embeddings are discarded from RAM right after saving.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    for prefix, source_file in catalog_sources:
        print(f"\nProcessing {source_file} (prefix='{prefix}')...")
        chunk_iter = pd.read_csv(source_file, sep="\t", chunksize=chunk_size)
        
        for chunk_idx, chunk in enumerate(tqdm(chunk_iter, desc=f"Catalog {prefix}")):
            emb_file = os.path.join(cache_dir, f"{prefix}_chunk_{chunk_idx:04d}_emb.npy")
            ids_file = os.path.join(cache_dir, f"{prefix}_chunk_{chunk_idx:04d}_ids.npy")
            
            # Checkpoint: skip if already saved to disk
            if os.path.exists(emb_file) and os.path.exists(ids_file):
                continue
                
            ids = chunk['entity_id'].tolist()
            text_features = preprocess_chunk(chunk, prefix="passage: ")
            
            # Encode on GPU in float16
            embeddings = model.encode(
                text_features, 
                batch_size=1024, 
                show_progress_bar=False, 
                normalize_embeddings=True
            )
            
            # Save chunk directly to disk without memory duplication
            emb_f16 = np.ascontiguousarray(embeddings, dtype=np.float16)
            np.save(emb_file, emb_f16)
            np.save(ids_file, np.array(ids))
            
            # Immediately free memory
            del chunk, text_features, embeddings, emb_f16, ids
            gc.collect()
            torch.cuda.empty_cache()


def search_top_candidates(val_s1_path, cache_dir, output_path, model, k=30, min_sim=0.5):
    """
    Encodes query companies (S1) and streams through cached catalog chunks on disk.
    Maintains a running Top-K leaderboard on the GPU with near-zero RAM usage.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading query IDs from: {val_s1_path}")
    
    # Read just the IDs to save memory
    s1_ids = pd.read_csv(val_s1_path, sep="\t", usecols=['entity_id'])['entity_id'].tolist()
    num_queries = len(s1_ids)
    print(f"Total queries to match: {num_queries}")
    
    s1_emb_cache = os.path.join(cache_dir, "s1_queries.npy")
    if os.path.exists(s1_emb_cache):
        print(f"Loading cached query embeddings from {s1_emb_cache}...")
        s1_embeddings = np.load(s1_emb_cache)
    else:
        print("Encoding query companies on GPU in chunks...")
        chunk_size = 50000
        s1_embs_list = []
        
        # Stream read to prevent memory fragmentation
        chunk_iter = pd.read_csv(val_s1_path, sep="\t", chunksize=chunk_size)
        total_chunks = (num_queries + chunk_size - 1) // chunk_size
        
        for chunk_df in tqdm(chunk_iter, desc="Encoding S1 queries", total=total_chunks):
            s1_text_features = preprocess_chunk(chunk_df, prefix="query: ")
            chunk_embeddings = model.encode(
                s1_text_features, 
                batch_size=256, 
                show_progress_bar=False, 
                normalize_embeddings=True
            )
            s1_embs_list.append(chunk_embeddings.astype(np.float16))
            
            del chunk_df, s1_text_features, chunk_embeddings
            gc.collect()
            
        s1_embeddings = np.vstack(s1_embs_list)
        np.save(s1_emb_cache, s1_embeddings)
        del s1_embs_list
        gc.collect()
        
    # Keep massive 1.7M query tensor on CPU to save 1.3 GB VRAM
    # Use from_numpy to share memory and avoid duplicating 1.7GB in RAM
    s1_tensor_cpu = torch.from_numpy(s1_embeddings)
    
    # 1. Build catalog entity ID index mapping
    print("Loading catalog entity ID mapping offsets...")
    emb_files = sorted(glob.glob(os.path.join(cache_dir, "s[23]_chunk_*_emb.npy")))
    chunk_offsets = []
    current_offset = 0
    for emb_f in emb_files:
        ids_f = emb_f.replace("_emb.npy", "_ids.npy")
        # Just load shape without building massive lists
        ids_arr = np.load(ids_f, allow_pickle=True)
        chunk_len = len(ids_arr)
        chunk_offsets.append((current_offset, chunk_len))
        current_offset += chunk_len
        del ids_arr
    print(f"Total catalog entities indexed: {current_offset}")
    
    scores_cache = os.path.join(cache_dir, "best_scores.npy")
    indices_cache = os.path.join(cache_dir, "best_indices.npy")
    
    if os.path.exists(scores_cache) and os.path.exists(indices_cache):
        print(f"\nFound cached top-k search results! Loading from disk and skipping 50-minute GPU search...")
        # Load directly onto CPU to bypass GPU
        best_scores = torch.from_numpy(np.load(scores_cache))
        best_indices = torch.from_numpy(np.load(indices_cache))
    else:
        # Initialize running Top-K trackers on GPU to remove all sync overhead (takes ~600MB)
        best_scores = torch.full((num_queries, k), -1.0, dtype=torch.float16, device=device)
        best_indices = torch.full((num_queries, k), -1, dtype=torch.int32, device=device)
        
        print(f"\nStreaming search across {len(emb_files)} catalog chunks on GPU...")
        for chunk_idx, emb_file in enumerate(tqdm(emb_files, desc="Searching catalog")):
            chunk_offset, chunk_len = chunk_offsets[chunk_idx]
            chunk_emb = np.load(emb_file)
            chunk_tensor = torch.tensor(chunk_emb, dtype=torch.float16, device=device)
            
            q_batch_size = 5000
            for q_start in range(0, num_queries, q_batch_size):
                q_end = min(q_start + q_batch_size, num_queries)
                
                q_tensor_gpu = s1_tensor_cpu[q_start:q_end].to(device, non_blocking=True)
                sim_scores = torch.matmul(q_tensor_gpu, chunk_tensor.T)
                
                chunk_top_scores, chunk_top_indices = torch.topk(
                    sim_scores, k=min(k, chunk_tensor.shape[0]), dim=1
                )
                
                # Vectorized GPU merge (all on GPU, no sync!)
                combined_scores = torch.cat([best_scores[q_start:q_end], chunk_top_scores], dim=1)
                global_chunk_top_idx = chunk_top_indices.to(torch.int32) + chunk_offset
                combined_indices = torch.cat([best_indices[q_start:q_end], global_chunk_top_idx], dim=1)
                
                new_scores, rank_idx = torch.topk(combined_scores, k=k, dim=1)
                
                best_scores[q_start:q_end] = new_scores
                best_indices[q_start:q_end] = torch.gather(combined_indices, 1, rank_idx)
                
            del chunk_emb, chunk_tensor
            
        print("\nSaving top-k search results to disk to prevent data loss on crash...")
        np.save(scores_cache, best_scores.cpu().numpy())
        np.save(indices_cache, best_indices.cpu().numpy())
        
    print("\nFreeing up RAM before formatting candidates...")
    if 's1_embeddings' in locals():
        del s1_embeddings
    if 's1_tensor_cpu' in locals():
        del s1_tensor_cpu
    gc.collect()

    print("Loading all catalog IDs into memory...")
    catalog_id_arrays = []
    for emb_f in emb_files:
        ids_f = emb_f.replace("_emb.npy", "_ids.npy")
        catalog_id_arrays.append(np.load(ids_f, allow_pickle=True))
    all_catalog_ids = np.concatenate(catalog_id_arrays)
    del catalog_id_arrays
    gc.collect()

    print("\nFormatting candidate results...")
    best_scores_np = best_scores.cpu().numpy()
    best_indices_np = best_indices.cpu().numpy()
    
    chunk_size = 50000
    header_written = False
    
    for start_row in tqdm(range(0, num_queries, chunk_size), desc="Formatting candidates"):
        end_row = min(start_row + chunk_size, num_queries)
        results = []
        for row in range(start_row, end_row):
            s1_id = s1_ids[row]
            valid_cands = []
            for idx, score in zip(best_indices_np[row], best_scores_np[row]):
                if idx >= 0 and score >= min_sim:
                    valid_cands.append(str(all_catalog_ids[idx]))
                    
            results.append({
                'source1_entity_id': str(s1_id),
                'candidate_entity_ids': ",".join(valid_cands)
            })
            
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, sep="\t", index=False, mode='w' if not header_written else 'a', header=not header_written)
        header_written = True
            
    print(f"Successfully generated candidate pairs! Saved to: {output_path}")


def create_blocking(val_s1_path, s2_path, s3_path, output_path, cache_dir, model_name="intfloat/multilingual-e5-small", k=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading Sentence Transformer model: {model_name}")
    # Using 'dtype' to utilize half-precision Tensor Cores cleanly
    model = SentenceTransformer(model_name, device=device, model_kwargs={"dtype": torch.float16})
    model.max_seq_length = 64
    
    catalog_sources = [
        ("s2", s2_path),
        ("s3", s3_path)
    ]
    
    # Step 1: Build & cache catalog embeddings to disk (with instant checkpoint-resume)
    build_catalog_embeddings(catalog_sources, cache_dir, model, chunk_size=50000)
    
    # Step 2: Stream search across cached embeddings to find top K candidates
    search_top_candidates(val_s1_path, cache_dir, output_path, model, k=k, min_sim=0.5)


if __name__ == "__main__":
    data_dir = r"../dataset"
    
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    
    output_dir = r"../output"
    cache_dir = os.path.join(output_dir, "embeddings_cache")
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, "full_train_candidate_pairs.tsv")
    
    create_blocking(
        val_s1_path=s1_path, 
        s2_path=s2_path, 
        s3_path=s3_path, 
        output_path=out_path, 
        cache_dir=cache_dir,
        model_name="intfloat/multilingual-e5-small", 
        k=100
    )
