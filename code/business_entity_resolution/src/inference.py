import os
import re
import gc
import json
import time
import psutil
import sqlite3
import numpy as np
import pandas as pd
# Lazy import in worker to avoid Windows multiprocessing spawn race
# import lightgbm as lgb
# import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from rapidfuzz import fuzz

from preprocess import normalize_text, get_compact_signature, get_acronym, decompose_address

FEATURE_COLS = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_compact_match', 'name_acronym_match', 'is_addr_missing',
    'street_num_match', 'street_name_sim', 'city_state_sim',
    'addr_token_sort', 'digits_match', 'country_match',
    'is_dba_pattern', 'source_origin'
]

digits_re = re.compile(r'\d+')
def extract_digits(text):
    if not text:
        return ""
    return "".join(digits_re.findall(str(text)))

def log_memory(label=""):
    mem = psutil.virtual_memory()
    print(f"  [MEM {label}] Used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}%)", flush=True)

def get_cpu_usage():
    per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    usage_str = " ".join([f"{int(c)}%" for c in per_cpu])
    return f"CPU: [{usage_str}]"


def build_test_sqlite_catalog(s1_path, s2_path, s3_path, db_path):
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT count(*) FROM catalog")
            cat_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM s1_catalog")
            s1_count = cur.fetchone()[0]
            if cat_count == 9969589 and s1_count == 1732544:
                print(f"  Found verified SQLite test catalog ({cat_count:,} catalog records, {s1_count:,} S1 records)!")
                conn.close()
                return
        except Exception:
            pass
        conn.close()
        
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    cur = conn.cursor()
    
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='catalog'")
    if cur.fetchone()[0] == 0:
        print("\nBuilding SQLite catalog for test S2 and S3 (~2 minutes)...")
        for source_file in [s2_path, s3_path]:
            for chunk in tqdm(pd.read_csv(source_file, sep="\t", chunksize=250000, usecols=['entity_id', 'business_name', 'business_address', 'country']), desc=f"Importing {os.path.basename(source_file)}"):
                chunk.to_sql("catalog", conn, if_exists="append", index=False)
                conn.commit()
        print("Indexing catalog table on entity_id...")
        conn.execute("CREATE INDEX idx_entity_id ON catalog(entity_id)")
        conn.commit()
    else:
        print("Found existing 'catalog' table in SQLite DB.")
        
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='s1_catalog'")
    if cur.fetchone()[0] == 0:
        print("Building SQLite table for test S1 (~30 seconds)...")
        for chunk in tqdm(pd.read_csv(s1_path, sep="\t", chunksize=250000, usecols=['entity_id', 'business_name', 'business_address', 'country']), desc="Importing S1"):
            chunk.to_sql("s1_catalog", conn, if_exists="append", index=False)
            conn.commit()
        print("Indexing s1_catalog table on entity_id...")
        conn.execute("CREATE INDEX idx_s1_entity_id ON s1_catalog(entity_id)")
        conn.commit()
    else:
        print("Found existing 's1_catalog' table in SQLite DB.")
        
    conn.commit()
    conn.close()
    print("  SQLite test database ready!")


# Global worker model caches
_WORKER_LGB = None
_WORKER_XGB = None
_WORKER_LGB_PATH = None
_WORKER_XGB_PATH = None

def get_worker_lgb(lgb_path):
    global _WORKER_LGB, _WORKER_LGB_PATH
    if _WORKER_LGB is None or _WORKER_LGB_PATH != lgb_path:
        import lightgbm as lgb
        _WORKER_LGB = lgb.Booster(model_file=lgb_path)
        _WORKER_LGB_PATH = lgb_path
    return _WORKER_LGB

def get_worker_xgb(xgb_path):
    global _WORKER_XGB, _WORKER_XGB_PATH
    if _WORKER_XGB is None or _WORKER_XGB_PATH != xgb_path:
        import xgboost as xgb
        _WORKER_XGB = xgb.Booster()
        _WORKER_XGB.load_model(xgb_path)
        _WORKER_XGB_PATH = xgb_path
    return _WORKER_XGB


def fast_15_features(s1_rec, c_rec, cand_id):
    s1_name, s1_comp, s1_acro, s1_addr, s1_num, s1_street, s1_cs, s1_d, s1_country = s1_rec
    c_name, c_comp, c_acro, c_addr, c_num, c_street, c_cs, c_d, c_country = c_rec
    
    if s1_name and c_name:
        name_ratio = fuzz.ratio(s1_name, c_name) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
        name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
        name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    else:
        name_ratio = name_token_sort = name_token_set = name_partial = 0.0
        
    if s1_comp and c_comp and (s1_comp == c_comp or (len(s1_comp)>=6 and s1_comp in c_comp) or (len(c_comp)>=6 and c_comp in s1_comp)):
        name_compact_match = 1.0
    else:
        name_compact_match = 0.0
        
    if (s1_acro and s1_acro == c_comp) or (c_acro and c_acro == s1_comp):
        name_acronym_match = 1.0
    else:
        name_acronym_match = 0.0
        
    is_addr_missing = 1.0 if (not s1_addr or not c_addr) else 0.0
    
    if s1_num and c_num:
        street_num_match = 1.0 if s1_num == c_num else 0.0
    elif not s1_num and not c_num:
        street_num_match = 0.5
    else:
        street_num_match = 0.5
        
    if s1_street and c_street:
        street_name_sim = fuzz.ratio(s1_street, c_street) / 100.0
    elif not s1_street and not c_street:
        street_name_sim = 0.5
    else:
        street_name_sim = 0.0
        
    if s1_cs and c_cs:
        city_state_sim = fuzz.token_set_ratio(s1_cs, c_cs) / 100.0
    elif not s1_cs and not c_cs:
        city_state_sim = 0.5
    else:
        city_state_sim = 0.0
        
    if s1_addr and c_addr:
        addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    else:
        addr_token_sort = 0.0
        
    if s1_d and c_d:
        digits_match = 1.0 if s1_d == c_d else 0.0
    elif not s1_d and not c_d:
        digits_match = 1.0
    else:
        digits_match = 0.0
        
    country_match = 1.0 if (s1_country and c_country and s1_country == c_country) else 0.0
    is_dba_pattern = 1.0 if (street_name_sim >= 0.85 and street_num_match == 1.0 and country_match == 1.0) else 0.0
    source_origin = 1.0 if cand_id.startswith('S2-') else 0.0
    
    return (name_ratio, name_token_sort, name_token_set, name_partial, name_compact_match, name_acronym_match,
            is_addr_missing, street_num_match, street_name_sim, city_state_sim, addr_token_sort, digits_match,
            country_match, is_dba_pattern, source_origin)


_WORKER_CONN = None
_WORKER_LGB = None
_WORKER_XGB = None

def get_worker_conn(db_path):
    global _WORKER_CONN
    if _WORKER_CONN is None:
        abs_path = os.path.abspath(db_path).replace('\\', '/')
        uri = f"file:{abs_path}?mode=ro&immutable=1"
        _WORKER_CONN = sqlite3.connect(uri, uri=True, timeout=60.0)
        _WORKER_CONN.execute("PRAGMA query_only = ON;")
    return _WORKER_CONN

def get_worker_lgb(lgb_path):
    global _WORKER_LGB
    if _WORKER_LGB is None:
        import lightgbm as lgb
        _WORKER_LGB = lgb.Booster(model_file=lgb_path)
    return _WORKER_LGB

def get_worker_xgb(xgb_path):
    global _WORKER_XGB
    if _WORKER_XGB is None and xgb_path and os.path.exists(xgb_path):
        import xgboost as xgb
        _WORKER_XGB = xgb.Booster()
        _WORKER_XGB.load_model(xgb_path)
    return _WORKER_XGB

def fetch_by_ids(cur, table, id_list, cols, batch_size=900):
    results = []
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        cur.execute(f"SELECT {cols} FROM {table} WHERE entity_id IN ({placeholders})", batch)
        results.extend(cur.fetchall())
    return results


def process_inference_chunk(args):
    """
    Worker process:
    1. Fetches candidate and query metadata from SQLite and pre-normalizes once.
    2. Computes the 15 features.
    3. Runs inference via LightGBM, XGBoost, or Ensemble Blend.
    4. Evaluates matches with calibrated threshold and singleton guard.
    5. Formats output TSV lines in strictly preserved query order.
    """
    chunk_idx, chunk_records, db_path, model_mode, lgb_path, xgb_path, threshold, singleton_cutoff = args
    
    if not chunk_records:
        return chunk_idx, "", 0, 0, 0
        
    try:
        s1_ids = [r[0] for r in chunk_records]
        needed_cand_ids = set()
        pairs_list = []
        
        for s1_id, cands_str in chunk_records:
            if not cands_str or cands_str == 'nan':
                continue
            cands = [c.strip() for c in cands_str.split(',') if c.strip()]
            for cid in cands:
                needed_cand_ids.add(cid)
                pairs_list.append((s1_id, cid))
                
        conn = get_worker_conn(db_path)
        cur = conn.cursor()
        
        # 1. Fetch S1 query metadata & pre-normalize
        s1_rows = fetch_by_ids(cur, "s1_catalog", s1_ids, "entity_id, business_name, business_address, country")
        
        s1_dict = {}
        for eid, name, addr, country in s1_rows:
            n_name = normalize_text(name)
            c_comp = get_compact_signature(name)
            c_acro = get_acronym(name)
            c_addr, s_num, s_name, c_cs, c_pc = decompose_address(addr)
            d_str = extract_digits(f"{n_name} {addr}")
            s1_dict[eid] = (n_name, c_comp, c_acro, c_addr, s_num, s_name, c_cs, d_str, country or "")
        del s1_rows
        
        # 2. Fetch candidate metadata & pre-normalize
        catalog_dict = {}
        if needed_cand_ids:
            cat_rows = fetch_by_ids(cur, "catalog", list(needed_cand_ids), "entity_id, business_name, business_address, country")
            
            for eid, name, addr, country in cat_rows:
                n_name = normalize_text(name)
                c_comp = get_compact_signature(name)
                c_acro = get_acronym(name)
                c_addr, s_num, s_name, c_cs, c_pc = decompose_address(addr)
                d_str = extract_digits(f"{n_name} {addr}")
                catalog_dict[eid] = (n_name, c_comp, c_acro, c_addr, s_num, s_name, c_cs, d_str, country or "")
            del cat_rows
        
        # 3. Compute 15 features
        num_pairs = len(pairs_list)
        empty_rec = ("", "", "", "", "", "", "", "", "")
        
        if num_pairs > 0:
            X_feats = np.empty((num_pairs, 15), dtype=np.float32)
            
            for i, (s1_id, cand_id) in enumerate(pairs_list):
                s1_rec = s1_dict.get(s1_id, empty_rec)
                c_rec = catalog_dict.get(cand_id, empty_rec)
                f = fast_15_features(s1_rec, c_rec, cand_id)
                for j in range(15):
                    X_feats[i, j] = f[j]
                    
            # 4. Predict probabilities with chosen model architecture
            if model_mode == "ensemble":
                bst_lgb = get_worker_lgb(lgb_path)
                probs_lgb = bst_lgb.predict(X_feats)
                
                bst_xgb = get_worker_xgb(xgb_path)
                dmat = xgb.DMatrix(X_feats, feature_names=FEATURE_COLS)
                probs_xgb = bst_xgb.predict(dmat)
                del dmat
                
                probs = 0.5 * probs_lgb + 0.5 * probs_xgb
            elif model_mode == "xgboost":
                bst_xgb = get_worker_xgb(xgb_path)
                dmat = xgb.DMatrix(X_feats, feature_names=FEATURE_COLS)
                probs = bst_xgb.predict(dmat)
                del dmat
            else: # lgbm
                bst_lgb = get_worker_lgb(lgb_path)
                probs = bst_lgb.predict(X_feats)
                
            # 5. Group candidate probabilities by s1_id
            preds_by_s1 = {}
            for (s1_id, cand_id), p in zip(pairs_list, probs):
                if s1_id not in preds_by_s1:
                    preds_by_s1[s1_id] = []
                preds_by_s1[s1_id].append((cand_id, p))
                
            # 6. Apply threshold and singleton guard
            matches_by_s1 = {}
            for s1_id, cand_probs in preds_by_s1.items():
                max_p = max(p for _, p in cand_probs)
                if max_p < singleton_cutoff:
                    matches_by_s1[s1_id] = []
                    continue
                    
                matched_cands = [cid for cid, p in cand_probs if p >= threshold]
                matches_by_s1[s1_id] = matched_cands
        else:
            matches_by_s1 = {}
            
        # 7. Build output TSV string strictly maintaining original chunk query order
        lines = []
        chunk_matches = 0
        chunk_singletons = 0
        
        for s1_id, _ in chunk_records:
            matched_cands = matches_by_s1.get(s1_id, [])
            if matched_cands:
                match_str = ",".join(matched_cands)
                lines.append(f"{s1_id}\t{match_str}\n")
                chunk_matches += len(matched_cands)
            else:
                lines.append(f"{s1_id}\t\n")
                chunk_singletons += 1
                
        return chunk_idx, "".join(lines), chunk_matches, chunk_singletons, len(chunk_records)
        
    except Exception as e:
        print(f"\n[Warning] Exception in chunk {chunk_idx}: {e}", flush=True)
        fallback_lines = [f"{s1_id}\t\n" for s1_id, _ in chunk_records]
        return chunk_idx, "".join(fallback_lines), 0, len(chunk_records), len(chunk_records)


def tsv_chunk_generator(tsv_path, chunk_size):
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        batch = []
        chunk_idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.find('\t')
            if idx != -1:
                batch.append((line[:idx], line[idx+1:]))
            if len(batch) >= chunk_size:
                yield chunk_idx, batch
                chunk_idx += 1
                batch = []
        if batch:
            yield chunk_idx, batch


def run_test_inference(cands_path, s1_path, s2_path, s3_path, out_dir, out_matching_path):
    log_memory("Inference Start")
    
    config_path = os.path.join(out_dir, "threshold_config_v3.json")
    lgb_path = os.path.join(out_dir, "lgbm_model_v3.txt")
    xgb_path = os.path.join(out_dir, "xgb_model_v3.json")
    
    # 1. Load optimal threshold and model mode
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            model_mode = cfg.get('model_mode', 'ensemble')
            threshold = float(cfg.get('optimal_threshold', 0.80))
            singleton_cutoff = float(cfg.get('singleton_cutoff', max(0.75, threshold - 0.05)))
            print(f"Loaded tuned configuration:")
            print(f"  - Model Architecture: {model_mode.upper()}")
            print(f"  - Decision Threshold: {threshold}")
            print(f"  - Singleton Safeguard Cutoff: {singleton_cutoff}")
            print(f"  - Validation Macro F0.5: {cfg.get('validation_macro_f05', 0):.5f}")
    else:
        model_mode = "lgbm"
        threshold = 0.80
        singleton_cutoff = 0.76
        print(f"Config not found at {config_path}. Using fallback mode: {model_mode}, threshold: {threshold}")
        
    db_path = os.path.join(os.path.dirname(out_matching_path), "test_catalog_temp.db")
    build_test_sqlite_catalog(s1_path, s2_path, s3_path, db_path)
    
    if os.path.exists(out_matching_path):
        os.remove(out_matching_path)
        
    num_cores = min(6, os.cpu_count())
    chunk_size = 1500
    MAX_IN_FLIGHT = 12
    
    print(f"\nStarting Multiprocessing Test Inference with {num_cores} workers...")
    print(f"  Mode: {model_mode.upper()} | Chunk size: {chunk_size:,} queries | Threshold: {threshold} | Cutoff: {singleton_cutoff}")
    print(f"  Target output: {out_matching_path}")
    log_memory("Before ProcessPool launch")
    
    start_time = time.time()
    total_queries = 1732544
    completed_queries = 0
    total_matches = 0
    total_singletons = 0
    next_chunk_to_write = 0
    pending_buffer = {}
    
    with open(out_matching_path, 'w', encoding='utf-8', buffering=2*1024*1024) as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        
        with ProcessPoolExecutor(max_workers=num_cores, max_tasks_per_child=20) as executor:
            futures = set()
            
            def harvest_and_write(futures_set):
                nonlocal next_chunk_to_write, completed_queries, total_matches, total_singletons
                done, remaining = wait(futures_set, return_when=FIRST_COMPLETED)
                for fut in done:
                    c_idx, tsv_text, c_matches, c_singletons, c_queries = fut.result()
                    pending_buffer[c_idx] = tsv_text
                    total_matches += c_matches
                    total_singletons += c_singletons
                    completed_queries += c_queries
                    
                while next_chunk_to_write in pending_buffer:
                    chunk_text = pending_buffer.pop(next_chunk_to_write)
                    if chunk_text:
                        f_out.write(chunk_text)
                    next_chunk_to_write += 1
                    
                return remaining

            for c_idx, batch in tsv_chunk_generator(cands_path, chunk_size):
                while len(futures) >= MAX_IN_FLIGHT:
                    futures = harvest_and_write(futures)
                    
                mem = psutil.virtual_memory()
                while mem.percent > 82.0:
                    print(f"  [System Memory Alert: {mem.percent}%] Pausing 2s to allow external apps to release RAM...", flush=True)
                    time.sleep(2.0)
                    mem = psutil.virtual_memory()
                    
                futures.add(executor.submit(
                    process_inference_chunk,
                    (c_idx, batch, db_path, model_mode, lgb_path, xgb_path, threshold, singleton_cutoff)
                ))
                
                if (c_idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    qps = completed_queries / max(elapsed, 0.001)
                    rem_sec = max(0, (total_queries - completed_queries) / max(qps, 1))
                    print(f"\n[Inference Progress] {completed_queries:,} / {total_queries:,} queries ({completed_queries/total_queries*100:.1f}%) in {elapsed:.1f}s | Speed: {qps:,.0f} queries/s | ETA: {rem_sec:.0f}s", flush=True)
                    print(f"  {get_cpu_usage()}", flush=True)
                    print(f"  Matches predicted: {total_matches:,} | Singletons: {total_singletons:,} ({total_singletons/max(completed_queries, 1)*100:.1f}%)", flush=True)
                    log_memory(f"Chunk {c_idx+1}")
                    
            while futures:
                futures = harvest_and_write(futures)
                
            while next_chunk_to_write in pending_buffer:
                chunk_text = pending_buffer.pop(next_chunk_to_write)
                if chunk_text:
                    f_out.write(chunk_text)
                next_chunk_to_write += 1

    total_elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"Test Inference Completed Successfully in {total_elapsed:.1f}s ({total_elapsed/60:.2f} minutes)!")
    print(f"  Total queries written: {completed_queries:,}")
    print(f"  Total matches predicted: {total_matches:,}")
    print(f"  Singletons (predicted empty): {total_singletons:,} ({total_singletons/max(completed_queries, 1)*100:.2f}%)")
    print(f"  Output saved to: {out_matching_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    
    base_dir = r"../"
    test_dir = os.path.join(base_dir, "6ab10eb3b23ba_student_resource", "student_resource", "dataset", "test")
    out_dir = os.path.join(base_dir, "output")
    
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    cands_path = os.path.join(out_dir, "candidate_pairs.tsv")
    out_matching_path = os.path.join(out_dir, "matching_results.tsv")
    
    run_test_inference(
        cands_path, s1_path, s2_path, s3_path, 
        out_dir, out_matching_path
    )
