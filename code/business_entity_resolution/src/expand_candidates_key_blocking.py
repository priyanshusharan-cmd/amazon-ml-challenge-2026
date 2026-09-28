import os
import sys
import time
import psutil
import sqlite3
import unidecode
import re
from collections import defaultdict
from tqdm import tqdm

def log_memory(label=""):
    mem = psutil.virtual_memory()
    print(f"[{label}] RAM Used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}%)", flush=True)

legal_pat = re.compile(r'\b(pvt|pv\.t|private|ltd|lt\.d|limited|corp|corporation|inc|incorporated|llc|sarl|sas|sa|eurl|enterprises|services|solutions|company|co|group)\b', re.I)
domain_pat = re.compile(r'\.(com|org|net|in|fr|co|biz|info)\b', re.I)
num_re = re.compile(r'^\s*(?:#|no\.?|plot|door)?\s*([0-9]+[a-z]?)', re.I)
pc_re = re.compile(r'\b\d{5,6}\b')

def get_clean_brand(name):
    if not name or name == 'None':
        return ''
    if not name.isascii():
        name = unidecode.unidecode(name)
    name = name.lower()
    name = domain_pat.sub('', name)
    name = name.replace('0', 'o').replace('5', 's').replace('1', 'i').replace('3', 'e')
    name = legal_pat.sub(' ', name)
    clean = re.sub(r'[^a-z0-9]', '', name)
    return clean

def get_addr_keys(addr):
    if not addr or addr == 'None':
        return None, None
    pc_m = pc_re.search(addr)
    pc = pc_m.group(0) if pc_m else None
    
    idx = addr.find(',')
    first_part = addr[:idx] if idx != -1 else addr
    num_m = num_re.search(first_part)
    num = num_m.group(1).lower() if num_m else None
    if num_m:
        rem = first_part[num_m.end():].strip(' -#/,')
    else:
        rem = first_part.strip()
    street = re.sub(r'[^a-z0-9 ]', '', rem.lower()).strip()
    
    street_key = (num, street) if (num and len(street) >= 3) else None
    pc_key = (pc, num) if (pc and num) else None
    return street_key, pc_key


def expand_candidates(input_cands_path, output_cands_path, db_path, max_extra_per_query=10, max_keys_freq=40):
    """
    Expands candidate pairs using lightweight in-memory key blocking (clean brand, street address, postal code).
    Matches strictly partitioned by ISO country.
    """
    start_time = time.time()
    log_memory("Expansion Start")
    print(f"\n{'='*70}")
    print(f"Candidate Expansion: {input_cands_path} -> {output_cands_path}")
    print(f"Catalog DB: {db_path} | Max Extra Cands/Query: {max_extra_per_query}")
    print(f"{'='*70}\n")
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    
    # 1. Load S1 query records from SQLite
    print("Step 1/4: Loading S1 queries and generating blocking keys...")
    t0 = time.time()
    cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog")
    
    s1_brand_map = defaultdict(list)
    s1_street_map = defaultdict(list)
    s1_pc_map = defaultdict(list)
    total_s1 = 0
    
    batch_size = 250000
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for eid, name, addr, country in rows:
            total_s1 += 1
            cntry = country or ''
            b = get_clean_brand(name)
            if b and len(b) >= 4:
                s1_brand_map[(cntry, b)].append(eid)
            sk, pk = get_addr_keys(addr)
            if sk:
                s1_street_map[(cntry, sk[0], sk[1])].append(eid)
            if pk:
                s1_pc_map[(cntry, pk[0], pk[1])].append(eid)
                
    print(f"  Indexed {total_s1:,} S1 queries in {time.time()-t0:.2f}s.")
    print(f"  Keys: {len(s1_brand_map):,} brands | {len(s1_street_map):,} streets | {len(s1_pc_map):,} postal keys.")
    log_memory("After S1 Keys")
    
    # 2. Stream through catalog table and match
    print("\nStep 2/4: Streaming through catalog table to retrieve key-matching candidates...")
    t0 = time.time()
    cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog")
    
    new_cands_by_s1 = defaultdict(list)
    cat_scanned = 0
    matched_hits = 0
    
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        cat_scanned += len(rows)
        for eid, name, addr, country in rows:
            cntry = country or ''
            # Brand match
            b = get_clean_brand(name)
            if b and len(b) >= 4:
                k = (cntry, b)
                if k in s1_brand_map:
                    targets = s1_brand_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(new_cands_by_s1[sid]) < max_extra_per_query:
                                new_cands_by_s1[sid].append(eid)
                                matched_hits += 1
            # Address match
            sk, pk = get_addr_keys(addr)
            if sk:
                k = (cntry, sk[0], sk[1])
                if k in s1_street_map:
                    targets = s1_street_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(new_cands_by_s1[sid]) < max_extra_per_query:
                                new_cands_by_s1[sid].append(eid)
                                matched_hits += 1
            if pk:
                k = (cntry, pk[0], pk[1])
                if k in s1_pc_map:
                    targets = s1_pc_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(new_cands_by_s1[sid]) < max_extra_per_query:
                                new_cands_by_s1[sid].append(eid)
                                matched_hits += 1
                                
        if cat_scanned % 2000000 == 0:
            print(f"  Scanned {cat_scanned:,} catalog rows in {time.time()-t0:.1f}s | Hits: {matched_hits:,}")
            
    print(f"  Catalog scan finished in {time.time()-t0:.2f}s ({cat_scanned:,} rows)!")
    print(f"  Queries with extra candidates: {len(new_cands_by_s1):,} / {total_s1:,} ({len(new_cands_by_s1)/total_s1*100:.1f}%)")
    conn.close()
    
    # Clean up key maps to free memory
    del s1_brand_map, s1_street_map, s1_pc_map
    import gc
    gc.collect()
    log_memory("After Catalog Match")
    
    # 3. Stream through input candidate pairs and merge
    print(f"\nStep 3/4: Merging existing candidates with key-blocking candidates into {output_cands_path}...")
    t0 = time.time()
    
    total_written = 0
    total_added_cands = 0
    
    with open(input_cands_path, 'r', encoding='utf-8') as f_in, \
         open(output_cands_path, 'w', encoding='utf-8', buffering=4*1024*1024) as f_out:
         
        header = f_in.readline()
        f_out.write(header)
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            idx = line.find('\t')
            if idx == -1:
                continue
            s1_id = line[:idx]
            cands_str = line[idx+1:]
            
            existing = [c.strip() for c in cands_str.split(',') if c.strip()]
            existing_set = set(existing)
            
            extra = new_cands_by_s1.get(s1_id, [])
            for cand in extra:
                if cand not in existing_set:
                    existing.append(cand)
                    existing_set.add(cand)
                    total_added_cands += 1
                    
            f_out.write(f"{s1_id}\t{','.join(existing)}\n")
            total_written += 1
            if total_written % 500000 == 0:
                print(f"  Merged {total_written:,} / {total_s1:,} queries ({total_written/total_s1*100:.1f}%)...")
                
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Candidate Expansion Complete in {elapsed:.1f}s ({elapsed/60:.1f}m)!")
    print(f"  Total Queries Processed: {total_written:,}")
    print(f"  Total Extra Candidates Added: {total_added_cands:,} (avg {total_added_cands/max(1, total_written):.2f} / query)")
    print(f"  Output File: {output_cands_path} ({os.path.getsize(output_cands_path)/(1024**2):.1f} MB)")
    print(f"{'='*70}\n")
    log_memory("End")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expand candidate pairs using In-Memory Key Blocking.")
    parser.add_argument("--mode", choices=["test", "train", "sample"], default="test", help="Target mode: test or train")
    args = parser.parse_args()
    
    base_dir = r"../"
    
    if args.mode == "test":
        in_path = os.path.join(base_dir, "output", "candidate_pairs.tsv")
        out_path = os.path.join(base_dir, "output", "candidate_pairs_v4.tsv")
        db_path = os.path.join(base_dir, "output", "test_catalog_temp.db")
        expand_candidates(in_path, out_path, db_path, max_extra_per_query=10)
    elif args.mode == "train":
        in_path = os.path.join(base_dir, "output", "full_train_candidate_pairs.tsv")
        out_path = os.path.join(base_dir, "output", "full_train_candidate_pairs_v4.tsv")
        db_path = os.path.join(base_dir, "output", "train_catalog_temp.db")
        expand_candidates(in_path, out_path, db_path, max_extra_per_query=10)
    elif args.mode == "sample":
        in_path = os.path.join(base_dir, "output", "val_candidate_pairs_sample.tsv")
        out_path = os.path.join(base_dir, "output", "val_candidate_pairs_sample_v4.tsv")
        db_path = os.path.join(base_dir, "output", "train_catalog_temp.db")
        expand_candidates(in_path, out_path, db_path, max_extra_per_query=10)
