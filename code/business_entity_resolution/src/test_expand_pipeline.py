import os
import sys
import time
import sqlite3
import pandas as pd
import unidecode
import re
from collections import defaultdict

def test_pipeline():
    print("Testing candidate expansion pipeline on validation sample...")
    
    val_sample_path = "output/val_candidate_pairs_sample.tsv"
    val_gt_path = "output/val_gt_split.tsv"
    db_path = "output/train_catalog_temp.db"
    
    gt_df = pd.read_csv(val_gt_path, sep="\t")
    val_df = pd.read_csv(val_sample_path, sep="\t")
    merged = pd.merge(val_df, gt_df, on="source1_entity_id", how="inner")
    
    gt_map = {}
    existing_cands = {}
    for _, row in merged.iterrows():
        sid = row["source1_entity_id"]
        gt_map[sid] = set(str(row["matched_entity_ids"]).split(",")) if pd.notna(row["matched_entity_ids"]) else set()
        existing_cands[sid] = [c.strip() for c in str(row["candidate_entity_ids"]).split(",") if c.strip()]
        
    total_true = sum(len(v) for v in gt_map.values())
    initial_recall = sum(len(gt_map[s].intersection(set(existing_cands[s]))) for s in gt_map)
    print(f"Initial Candidate Recall: {initial_recall}/{total_true} ({initial_recall/total_true*100:.2f}%)")
    
    legal_pat = re.compile(r'\b(pvt|pv\.t|private|ltd|lt\.d|limited|corp|corporation|inc|incorporated|llc|sarl|sas|sa|eurl|enterprises|services|solutions|company|co|group)\b', re.I)
    domain_pat = re.compile(r'\.(com|org|net|in|fr|co|biz|info)\b', re.I)
    num_re = re.compile(r'^\s*(?:#|no\.?|plot|door)?\s*([0-9]+[a-z]?)', re.I)
    pc_re = re.compile(r'\b\d{5,6}\b')

    def get_brand(name):
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

    # Fetch S1 queries
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    
    s1_ids = list(gt_map.keys())
    placeholders = ','.join(['?']*len(s1_ids))
    cur.execute(f"SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id IN ({placeholders})", s1_ids)
    s1_rows = cur.fetchall()
    
    s1_brand_map = defaultdict(list)
    s1_street_map = defaultdict(list)
    s1_pc_map = defaultdict(list)
    
    for eid, name, addr, country in s1_rows:
        cntry = country or ''
        b = get_brand(name)
        if b and len(b) >= 4:
            s1_brand_map[(cntry, b)].append(eid)
        sk, pk = get_addr_keys(addr)
        if sk:
            s1_street_map[(cntry, sk[0], sk[1])].append(eid)
        if pk:
            s1_pc_map[(cntry, pk[0], pk[1])].append(eid)
            
    print(f"Built S1 indices: {len(s1_brand_map):,} brands, {len(s1_street_map):,} streets, {len(s1_pc_map):,} postal keys.")
    
    # Stream catalog
    print("Streaming through catalog table...")
    t0 = time.time()
    cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog")
    
    new_candidates = defaultdict(set)
    batch_size = 500000
    rows_scanned = 0
    
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        rows_scanned += len(rows)
        for eid, name, addr, country in rows:
            cntry = country or ''
            b = get_brand(name)
            if b and len(b) >= 4:
                k = (cntry, b)
                if k in s1_brand_map:
                    for sid in s1_brand_map[k]:
                        if len(new_candidates[sid]) < 10:
                            new_candidates[sid].add(eid)
            sk, pk = get_addr_keys(addr)
            if sk:
                k = (cntry, sk[0], sk[1])
                if k in s1_street_map:
                    for sid in s1_street_map[k]:
                        if len(new_candidates[sid]) < 10:
                            new_candidates[sid].add(eid)
            if pk:
                k = (cntry, pk[0], pk[1])
                if k in s1_pc_map:
                    for sid in s1_pc_map[k]:
                        if len(new_candidates[sid]) < 10:
                            new_candidates[sid].add(eid)
                            
    t1 = time.time()
    conn.close()
    print(f"Catalog scan completed in {t1-t0:.2f}s ({rows_scanned:,} rows)!")
    
    # Evaluate expanded candidates
    expanded_cands = {}
    total_added = 0
    for sid in s1_ids:
        cur_list = list(existing_cands[sid])
        cur_set = set(cur_list)
        for cand in new_candidates.get(sid, []):
            if cand not in cur_set:
                cur_list.append(cand)
                cur_set.add(cand)
                total_added += 1
        expanded_cands[sid] = cur_list
        
    final_recall = sum(len(gt_map[s].intersection(set(expanded_cands[s]))) for s in gt_map)
    print(f"\nFinal Expanded Candidate Recall: {final_recall}/{total_true} ({final_recall/total_true*100:.2f}%)  [+{(final_recall-initial_recall)/total_true*100:.2f}% gain!]")
    print(f"Total new candidates added across {len(s1_ids):,} queries: {total_added:,} (avg {total_added/len(s1_ids):.2f} extra cands/query)")

if __name__ == "__main__":
    test_pipeline()
