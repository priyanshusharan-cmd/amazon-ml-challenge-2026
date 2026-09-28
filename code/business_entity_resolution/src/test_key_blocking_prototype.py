import sqlite3
import time
import sys
from collections import defaultdict
sys.path.append('code/business_entity_resolution/src')
from preprocess import normalize_text, get_compact_signature, decompose_address

print("Connecting to train catalog DB...")
conn = sqlite3.connect('file:output/train_catalog_temp.db?mode=ro&immutable=1', uri=True)
cur = conn.cursor()

# Test indexing on 200,000 catalog rows to measure memory and speed
print("Fetching 200,000 catalog rows...")
t0 = time.time()
cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog LIMIT 200000")
rows = cur.fetchall()
t1 = time.time()
print(f"Fetched in {t1-t0:.2f}s")

idx_comp = defaultdict(list)
idx_addr_num_street = defaultdict(list)
idx_addr_pc_num = defaultdict(list)

t0 = time.time()
for eid, name, addr, country in rows:
    c_country = country or ""
    # 1. Compact signature
    comp = get_compact_signature(name)
    if comp and len(comp) >= 4:
        idx_comp[(c_country, comp)].append(eid)
        
    # 2. Address keys
    if addr:
        clean_addr, s_num, s_name, cs, pc = decompose_address(addr)
        if s_num and s_name and len(s_name) >= 3:
            idx_addr_num_street[(c_country, s_num, s_name)].append(eid)
        if pc and s_num:
            idx_addr_pc_num[(c_country, pc, s_num)].append(eid)
t1 = time.time()

print(f"Indexed 200k rows in {t1-t0:.2f}s")
print(f"Distinct compact keys: {len(idx_comp):,}")
print(f"Distinct (num, street) keys: {len(idx_addr_num_street):,}")
print(f"Distinct (pc, num) keys: {len(idx_addr_pc_num):,}")

# Check key size distribution
large_comp_keys = sum(1 for v in idx_comp.values() if len(v) > 20)
print(f"Compact keys with > 20 matches: {large_comp_keys} / {len(idx_comp)}")

conn.close()
