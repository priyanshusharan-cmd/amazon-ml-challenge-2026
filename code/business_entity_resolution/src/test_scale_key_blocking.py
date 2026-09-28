import sqlite3
import time
import psutil
import unidecode
import re
from collections import defaultdict

def log_memory(label=""):
    mem = psutil.virtual_memory()
    print(f"[{label}] RAM Used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}%)", flush=True)

legal_pat = re.compile(r'\b(pvt|pv\.t|private|ltd|lt\.d|limited|corp|corporation|inc|incorporated|llc|sarl|sas|sa|eurl|enterprises|services|solutions|company|co|group)\b', re.I)
domain_pat = re.compile(r'\.(com|org|net|in|fr|co|biz|info)\b', re.I)

def clean_brand(name):
    if not name or name == 'None':
        return ''
    if not name.isascii():
        name = unidecode.unidecode(name)
    name = name.lower()
    name = domain_pat.sub('', name)
    name = name.replace('0', 'o').replace('5', 's').replace('1', 'i').replace('3', 'e')
    name = legal_pat.sub(' ', name)
    clean = re.sub(r'[^a-z]', '', name)
    return clean

num_re = re.compile(r'^\s*(?:#|no\.?|plot|door)?\s*([0-9]+[a-z]?)', re.I)
def fast_addr_key(addr):
    if not addr or addr == 'None':
        return None, None
    idx = addr.find(',')
    first_part = addr[:idx] if idx != -1 else addr
    num_m = num_re.search(first_part)
    num = num_m.group(1).lower() if num_m else None
    if num_m:
        rem = first_part[num_m.end():].strip(' -#/,')
    else:
        rem = first_part.strip()
    street = re.sub(r'[^a-z0-9 ]', '', rem.lower()).strip()
    return num, street if len(street) >= 3 else None

log_memory("Start")
print("Connecting to train catalog DB...")
conn = sqlite3.connect('file:output/train_catalog_temp.db?mode=ro&immutable=1', uri=True)
cur = conn.cursor()

# 1. Fetch 100,000 S1 queries
print("Fetching 100,000 S1 queries...")
t0 = time.time()
cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog LIMIT 100000")
s1_rows = cur.fetchall()
t1 = time.time()
print(f"Fetched 100k queries in {t1-t0:.2f}s")

# 2. Build S1 keys
t0 = time.time()
s1_brand_dict = defaultdict(list)
s1_addr_dict = defaultdict(list)

for eid, name, addr, country in s1_rows:
    c_country = country or ''
    b = clean_brand(name)
    if b and len(b) >= 4:
        s1_brand_dict[(c_country, b)].append(eid)
    num, street = fast_addr_key(addr)
    if num and street:
        s1_addr_dict[(c_country, num, street)].append(eid)
t1 = time.time()
print(f"Indexed 100k S1 queries in {t1-t0:.2f}s | Distinct brands: {len(s1_brand_dict):,} | Addr keys: {len(s1_addr_dict):,}")
log_memory("After S1 Indexing")

# 3. Stream through entire 10 million catalog
print("\nStreaming through all 10,320,219 catalog rows...")
t0 = time.time()
cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog")

hits = 0
batch_size = 500000
rows_read = 0

candidates_by_s1 = defaultdict(list)

while True:
    batch = cur.fetchmany(batch_size)
    if not batch:
        break
    rows_read += len(batch)
    for eid, name, addr, country in batch:
        c_country = country or ''
        b = clean_brand(name)
        if b and len(b) >= 4:
            k = (c_country, b)
            if k in s1_brand_dict:
                for s1_id in s1_brand_dict[k]:
                    if len(candidates_by_s1[s1_id]) < 15:
                        candidates_by_s1[s1_id].append(eid)
                hits += 1
        num, street = fast_addr_key(addr)
        if num and street:
            k_addr = (c_country, num, street)
            if k_addr in s1_addr_dict:
                for s1_id in s1_addr_dict[k_addr]:
                    if len(candidates_by_s1[s1_id]) < 15:
                        candidates_by_s1[s1_id].append(eid)
                hits += 1
    if rows_read % 2000000 == 0:
        print(f"  Scanned {rows_read:,} / 10,320,219 catalog rows ({rows_read/10320219*100:.1f}%) in {time.time()-t0:.1f}s | Hits: {hits:,}")

t1 = time.time()
print(f"\nCompleted catalog scan in {t1-t0:.2f}s! Total hits: {hits:,}")
print(f"Queries with candidates: {len(candidates_by_s1):,} / 100,000 ({len(candidates_by_s1)/1000:.1f}%)")
log_memory("After Catalog Scan")

conn.close()
