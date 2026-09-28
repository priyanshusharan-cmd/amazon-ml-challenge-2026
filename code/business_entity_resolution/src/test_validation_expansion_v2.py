import sqlite3
import time
import pandas as pd
import unidecode
import re
from collections import defaultdict

print("Loading validation sample & ground truth...")
val_cands_df = pd.read_csv('output/val_candidate_pairs_sample.tsv', sep='\t')
gt_df = pd.read_csv('output/val_gt_split.tsv', sep='\t')
merged = pd.merge(val_cands_df, gt_df, on='source1_entity_id', how='inner')

gt_map = {}
existing_cands = {}
all_s1_needed = set()
for _, row in merged.iterrows():
    s1_id = row['source1_entity_id']
    all_s1_needed.add(s1_id)
    if pd.notna(row['matched_entity_ids']):
        gt_map[s1_id] = set(str(row['matched_entity_ids']).split(','))
    else:
        gt_map[s1_id] = set()
    existing_cands[s1_id] = set(str(row['candidate_entity_ids']).split(','))

total_true = sum(len(v) for v in gt_map.values())
initial_captured = sum(len(gt_map[s1].intersection(existing_cands[s1])) for s1 in all_s1_needed)
print(f"Initial E5 Candidate Recall: {initial_captured}/{total_true} ({initial_captured/total_true*100:.2f}%)")

def fast_compact(name):
    if not name or name == 'None':
        return ''
    if not name.isascii():
        name = unidecode.unidecode(name)
    clean = ''.join(c for c in name.lower() if c.isalnum())
    for suf in ('privatelimited', 'pvtlimited', 'pvtltd', 'private', 'limited', 'corporation', 'incorporated', 'corp', 'inc', 'llc', 'sarl', 'sas', 'sa', 'eurl', 'societe'):
        if clean.endswith(suf):
            clean = clean[:-len(suf)]
            break
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

# Connect to train DB
conn = sqlite3.connect('file:output/train_catalog_temp.db?mode=ro&immutable=1', uri=True)
cur = conn.cursor()

# Fetch S1 records
placeholders = ','.join(['?']*len(all_s1_needed))
cur.execute(f"SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id IN ({placeholders})", list(all_s1_needed))
s1_records = {r[0]: (r[1], r[2], r[3] or '') for r in cur.fetchall()}

# Build S1 lookup keys with defaultdict(list)
s1_compact_keys = defaultdict(list)
s1_addr_keys = defaultdict(list)

for s1_id, (name, addr, country) in s1_records.items():
    comp = fast_compact(name)
    if comp and len(comp) >= 4:
        s1_compact_keys[(country, comp)].append(s1_id)
    num, street = fast_addr_key(addr)
    if num and street:
        s1_addr_keys[(country, num, street)].append(s1_id)

print(f"Total S1 compact keys: {len(s1_compact_keys):,} | Addr keys: {len(s1_addr_keys):,}")

# Now scan catalog to find matches on compact keys & address keys
print("Scanning catalog for matches...")
t0 = time.time()
cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog")

new_candidates_added = defaultdict(set)
matched_catalog_rows = 0

batch_size = 500000
while True:
    rows = cur.fetchmany(batch_size)
    if not rows:
        break
    for eid, name, addr, country in rows:
        c_country = country or ''
        comp = fast_compact(name)
        if comp and len(comp) >= 4:
            key = (c_country, comp)
            if key in s1_compact_keys:
                for s1_id in s1_compact_keys[key]:
                    new_candidates_added[s1_id].add(eid)
                matched_catalog_rows += 1
        num, street = fast_addr_key(addr)
        if num and street:
            key_addr = (c_country, num, street)
            if key_addr in s1_addr_keys:
                for s1_id in s1_addr_keys[key_addr]:
                    new_candidates_added[s1_id].add(eid)

t1 = time.time()
print(f"Catalog scanned in {t1-t0:.2f}s!")

# Measure new recall
final_captured = 0
total_extra_cands = 0
for s1_id in all_s1_needed:
    union_set = existing_cands[s1_id].union(new_candidates_added.get(s1_id, set()))
    final_captured += len(gt_map[s1_id].intersection(union_set))
    total_extra_cands += len(new_candidates_added.get(s1_id, set()) - existing_cands[s1_id])

print(f"\nAfter Compact + Address Key Blocking:")
print(f"Recall: {final_captured}/{total_true} ({final_captured/total_true*100:.2f}%)  [+{(final_captured - initial_captured)/total_true*100:.2f}% gain!]")
print(f"Extra candidates added across {len(all_s1_needed):,} queries: {total_extra_cands:,} (avg {total_extra_cands/len(all_s1_needed):.2f} extra cands/query)")

conn.close()
