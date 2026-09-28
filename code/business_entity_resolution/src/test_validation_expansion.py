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

# Connect to train DB
conn = sqlite3.connect('file:output/train_catalog_temp.db?mode=ro&immutable=1', uri=True)
cur = conn.cursor()

# Fetch S1 records
placeholders = ','.join(['?']*len(all_s1_needed))
cur.execute(f"SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id IN ({placeholders})", list(all_s1_needed))
s1_records = {r[0]: (r[1], r[2], r[3] or '') for r in cur.fetchall()}

# Build S1 lookup keys
s1_compact_keys = {}
for s1_id, (name, addr, country) in s1_records.items():
    comp = fast_compact(name)
    if comp and len(comp) >= 4:
        s1_compact_keys[(country, comp)] = s1_id

print(f"Total S1 compact keys: {len(s1_compact_keys):,}")

# Now scan catalog to find matches on compact keys
print("Scanning catalog for compact matches...")
t0 = time.time()
cur.execute("SELECT entity_id, business_name, country FROM catalog")

new_candidates_added = defaultdict(set)
matched_catalog_rows = 0

batch_size = 500000
while True:
    rows = cur.fetchmany(batch_size)
    if not rows:
        break
    for eid, name, country in rows:
        comp = fast_compact(name)
        if comp and len(comp) >= 4:
            key = (country or '', comp)
            if key in s1_compact_keys:
                s1_id = s1_compact_keys[key]
                new_candidates_added[s1_id].add(eid)
                matched_catalog_rows += 1

t1 = time.time()
print(f"Catalog scanned in {t1-t0:.2f}s! Found {matched_catalog_rows:,} candidate hits.")

# Measure new recall
final_captured = 0
total_extra_cands = 0
for s1_id in all_s1_needed:
    union_set = existing_cands[s1_id].union(new_candidates_added.get(s1_id, set()))
    final_captured += len(gt_map[s1_id].intersection(union_set))
    total_extra_cands += len(new_candidates_added.get(s1_id, set()) - existing_cands[s1_id])

print(f"\nAfter Compact Key Blocking:")
print(f"Recall: {final_captured}/{total_true} ({final_captured/total_true*100:.2f}%)  [+{(final_captured - initial_captured)/total_true*100:.2f}% gain!]")
print(f"Extra candidates added across {len(all_s1_needed):,} queries: {total_extra_cands:,} (avg {total_extra_cands/len(all_s1_needed):.2f} extra cands/query)")

conn.close()
