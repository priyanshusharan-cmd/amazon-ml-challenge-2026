import sqlite3
import pandas as pd
import unidecode
import re
from collections import defaultdict

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

conn = sqlite3.connect('file:output/train_catalog_temp.db?mode=ro&immutable=1', uri=True)
cur = conn.cursor()

# Get missed pairs
missed_pairs = []
for s1_id in all_s1_needed:
    for m in gt_map[s1_id] - existing_cands[s1_id]:
        missed_pairs.append((s1_id, m))

print(f"Total missed true pairs: {len(missed_pairs)}")

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

matched_by_clean_brand = 0
matched_by_prefix8 = 0
matched_by_addr = 0
matched_by_any = 0

for s1_id, cand_id in missed_pairs:
    s1_row = cur.execute("SELECT business_name, business_address, country FROM s1_catalog WHERE entity_id=?", (s1_id,)).fetchone()
    c_row = cur.execute("SELECT business_name, business_address, country FROM catalog WHERE entity_id=?", (cand_id,)).fetchone()
    if not s1_row or not c_row:
        continue
    s1_b = clean_brand(s1_row[0])
    c_b = clean_brand(c_row[0])
    
    is_brand = bool(s1_b and c_b and len(s1_b) >= 4 and s1_b == c_b)
    is_pref8 = bool(s1_b and c_b and len(s1_b) >= 8 and len(c_b) >= 8 and s1_b[:8] == c_b[:8])
    
    s1_num, s1_street = fast_addr_key(s1_row[1])
    c_num, c_street = fast_addr_key(c_row[1])
    is_addr = bool(s1_num and c_num and s1_num == c_num and s1_street and c_street and s1_street == c_street)
    
    if is_brand: matched_by_clean_brand += 1
    if is_pref8: matched_by_prefix8 += 1
    if is_addr: matched_by_addr += 1
    if is_brand or is_pref8 or is_addr: matched_by_any += 1

print(f"Matched by clean_brand (exact): {matched_by_clean_brand}/{len(missed_pairs)} ({matched_by_clean_brand/len(missed_pairs)*100:.1f}%)")
print(f"Matched by prefix-8 brand: {matched_by_prefix8}/{len(missed_pairs)} ({matched_by_prefix8/len(missed_pairs)*100:.1f}%)")
print(f"Matched by exact street address: {matched_by_addr}/{len(missed_pairs)} ({matched_by_addr/len(missed_pairs)*100:.1f}%)")
print(f"===> TOTAL MISSED RECOVERED: {matched_by_any}/{len(missed_pairs)} ({matched_by_any/len(missed_pairs)*100:.1f}%)")
print(f"===> PROJECTED BLOCKING RECALL: {(initial_captured + matched_by_any)/total_true*100:.2f}%")

conn.close()
