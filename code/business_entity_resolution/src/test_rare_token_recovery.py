import sqlite3
import pandas as pd
import unidecode
import re
from collections import defaultdict, Counter

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

conn = sqlite3.connect('file:output/train_catalog_temp.db?mode=ro&immutable=1', uri=True)
cur = conn.cursor()

# Get missed pairs
missed_pairs = []
for s1_id in all_s1_needed:
    for m in gt_map[s1_id] - existing_cands[s1_id]:
        missed_pairs.append((s1_id, m))

print(f"Total missed true pairs: {len(missed_pairs)}")

# Stopwords
stopwords = {
    'private', 'limited', 'corporation', 'incorporated', 'llc', 'sarl', 'sas', 'sa', 'and', 'the', 'for', 'with',
    'services', 'solutions', 'enterprises', 'company', 'international', 'trading', 'group', 'industries',
    'pvt', 'ltd', 'inc', 'corp', 'india', 'united', 'states', 'france', 'north', 'south', 'east', 'west',
    'central', 'global', 'technologies', 'technology', 'management', 'holdings', 'consulting', 'associates'
}

def extract_tokens(text):
    if not text or text == 'None':
        return []
    if not text.isascii():
        text = unidecode.unidecode(text)
    words = re.findall(r'\b[a-z0-9]{4,}\b', text.lower())
    return [w for w in words if w not in stopwords and not w.isdigit()]

# Test token intersection on the missed pairs
token_matched = 0
for s1_id, cand_id in missed_pairs:
    s1_row = cur.execute("SELECT business_name, business_address, country FROM s1_catalog WHERE entity_id=?", (s1_id,)).fetchone()
    c_row = cur.execute("SELECT business_name, business_address, country FROM catalog WHERE entity_id=?", (cand_id,)).fetchone()
    if not s1_row or not c_row:
        continue
    s1_toks = set(extract_tokens(f"{s1_row[0]} {s1_row[1]}"))
    c_toks = set(extract_tokens(f"{c_row[0]} {c_row[1]}"))
    common = s1_toks.intersection(c_toks)
    if common:
        token_matched += 1

print(f"Missed pairs sharing >= 1 rare token (name or addr): {token_matched}/{len(missed_pairs)} ({token_matched/len(missed_pairs)*100:.1f}%)")
print(f"===> TOTAL RECOVERABLE WITH TOKENS: {initial_captured + token_matched}/{total_true} ({(initial_captured + token_matched)/total_true*100:.2f}%)")

conn.close()
