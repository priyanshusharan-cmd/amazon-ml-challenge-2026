import sqlite3
import pandas as pd
import sys
sys.path.append('code/business_entity_resolution/src')
from preprocess import normalize_text, get_compact_signature, decompose_address

val_cands = pd.read_csv('output/val_candidate_pairs_sample.tsv', sep='\t')
gt = pd.read_csv('output/val_gt_split.tsv', sep='\t')

merged = pd.merge(val_cands, gt, on='source1_entity_id', how='inner')

conn = sqlite3.connect('file:output/train_catalog_temp.db?mode=ro&immutable=1', uri=True)
cur = conn.cursor()

missed_pairs = []
found_pairs = []
for _, row in merged.iterrows():
    s1_id = row['source1_entity_id']
    if pd.isna(row['matched_entity_ids']):
        continue
    true_ids = set(str(row['matched_entity_ids']).split(','))
    cands = set(str(row['candidate_entity_ids']).split(','))
    for m in true_ids - cands:
        missed_pairs.append((s1_id, m))
    for f in true_ids.intersection(cands):
        found_pairs.append((s1_id, f))

print(f"Total true pairs in sample: {len(missed_pairs) + len(found_pairs)}")
print(f"Found in E5 dense candidates: {len(found_pairs)} ({len(found_pairs)/(len(missed_pairs)+len(found_pairs))*100:.2f}%)")
print(f"Missed by E5 dense candidates: {len(missed_pairs)} ({len(missed_pairs)/(len(missed_pairs)+len(found_pairs))*100:.2f}%)")

# Now let's test various candidate blocking strategies on the MISSED pairs:
import re
from collections import Counter

def get_name_tokens(name):
    norm = normalize_text(name)
    # filter stop words
    stopwords = {'private', 'limited', 'corporation', 'incorporated', 'llc', 'sarl', 'sas', 'sa', 'and', 'the', 'of', 'in', 'services', 'solutions', 'enterprises', 'company', 'co'}
    tokens = [w for w in re.findall(r'\b[a-z0-9]{3,}\b', norm) if w not in stopwords]
    return tokens

def get_address_tokens(addr):
    if not addr:
        return []
    clean, num, street, cs, pc = decompose_address(addr)
    # get road/area tokens >= 4 chars or digits
    stopwords = {'street', 'road', 'avenue', 'lane', 'drive', 'boulevard', 'suite', 'floor', 'near', 'opp', 'behind', 'beside', 'dist', 'state', 'india', 'united', 'states', 'france'}
    tokens = [w for w in re.findall(r'\b[a-z0-9]{3,}\b', clean) if w not in stopwords]
    return tokens

# Let's test what keys match the missed pairs
exact_comp_match = 0
prefix6_comp_match = 0
name_token2_match = 0
addr_token3_match = 0
addr_num_and_token_match = 0
union_match = 0

for s1_id, cand_id in missed_pairs:
    s1_row = cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id=?", (s1_id,)).fetchone()
    c_row = cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id=?", (cand_id,)).fetchone()
    if not s1_row or not c_row:
        continue
        
    s1_comp = get_compact_signature(s1_row[1])
    c_comp = get_compact_signature(c_row[1])
    
    # 1. Exact compact signature
    is_exact_comp = bool(s1_comp and c_comp and s1_comp == c_comp)
    if is_exact_comp: exact_comp_match += 1
    
    # 2. Prefix 6 chars compact signature
    is_pref6 = bool(s1_comp and c_comp and len(s1_comp) >= 6 and len(c_comp) >= 6 and s1_comp[:6] == c_comp[:6])
    if is_pref6: prefix6_comp_match += 1
    
    # 3. Share at least 1 rare/distinct name token (length >= 4)
    s1_ntok = set(get_name_tokens(s1_row[1]))
    c_ntok = set(get_name_tokens(c_row[1]))
    is_ntok = bool(s1_ntok.intersection(c_ntok))
    if is_ntok: name_token2_match += 1
    
    # 4. Share at least 2 address tokens
    s1_atok = set(get_address_tokens(s1_row[2]))
    c_atok = set(get_address_tokens(c_row[2]))
    is_atok = bool(len(s1_atok.intersection(c_atok)) >= 2)
    if is_atok: addr_token3_match += 1
    
    # 5. Share street number AND at least 1 address token
    s1_addr, s1_num, s1_street, s1_cs, s1_pc = decompose_address(s1_row[2])
    c_addr, c_num, c_street, c_cs, c_pc = decompose_address(c_row[2])
    is_num_tok = bool(s1_num and c_num and s1_num == c_num and (s1_atok.intersection(c_atok)))
    if is_num_tok: addr_num_and_token_match += 1
    
    if is_exact_comp or is_pref6 or is_ntok or is_atok or is_num_tok:
        union_match += 1

print(f"\n--- Missed Candidates Recovery Rates ({len(missed_pairs)} total missed) ---")
print(f"1. Exact Compact Signature: {exact_comp_match}/{len(missed_pairs)} ({exact_comp_match/len(missed_pairs)*100:.1f}%)")
print(f"2. Compact Signature Prefix (len>=6, prefix-6): {prefix6_comp_match}/{len(missed_pairs)} ({prefix6_comp_match/len(missed_pairs)*100:.1f}%)")
print(f"3. Common Distinct Name Token (len>=4, non-stopword): {name_token2_match}/{len(missed_pairs)} ({name_token2_match/len(missed_pairs)*100:.1f}%)")
print(f"4. Common Address Tokens (>= 2 tokens): {addr_token3_match}/{len(missed_pairs)} ({addr_token3_match/len(missed_pairs)*100:.1f}%)")
print(f"5. Common House Num + Addr Token: {addr_num_and_token_match}/{len(missed_pairs)} ({addr_num_and_token_match/len(missed_pairs)*100:.1f}%)")
print(f"===> COMBINED UNION RECOVERY: {union_match}/{len(missed_pairs)} ({union_match/len(missed_pairs)*100:.1f}%) <===")
print(f"===> NEW TOTAL BLOCKING RECALL: {(len(found_pairs) + union_match)/(len(found_pairs) + len(missed_pairs))*100:.2f}% <===")

conn.close()
