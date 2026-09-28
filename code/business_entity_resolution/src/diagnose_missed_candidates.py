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
for _, row in merged.iterrows():
    s1_id = row['source1_entity_id']
    if pd.isna(row['matched_entity_ids']):
        continue
    true_ids = set(str(row['matched_entity_ids']).split(','))
    cands = set(str(row['candidate_entity_ids']).split(','))
    missed = true_ids - cands
    for m in missed:
        missed_pairs.append((s1_id, m))

print(f"Total missed true pairs: {len(missed_pairs)}")

# Let's inspect the first 20 missed pairs
print("\nFirst 20 missed pairs breakdown:")
key_addr_match = 0
key_name_match = 0
key_either_match = 0

for i, (s1_id, cand_id) in enumerate(missed_pairs):
    s1_row = cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id=?", (s1_id,)).fetchone()
    c_row = cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id=?", (cand_id,)).fetchone()
    
    if not s1_row or not c_row:
        continue
        
    s1_name_comp = get_compact_signature(s1_row[1])
    c_name_comp = get_compact_signature(c_row[1])
    
    s1_addr, s1_num, s1_street, s1_cs, s1_pc = decompose_address(s1_row[2])
    c_addr, c_num, c_street, c_cs, c_pc = decompose_address(c_row[2])
    
    # Key checks
    addr_match = bool(s1_pc and c_pc and s1_pc == c_pc and s1_num and c_num and s1_num == c_num)
    name_match = bool(s1_name_comp and c_name_comp and (s1_name_comp == c_name_comp or s1_name_comp in c_name_comp or c_name_comp in s1_name_comp))
    
    if addr_match:
        key_addr_match += 1
    if name_match:
        key_name_match += 1
    if addr_match or name_match:
        key_either_match += 1
        
    if i < 15:
        print(f"\n--- Missed #{i+1} ---")
        print(f"S1: {s1_id} | Name: '{s1_row[1]}' | Addr: '{s1_row[2]}' | Country: '{s1_row[3]}'")
        print(f"      -> comp: '{s1_name_comp}' | pc: '{s1_pc}' | num: '{s1_num}' | street: '{s1_street}'")
        print(f"C : {cand_id} | Name: '{c_row[1]}' | Addr: '{c_row[2]}' | Country: '{c_row[3]}'")
        print(f"      -> comp: '{c_name_comp}' | pc: '{c_pc}' | num: '{c_num}' | street: '{c_street}'")
        print(f"      Matched by AddrKey: {addr_match} | Matched by NameKey: {name_match}")

conn.close()
print(f"\nSummary over all {len(missed_pairs)} missed pairs:")
print(f"Recoverable by AddrKey (pc + num): {key_addr_match}/{len(missed_pairs)} ({key_addr_match/len(missed_pairs)*100:.1f}%)")
print(f"Recoverable by NameKey (comp match/sub): {key_name_match}/{len(missed_pairs)} ({key_name_match/len(missed_pairs)*100:.1f}%)")
print(f"Recoverable by EITHER: {key_either_match}/{len(missed_pairs)} ({key_either_match/len(missed_pairs)*100:.1f}%)")
