import os
import sys
import time
import sqlite3
import random
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz

sys.path.append(os.path.dirname(__file__))
from preprocess import normalize_text, get_compact_signature, get_acronym, decompose_address

digits_cache = {}
def extract_digits(text):
    if not text:
        return ""
    if text not in digits_cache:
        digits_cache[text] = "".join(c for c in text if c.isdigit())
    return digits_cache[text]

norm_text_cache = {}
def get_norm_text(text):
    if not text:
        return ""
    if text not in norm_text_cache:
        norm_text_cache[text] = normalize_text(text)
    return norm_text_cache[text]

decomp_addr_cache = {}
def get_decomp_address(text):
    if not text:
        return ("", "", "", "", "")
    if text not in decomp_addr_cache:
        decomp_addr_cache[text] = decompose_address(text)
    return decomp_addr_cache[text]

def compute_15_features(s1_raw_name, s1_raw_addr, s1_country, c_raw_name, c_raw_addr, c_country, cand_id):
    s1_name = get_norm_text(s1_raw_name)
    c_name = get_norm_text(c_raw_name)
    
    s1_clean_addr, s1_num, s1_street, s1_cs, s1_pc = get_decomp_address(s1_raw_addr)
    c_clean_addr, c_num, c_street, c_cs, c_pc = get_decomp_address(c_raw_addr)
    
    # 1-4. Name similarities
    if s1_name and c_name:
        name_ratio = fuzz.ratio(s1_name, c_name) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
        name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
        name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    else:
        name_ratio = name_token_sort = name_token_set = name_partial = 0.0
        
    # 5. Compact signature match
    s1_comp = get_compact_signature(s1_raw_name)
    c_comp = get_compact_signature(c_raw_name)
    if s1_comp and c_comp and (s1_comp == c_comp or (len(s1_comp)>=6 and s1_comp in c_comp) or (len(c_comp)>=6 and c_comp in s1_comp)):
        name_compact_match = 1.0
    else:
        name_compact_match = 0.0
        
    # 6. Acronym match
    s1_acro = get_acronym(s1_raw_name)
    c_acro = get_acronym(c_raw_name)
    if (s1_acro and s1_acro == c_comp) or (c_acro and c_acro == s1_comp):
        name_acronym_match = 1.0
    else:
        name_acronym_match = 0.0
        
    # 7. is_addr_missing
    is_addr_missing = 1.0 if (not s1_clean_addr or not c_clean_addr) else 0.0
    
    # 8. street_num_match
    if s1_num and c_num:
        street_num_match = 1.0 if s1_num == c_num else 0.0
    elif not s1_num and not c_num:
        street_num_match = 0.5
    else:
        street_num_match = 0.5
        
    # 9. street_name_sim
    if s1_street and c_street:
        street_name_sim = fuzz.ratio(s1_street, c_street) / 100.0
    elif not s1_street and not c_street:
        street_name_sim = 0.5
    else:
        street_name_sim = 0.0
        
    # 10. city_state_sim
    if s1_cs and c_cs:
        city_state_sim = fuzz.token_set_ratio(s1_cs, c_cs) / 100.0
    elif not s1_cs and not c_cs:
        city_state_sim = 0.5
    else:
        city_state_sim = 0.0
        
    # 11. addr_token_sort
    if s1_clean_addr and c_clean_addr:
        addr_token_sort = fuzz.token_sort_ratio(s1_clean_addr, c_clean_addr) / 100.0
    else:
        addr_token_sort = 0.0
        
    # 12. digits_match
    s1_d = extract_digits(f"{s1_name} {s1_raw_addr}")
    c_d = extract_digits(f"{c_name} {c_raw_addr}")
    if s1_d and c_d:
        digits_match = 1.0 if s1_d == c_d else 0.0
    elif not s1_d and not c_d:
        digits_match = 1.0
    else:
        digits_match = 0.0
        
    # 13. country_match
    country_match = 1.0 if (s1_country and c_country and s1_country == c_country) else 0.0
    
    # 14. is_dba_pattern
    is_dba_pattern = 1.0 if (street_name_sim >= 0.85 and street_num_match == 1.0 and country_match == 1.0) else 0.0
    
    # 15. source_origin
    source_origin = 1.0 if cand_id.startswith('S2-') else 0.0
    
    return (name_ratio, name_token_sort, name_token_set, name_partial, name_compact_match, name_acronym_match,
            is_addr_missing, street_num_match, street_name_sim, city_state_sim, addr_token_sort, digits_match,
            country_match, is_dba_pattern, source_origin)

if __name__ == "__main__":
    db_path = r"../output/train_catalog_temp.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Fetch 5 sample S1 and candidates
    cur.execute("SELECT source1_entity_id, matched_entity_ids FROM ground_truth LIMIT 5")
    samples = cur.fetchall()
    print("Testing 15-feature extraction on real Ground Truth samples:")
    for s1_id, matches in samples:
        cur.execute("SELECT business_name, business_address, country FROM s1_catalog WHERE entity_id = ?", (s1_id,))
        s1_row = cur.fetchone()
        for cid in matches.split(','):
            cur.execute("SELECT business_name, business_address, country FROM catalog WHERE entity_id = ?", (cid,))
            c_row = cur.fetchone()
            if s1_row and c_row:
                feats = compute_15_features(s1_row[0], s1_row[1], s1_row[2], c_row[0], c_row[1], c_row[2], cid)
                print(f"\nS1: {s1_row[0]} ({s1_row[2]})")
                print(f"Cand ({cid}): {c_row[0]} ({c_row[2]})")
                print(f"  Name ratios: ratio={feats[0]:.2f}, sort={feats[1]:.2f}, set={feats[2]:.2f}, partial={feats[3]:.2f}")
                print(f"  Address: street_num={feats[7]:.1f}, street_sim={feats[8]:.2f}, city_sim={feats[9]:.2f}, addr_sort={feats[10]:.2f}")
                print(f"  Flags: compact={feats[4]}, acronym={feats[5]}, dba={feats[13]}, country={feats[12]}")
    conn.close()
    print("\nFeature smoke test PASSED successfully!")
