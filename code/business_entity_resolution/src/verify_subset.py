import os

m_path = "output/matching_results.tsv"
c_path = "output/candidate_pairs.tsv"

print("Verifying that matching_results.tsv is a strict subset of candidate_pairs.tsv...")
errors = 0
total_checked = 0
total_matches = 0

with open(m_path, 'r', encoding='utf-8') as f_m, open(c_path, 'r', encoding='utf-8') as f_c:
    h_m = f_m.readline().strip()
    h_c = f_c.readline().strip()
    assert h_m == "source1_entity_id\tmatched_entity_ids"
    assert h_c == "source1_entity_id\tcandidate_entity_ids"
    
    for line_m, line_c in zip(f_m, f_c):
        total_checked += 1
        s1_m, tab_m, rest_m = line_m.partition('\t')
        s1_c, tab_c, rest_c = line_c.partition('\t')
        
        if s1_m != s1_c:
            print(f"Alignment error at query {total_checked}: {s1_m} != {s1_c}")
            errors += 1
            break
            
        m_ids = [x.strip() for x in rest_m.strip().split(',') if x.strip()]
        c_ids = set([x.strip() for x in rest_c.strip().split(',') if x.strip()])
        
        for mid in m_ids:
            total_matches += 1
            if mid not in c_ids:
                print(f"Subset violation at {s1_m}: matched {mid} not in candidates!")
                errors += 1
                if errors > 5:
                    break
                    
        if total_checked % 500000 == 0:
            print(f"  Verified {total_checked:,} queries ({total_matches:,} matches)...")

if errors == 0:
    print(f"\n100% PERFECT! All {total_checked:,} queries verified.")
    print(f"Total Matches: {total_matches:,} | Violations: 0")
else:
    print(f"\nFound {errors} errors!")
