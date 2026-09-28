import pandas as pd

val_cands = pd.read_csv(r'../output/val_candidate_pairs_sample.tsv', sep='\t')
gt = pd.read_csv(r'../dataset/splits/val_gt_split.tsv', sep='\t')

merged = pd.merge(val_cands, gt, on='source1_entity_id', how='inner')

total_true_matches = 0
captured_matches = 0

for _, row in merged.iterrows():
    true_ids = set(str(row['matched_entity_ids']).split(','))
    cands = set(str(row['candidate_entity_ids']).split(','))
    
    total_true_matches += len(true_ids)
    captured_matches += len(true_ids.intersection(cands))

print(f'Blocking Recall (Top 30): {captured_matches/total_true_matches*100:.2f}% ({captured_matches}/{total_true_matches} true matches found)')
