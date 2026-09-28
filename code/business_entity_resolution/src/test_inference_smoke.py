import os
import sys
sys.path.append('code/business_entity_resolution/src')
from inference import run_test_inference

base_dir = r"../"
test_dir = os.path.join(base_dir, "6ab10eb3b23ba_student_resource", "student_resource", "dataset", "test")
out_dir = os.path.join(base_dir, "output")

s1_path = os.path.join(test_dir, "test_source1.tsv")
s2_path = os.path.join(test_dir, "test_source2.tsv")
s3_path = os.path.join(test_dir, "test_source3.tsv")

cands_path = os.path.join(out_dir, "candidate_pairs.tsv")
out_matching_path = os.path.join(out_dir, "smoke_matching_results.tsv")

# Create a small candidate slice of 3000 queries
smoke_cands_path = os.path.join(out_dir, "smoke_cands.tsv")
with open(cands_path, 'r', encoding='utf-8') as f_in, open(smoke_cands_path, 'w', encoding='utf-8') as f_out:
    f_out.write(f_in.readline())
    for _ in range(3000):
        f_out.write(f_in.readline())

print("Running smoke inference on 3,000 queries...")
run_test_inference(smoke_cands_path, s1_path, s2_path, s3_path, out_dir, out_matching_path)
print(f"Smoke test output created: {out_matching_path} ({os.path.getsize(out_matching_path)} bytes)")

# Clean up
if os.path.exists(smoke_cands_path):
    os.remove(smoke_cands_path)
if os.path.exists(out_matching_path):
    os.remove(out_matching_path)
print("Smoke test PASSED successfully!")
