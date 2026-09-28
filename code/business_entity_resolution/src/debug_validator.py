import sys, traceback
sys.path.append('6ab10eb3b23ba_student_resource/student_resource/utils')
import validate_submission

try:
    validate_submission.validate('output/matching_results.tsv', 'output/candidate_pairs.tsv', '6ab10eb3b23ba_student_resource/student_resource/dataset/test')
except Exception as e:
    traceback.print_exc()
