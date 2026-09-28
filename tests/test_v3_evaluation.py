"""Hand-calculated checks for macro scoring and shifted-negative accounting."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "experiments" / "v3" / "evaluation.py"
SPEC = importlib.util.spec_from_file_location("v3_evaluation", MODULE)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        # Input rows intentionally unordered; IDs still dense.
        self.s1 = {"s1_idx": np.array([2, 0, 3, 1]),
                   "country": np.array(["India", "US", "US", "India"]),
                   "fold": np.array(["DEV", "DEV", "TRAIN", "DEV"])}
        self.truth = np.array([0, 0, 1, 3, -1, -1], dtype=np.int32)
        # q3 has TRAIN truth yet causes a DEV false positive. q4 is unmatched.
        self.pred = {"q_idx": np.array([0, 2, 3, 4]),
                     "s1_idx": np.array([0, 1, 0, 1]),
                     "p": np.array([.9, .9, .9, .9])}

    def test_exact_metric_and_only_unmatched_multiplicity(self):
        ctx = evaluation.EvaluationContext(self.s1, self.truth)
        out = ctx.evaluate(self.pred, thresholds=[.7], negative_multiplicities=[1, 4], chunk_rows=2)
        base, stress = out["results"]
        self.assertAlmostEqual(base["overall"]["macro_f05"], (.5 + 5 / 9 + 1) / 3)
        self.assertAlmostEqual(stress["overall"]["macro_f05"], (.5 + 5 / 21 + 1) / 3)
        self.assertEqual(base["overall"]["tp"], 2)
        self.assertEqual(base["overall"]["fp"], 2)
        self.assertEqual(stress["overall"]["fp"], 5)
        self.assertEqual(stress["overall"]["fn"], 1)
        self.assertAlmostEqual(base["overall"]["precision_micro"], .5)
        self.assertAlmostEqual(base["overall"]["recall_micro"], 2 / 3)
        self.assertEqual(base["zero_match"]["macro_f05"], 1)
        self.assertEqual(base["by_country"]["US"]["macro_f05"], .5)

    def test_dropped_truth_becomes_unmatched_without_changing_retained_truth(self):
        ctx = evaluation.EvaluationContext(self.s1, self.truth, dropped_s1=np.array([3]))
        row = ctx.evaluate(self.pred, thresholds=[.7], negative_multiplicities=[4])["results"][0]
        self.assertEqual(row["overall"]["fp"], 8)
        self.assertEqual(row["overall"]["fn"], 1)
        self.assertAlmostEqual(row["overall"]["macro_f05"], (5 / 22 + 5 / 21 + 1) / 3)

    def test_no_predictions_preserves_truth_and_empty_entities(self):
        ctx = evaluation.EvaluationContext(self.s1, self.truth)
        pred = {"q_idx": np.array([], dtype=int), "s1_idx": np.array([], dtype=int), "p1": np.array([])}
        row = ctx.evaluate(pred, thresholds=[.7], negative_multiplicities=[1])["results"][0]
        self.assertAlmostEqual(row["overall"]["macro_f05"], 1 / 3)
        self.assertEqual(row["overall"]["fn"], 3)
        self.assertIsNone(row["overall"]["precision_micro"])

    def test_duplicate_prediction_rejected(self):
        self.pred["q_idx"] = np.array([0, 0, 3, 4])
        with self.assertRaisesRegex(ValueError, "one candidate"):
            evaluation.EvaluationContext(self.s1, self.truth).evaluate(self.pred)

    def test_reused_confirmation_cannot_be_called_sealed(self):
        with self.assertRaisesRegex(ValueError, "spent validation"):
            evaluation.EvaluationContext(self.s1, self.truth, fold="CONF")
        self.s1["fold"] = np.array(["CONF", "CONF", "TRAIN", "CONF"])
        out = evaluation.EvaluationContext(self.s1, self.truth, fold="CONF", reused_conf=True).evaluate(
            self.pred, thresholds=[.7], negative_multiplicities=[1])
        self.assertIn("not an independent sealed", out["validation_status"])


if __name__ == "__main__":
    unittest.main()
