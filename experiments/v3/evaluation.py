"""Bounded-memory diagnostics for the exact macro-per-S1 F0.5.

All query predictions are counted, including queries whose true S1 lies outside
the evaluated fold. Omitting those queries would hide false positives.

Negative multiplicity is a sensitivity diagnostic: it gives accepted truly
unmatched queries extra FP weight while preserving every retained S1's truth
count. It is NOT a forecast of test/leaderboard performance. It cannot reproduce
new countries, different corruption mechanisms, or changes to retrieval.

The v2 confirmation split has already been opened. Using CONF here requires an
explicit acknowledgement and always reports it as reused validation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

DEFAULT_THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
STRESS_DESCRIPTION = (
    "Approximate distribution-shift diagnostic, not a target-score forecast. "
    "Each accepted unmatched query contributes multiplicity times its false "
    "positive count; retained truth counts and crossassigned matched-query "
    "false positives stay unchanged. Records whose true S1 was explicitly "
    "dropped count as unmatched. Model scores and candidate choices stay fixed."
)


def _column(table: Any, name: str) -> np.ndarray:
    value = table[name]
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    return array


def _integer_column(table: Any, name: str) -> np.ndarray:
    array = _column(table, name)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer IDs")
    return array


def _has_column(table: Any, name: str) -> bool:
    return name in (table if isinstance(table, Mapping) else table.columns)


class EvaluationContext:
    """Reusable evaluator; arrays fit within the current 16 GB machine's budget.

    ``s1_table`` needs dense ``s1_idx`` IDs 0..N-1 (any row order), ``country``
    and ``fold``. Extra columns such as ``name_core`` are accepted. ``qtruth``
    is an integer array indexed by q_idx; -1 denotes an unmatched query.
    Prediction tables need unique q_idx, s1_idx, and p (or p1) columns and may
    omit queries with no candidates. Truth for omitted queries still counts.

    Input arrays are referenced, not copied wholesale. At normal 10-million
    query / 2.2-million S1 scale the evaluator's extra array memory is below
    1 GB, using chunks for query-level operations.
    """

    def __init__(
        self,
        s1_table: Any,
        qtruth: np.ndarray,
        fold: str = "DEV",
        dropped_s1: Any = None,
        reused_conf: bool = False,
    ) -> None:
        if fold == "CONF" and not reused_conf:
            raise ValueError("CONF is spent validation; set reused_conf=True to acknowledge reuse")
        self.fold = fold
        self.reused_conf = fold == "CONF"
        ids = _integer_column(s1_table, "s1_idx")
        n_s1 = len(ids)
        if not n_s1 or not np.array_equal(np.sort(ids), np.arange(n_s1)):
            raise ValueError("s1_idx must contain each dense catalog ID 0..N-1 exactly once")
        country, folds = _column(s1_table, "country"), _column(s1_table, "fold")
        if len(country) != n_s1 or len(folds) != n_s1:
            raise ValueError("S1 table column lengths differ")
        order = np.argsort(ids)
        self.country = country[order]
        self.eval_mask = folds[order] == fold
        self.n_s1 = n_s1
        self.qtruth = np.asarray(qtruth)
        if self.qtruth.ndim != 1 or self.qtruth.dtype.kind not in "iu":
            raise ValueError("qtruth must be a one-dimensional integer array; use -1 for unmatched")
        if len(self.qtruth) and (self.qtruth.min() < -1 or self.qtruth.max() >= n_s1):
            raise ValueError("qtruth contains IDs outside the catalog")
        self.dropped = np.zeros(n_s1, dtype=bool)
        if dropped_s1 is not None:
            drop = np.asarray(dropped_s1)
            if drop.size:
                if drop.ndim != 1 or drop.dtype.kind not in "iu" or drop.min() < 0 or drop.max() >= n_s1:
                    raise ValueError("dropped_s1 must contain valid catalog IDs")
                self.dropped[drop] = True
        self.eval_mask &= ~self.dropped
        if not self.eval_mask.any():
            raise ValueError(f"No retained S1 entities in fold {fold!r}")
        self.truth_count = np.zeros(n_s1, dtype=np.int64)
        for start in range(0, len(self.qtruth), 500_000):
            truth = self.qtruth[start:start + 500_000]
            positive = truth[truth >= 0]
            self.truth_count += np.bincount(positive, minlength=n_s1)
        self.truth_count[self.dropped] = 0
        self.country_values = sorted(np.unique(self.country[self.eval_mask]).tolist())

    def _summary(self, tp: np.ndarray, fp: np.ndarray, mask: np.ndarray) -> dict:
        n = int(np.count_nonzero(mask))
        if n == 0:
            return dict(n_entities=0, macro_f05=None, tp=0, fp=0, fn=0,
                        precision_micro=None, recall_micro=None)
        # Slice before arithmetic: only the requested fold's arrays are copied.
        t, f, nt = tp[mask], fp[mask], self.truth_count[mask]
        fn = nt - t
        if np.any(fn < 0):
            raise ValueError("More true positive predictions than truth; check duplicate query IDs")
        denominator = 5 * t + 4 * f + fn
        scores = np.ones(n, dtype=np.float64)
        np.divide(5 * t, denominator, out=scores, where=denominator != 0)
        sum_tp, sum_fp, sum_fn = int(t.sum()), int(f.sum()), int(fn.sum())
        return {
            "n_entities": n,
            "macro_f05": float(scores.mean()),
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
            "precision_micro": sum_tp / (sum_tp + sum_fp) if sum_tp + sum_fp else None,
            "recall_micro": sum_tp / (sum_tp + sum_fn) if sum_tp + sum_fn else None,
        }

    def evaluate(
        self,
        predictions: Any,
        thresholds: Any = DEFAULT_THRESHOLDS,
        negative_multiplicities: Any = (1, 2, 4),
        chunk_rows: int = 500_000,
    ) -> dict:
        qids, sids = _integer_column(predictions, "q_idx"), _integer_column(predictions, "s1_idx")
        probability = _column(predictions, "p" if _has_column(predictions, "p") else "p1")
        if not (len(qids) == len(sids) == len(probability)):
            raise ValueError("Prediction column lengths differ")
        if len(qids):
            if qids.min() < 0 or qids.max() >= len(self.qtruth):
                raise ValueError("Prediction q_idx outside qtruth")
            if sids.min() < 0 or sids.max() >= self.n_s1:
                raise ValueError("Prediction s1_idx outside catalog")
            if np.unique(qids).size != len(qids):
                raise ValueError("Predictions must contain one candidate per q_idx at most")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Prediction probabilities must be finite and between 0 and 1")
        thresholds = tuple(float(t) for t in thresholds)
        if not thresholds or any(not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
            raise ValueError("Thresholds must be a nonempty sequence in [0, 1]")
        multiplicities = tuple(negative_multiplicities)
        if not multiplicities or any(isinstance(k, (bool, np.bool_)) or int(k) != k or k < 1 for k in multiplicities):
            raise ValueError("Negative multiplicities must be positive integers")
        multiplicities = tuple(int(k) for k in multiplicities)
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        results = []
        for threshold in thresholds:
            tp = np.zeros(self.n_s1, dtype=np.int64)
            fp_matched = np.zeros(self.n_s1, dtype=np.int64)
            fp_unmatched = np.zeros(self.n_s1, dtype=np.int64)
            accepted_all = 0
            for start in range(0, len(qids), chunk_rows):
                end = start + chunk_rows
                ps = sids[start:end]
                keep = (probability[start:end] >= threshold) & ~self.dropped[ps]
                accepted_all += int(np.count_nonzero(keep))
                # Count contributions only to evaluated targets; queries remain unrestricted.
                keep &= self.eval_mask[ps]
                si = ps[keep]
                actual = self.qtruth[qids[start:end][keep]]
                is_tp = actual == si
                unmatched = actual < 0
                known = ~unmatched
                unmatched[known] |= self.dropped[actual[known]]
                tp += np.bincount(si[is_tp], minlength=self.n_s1)
                fp_unmatched += np.bincount(si[~is_tp & unmatched], minlength=self.n_s1)
                fp_matched += np.bincount(si[~is_tp & ~unmatched], minlength=self.n_s1)
            for multiplicity in multiplicities:
                fp = fp_matched + multiplicity * fp_unmatched
                results.append({
                    "threshold": threshold,
                    "negative_multiplicity": multiplicity,
                    "accepted_original_queries": accepted_all,
                    "acceptance_rate_all_queries": accepted_all / len(self.qtruth) if len(self.qtruth) else None,
                    "overall": self._summary(tp, fp, self.eval_mask),
                    "by_country": {str(country): self._summary(tp, fp, self.eval_mask & (self.country == country))
                                   for country in self.country_values},
                    "zero_match": self._summary(tp, fp, self.eval_mask & (self.truth_count == 0)),
                })
        return {
            "fold": self.fold,
            "validation_status": "reused confirmation; not an independent sealed result" if self.reused_conf else "development diagnostics",
            "metric": "macro over retained S1 of 5TP/(5TP+4FP+FN); empty truth and prediction score 1",
            "all_query_predictions_counted": True,
            "n_queries": int(len(self.qtruth)),
            "n_predictions": int(len(qids)),
            "n_dropped_s1": int(np.count_nonzero(self.dropped)),
            "stress_description": STRESS_DESCRIPTION,
            "results": results,
        }


def evaluate_predictions(s1_table: Any, qtruth: np.ndarray, predictions: Any, *,
                         fold: str = "DEV", dropped_s1: Any = None,
                         reused_conf: bool = False, **kwargs: Any) -> dict:
    """Convenience wrapper; reuse EvaluationContext when evaluating several models."""
    return EvaluationContext(s1_table, qtruth, fold, dropped_s1, reused_conf).evaluate(predictions, **kwargs)
