"""Export a frozen v3 candidate without touching the existing submission.

Uses the original candidate TSV by hard link: no 100M-pair materialization.
The caller must first run a passing frozen comparison and score v2test.
"""
import argparse
from itertools import zip_longest
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from common import BASELINE, CACHE_DIR, FIRST, ROOT, RUN, atomic_json, memory, pl, sha


def export(mode):
    modeldir = RUN / mode
    selection = json.loads((modeldir / "frozen_selection.json").read_text())
    fit = json.loads((modeldir / "fit.json").read_text())
    assert sha(modeldir / "model.txt") == fit["model_sha"], "Model changed after fitting"
    assert sha(modeldir / "pred_v2train.parquet") == selection["inputs"]["challenger_predictions_sha256"], "Frozen validation scores changed"
    for split in ("v2train", "v2test"):
        scoring = json.loads((modeldir / f"score_parts_{split}/manifest.json").read_text())
        assert scoring["inputs"]["model"] == fit["model_sha"], "Validation/test model mismatch"
    confirmation = json.loads((modeldir / "confirmation_reused.json").read_text())
    if confirmation["deltas"][0]["overall"] <= 0:
        raise RuntimeError("The frozen candidate did not improve reused confirmation")
    stress = json.loads((modeldir / "wrong_branch.json").read_text())
    if not stress["passes"]:
        raise RuntimeError("Wrong-branch stress gate failed")
    memory(3)
    predictions = pl.read_parquet(modeldir / "pred_v2test.parquet").sort("q_idx")
    baseline = pl.read_parquet(BASELINE / "top3_v2test.parquet", columns=["q_idx", "s1_idx", "p1"]).sort("q_idx")
    assert predictions.height == baseline.height
    for col in ("q_idx", "s1_idx"):
        assert np.array_equal(predictions[col].to_numpy(), baseline[col].to_numpy()), col
    w = np.float32(selection["baseline_weight"])
    p = (predictions["p"].to_numpy() * (np.float32(1) - w) + baseline["p1"].to_numpy() * w).astype(np.float32)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities"
    matches = predictions.select("q_idx", "s1_idx").filter(pl.Series(p >= selection["threshold"]))
    assert matches["q_idx"].n_unique() == matches.height
    matched_records = matches.height
    del predictions, baseline, p
    dest = ROOT / "output" / ("v3_" + mode)
    dest.mkdir(exist_ok=True)
    candidate_source = CACHE_DIR / "runs/submission_C4_stack_on_C3_direct_x_more_data/candidate_pairs.tsv"
    target = dest / "matching_results.tsv"
    s1_ids = pl.read_parquet(CACHE_DIR / "test_source1.parquet", columns=["entity_id"])["entity_id"]
    q_ids = pl.concat([pl.read_parquet(CACHE_DIR / f"test_{s}.parquet", columns=["entity_id"])
                       for s in ("source2", "source3")])["entity_id"]
    assert int(matches["q_idx"].max()) < len(q_ids)
    assert int(matches["s1_idx"].max()) < len(s1_ids)
    from predict import write_grouped
    temp = target.with_suffix(".tmp")
    n_rows, nonempty = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", temp)
    assert n_rows == len(s1_ids)
    # Bounded official-format and exact candidate-subset check, row by row.
    with temp.open(encoding="utf-8") as a, candidate_source.open(encoding="utf-8") as b:
        assert next(a).rstrip("\n") == "source1_entity_id\tmatched_entity_ids"
        assert next(b).rstrip("\n") == "source1_entity_id\tcandidate_entity_ids"
        checked = 0
        for left, right in zip_longest(a, b):
            assert left is not None and right is not None, "Mismatched S1 row coverage"
            sid, mids = left.rstrip("\n").split("\t")
            cid, cids = right.rstrip("\n").split("\t")
            assert sid == cid, "Mismatched S1 order"
            requested = mids.split(",") if mids else []
            assert len(requested) == len(set(requested))
            assert set(requested).issubset(set(cids.split(",") if cids else [])), sid
            checked += 1
        assert checked == n_rows
    os.replace(temp, target)
    candidate = dest / "candidate_pairs.tsv"
    if not candidate.exists():
        try:
            os.link(candidate_source, candidate)
        except OSError:
            if shutil.disk_usage(dest).free < candidate_source.stat().st_size + 2 * 2**30:
                raise RuntimeError("Not enough disk space for a candidate-file copy")
            shutil.copy2(candidate_source, candidate)
    assert sha(candidate) == sha(candidate_source)
    del q_ids, s1_ids, matches
    memory()
    # Matching-only official validation avoids loading 100M candidate strings.
    # The validator otherwise defaults to output/candidate_pairs.tsv even when
    # --candidate is omitted. Candidate coverage was already checked above.
    skipped_candidate = dest / "candidate_already_stream_validated.tsv"
    assert not skipped_candidate.exists()
    result = subprocess.run([sys.executable, str(ROOT / "utils/validate_submission.py"),
                             "--matching", str(target), "--candidate", str(skipped_candidate),
                             "--test-dir", str(ROOT / "dataset/test")],
                            text=True, capture_output=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # validator prints non-ASCII (Windows cp1252 otherwise)
    (dest / "validation.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Official matching-file validation failed; see validation.log")
    atomic_json(dest / "bundle.json", {"mode": mode, "threshold": selection["threshold"],
                "baseline_weight": float(w), "s1_rows": n_rows, "s1_with_matches": nonempty,
                "matched_records": matched_records, "model_sha256": fit["model_sha"],
                "matching_sha256": sha(target), "candidate_sha256": sha(candidate),
                "official_matching_validator": "PASS", "streamed_candidate_subset": "PASS",
                "leaderboard_score": None,
                "note": "Experimental candidate; leaderboard performance is unmeasured. Original output files preserved."})
    print(f"Exported and validated {target}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["direct", "roles"])
    export(ap.parse_args().mode)
