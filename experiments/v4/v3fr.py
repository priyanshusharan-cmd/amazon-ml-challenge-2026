"""Run the FROZEN v3 role model on v4test (France-locale test split) without editing any v3 script.
Stages (each resumable inside the v3 functions): build_rows -> enrich_rows -> predict -> export.
Export mirrors experiments/v3/export.py (threshold/baseline_weight from frozen_selection.json) but writes to
output/v3_roles_fr/ and reads v4test inputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3"
sys.path.insert(0, str(V3))
import numpy as np  # noqa: E402

from common import BASELINE, CACHE_DIR, ROOT, RUN, atomic_json, memory, pl, sha  # noqa: E402

SPLIT = "v4test"


def stage_rows():
    _load_v3("build_rows").build(SPLIT)


def stage_enrich():
    _load_v3("enrich_rows").enrich(SPLIT)


def _load_v3(name):
    """import a v3 module by path (a same-named module exists in business_entity_resolution/src)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"v3_{name}", V3 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_predict():
    train = _load_v3("train")
    assert hasattr(train, "predict"), "wrong train module"
    train.predict("roles", SPLIT)


def stage_export():
    modeldir = RUN / "roles"
    sel = json.loads((modeldir / "frozen_selection.json").read_text())
    fit = json.loads((modeldir / "fit.json").read_text())
    assert sha(modeldir / "model.txt") == fit["model_sha"], "model changed"
    memory(3)
    pred = pl.read_parquet(modeldir / f"pred_{SPLIT}.parquet").sort("q_idx")
    base = pl.read_parquet(BASELINE / f"top3_{SPLIT}.parquet", columns=["q_idx", "s1_idx", "p1"]).sort("q_idx")
    assert pred.height == base.height and np.array_equal(pred["q_idx"].to_numpy(), base["q_idx"].to_numpy())
    assert np.array_equal(pred["s1_idx"].to_numpy(), base["s1_idx"].to_numpy())
    w = np.float32(sel["baseline_weight"])
    p = (pred["p"].to_numpy() * (np.float32(1) - w) + base["p1"].to_numpy() * w).astype(np.float32)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    matches = pred.select("q_idx", "s1_idx").filter(pl.Series(p >= sel["threshold"]))
    dest = ROOT / "output" / "v3_roles_fr"
    dest.mkdir(exist_ok=True)
    s1_ids = pl.read_parquet(CACHE_DIR / "test_source1.parquet", columns=["entity_id"])["entity_id"]
    q_ids = pl.concat([pl.read_parquet(CACHE_DIR / f"test_{s}.parquet", columns=["entity_id"]) for s in ("source2", "source3")])["entity_id"]
    sys.path.insert(0, str(ROOT / "business_entity_resolution" / "src"))
    from predict import write_grouped
    # candidate file for v4test (France candidates changed) comes from the C4+FR bundle of the same split
    cand_src = CACHE_DIR / "runs/submission_C4_stack_on_C3_direct_x_more_data_v4test/candidate_pairs.tsv"
    found = 0
    for f in sorted((CACHE_DIR / f"{SPLIT}_cands").glob("*.parquet")):
        found += matches.join(pl.read_parquet(f, columns=["s1_idx", "q_idx"], memory_map=False), on=["s1_idx", "q_idx"], how="semi").height
    assert found == matches.height, "matches not subset of candidates"
    tmp = dest / "matching_results.tmp"
    n_rows, nonempty = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", tmp)
    assert n_rows == len(s1_ids)
    os.replace(tmp, dest / "matching_results.tsv")
    cand = dest / "candidate_pairs.tsv"
    if not cand.exists():
        os.link(cand_src, cand)
    res = subprocess.run([sys.executable, str(ROOT / "utils/validate_submission.py"), "--matching", str(dest / "matching_results.tsv"),
                          "--candidate", str(dest / "__not_loaded__.tsv"), "--test-dir", str(ROOT / "dataset/test")],
                         text=True, capture_output=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    (dest / "validation.log").write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode:
        raise RuntimeError("validator failed")
    atomic_json(dest / "bundle.json", {"model": "v3 roles (frozen) on v4test (France-locale)", "threshold": sel["threshold"],
                "baseline_weight": float(w), "s1_rows": n_rows, "s1_with_matches": nonempty, "matched_records": matches.height,
                "matching_sha256": sha(dest / "matching_results.tsv"), "candidate_sha256": sha(cand),
                "official_matching_validator": "PASS", "streamed_candidate_subset": "PASS", "leaderboard_score": None})
    print("exported", dest, matches.height, flush=True)


if __name__ == "__main__":
    {"rows": stage_rows, "enrich": stage_enrich, "predict": stage_predict, "export": stage_export}[sys.argv[1]]()
