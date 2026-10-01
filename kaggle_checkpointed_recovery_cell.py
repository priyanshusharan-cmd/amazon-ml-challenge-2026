"""Paste this entire file into ONE Kaggle notebook code cell.

The expensive pipeline cache lives in /kaggle/working/er_checkpoint so Kaggle
publishes it even when a saved run fails.  Scientific dependencies live in an
isolated temporary venv, and every pipeline stage runs in a fresh interpreter.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


COMMIT = "22cc08054eaed1f525b8f3d62257b7a5da60e6d6"
REPO_URL = "https://github.com/priyanshusharan-cmd/amazon-ml-challenge-2026.git"
EXPECTED_MATCH_SHA = "5a7f2c062a2e109fa0c8837e0bb0680a7dc6ccfb972636d994378ccd754c237e"
EXPECTED_CAND_SHA = "168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32"

# Persistent: everything expensive is here and is included in Kaggle Output.
WORK = Path("/kaggle/working/er_checkpoint")
CACHE = WORK / "cache"
GENERATED = WORK / "generated"
PROGRESS = WORK / "checkpoint_manifest.json"

# Temporary: cheap to recreate in a later session.
TEMP = Path("/tmp/er_checkpoint_runtime")
REPO = TEMP / "repo"
DATA = TEMP / "dataset"
VENV = TEMP / "venv"
PYTHON = VENV / "bin" / "python"
VENV_READY = VENV / ".bootstrap_complete"

FINAL_MATCH = Path("/kaggle/working/matching_results.tsv")
FINAL_CAND = Path("/kaggle/working/candidate_pairs.tsv")

WORK.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
GENERATED.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"commit": COMMIT, "completed_stages": [], "events": []}


def save_event(progress, stage, status):
    progress["commit"] = COMMIT
    progress["events"].append({"stage": stage, "status": status, "time": time.time()})
    progress["free_gb"] = round(shutil.disk_usage("/kaggle/working").free / 1024**3, 2)
    atomic_json(PROGRESS, progress)


def run(cmd, *, cwd=None, env=None):
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def restore_attached_checkpoint():
    """Restore a checkpoint attached with Add Input after a previous failed run."""
    if any(CACHE.iterdir()) or PROGRESS.exists():
        return
    candidates = sorted(Path("/kaggle/input").glob("**/er_checkpoint/checkpoint_manifest.json"))
    if not candidates:
        print("No attached prior checkpoint; starting a fresh persistent run.", flush=True)
        return
    source = candidates[-1].parent
    print("Restoring attached checkpoint from:", source, flush=True)
    for name in ("cache", "generated"):
        src = source / name
        dst = WORK / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    shutil.copy2(source / "checkpoint_manifest.json", PROGRESS)
    print("Checkpoint restored.", flush=True)


restore_attached_checkpoint()

# Clone the exact audited source revision. Repository source is not modified.
TEMP.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, REPO])
run(["git", "-C", REPO, "fetch", "--depth", "1", "origin", COMMIT])
run(["git", "-C", REPO, "checkout", "--detach", "FETCH_HEAD"])
actual_commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
assert actual_commit == COMMIT, (actual_commit, COMMIT)

BER = REPO / "business_entity_resolution"
SRC = BER / "src"

# Locate all seven challenge files and expose the layout expected by config.py.
required = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}
if DATA.exists():
    shutil.rmtree(DATA)
for split, names in required.items():
    target_dir = DATA / split
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        found = sorted(Path("/kaggle/input").glob(f"**/{name}"))
        if not found:
            raise FileNotFoundError(f"Missing Kaggle input: {name}")
        target = target_dir / name
        target.symlink_to(found[0])
        print(name, "->", found[0], flush=True)

# Create an isolated environment. Kaggle's /usr/bin/python3 has a broken
# ensurepip bootstrap, so create the venv without pip and seed it using the
# already-working outer pip. This still keeps every pipeline import in fresh
# child interpreters and prevents the previous stale-NumPy failure.
if not VENV_READY.exists():
    if VENV.exists():
        shutil.rmtree(VENV)
    run([sys.executable, "-m", "venv", "--without-pip",
         "--system-site-packages", VENV])
    run([sys.executable, "-m", "pip", "--python", PYTHON,
         "install", "--upgrade", "pip"])
    VENV_READY.touch()

requirements = []
for line in (BER / "requirements.txt").read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and not stripped.startswith("torch=="):
        requirements.append(stripped)
req_file = TEMP / "requirements_without_torch.txt"
req_file.write_text("\n".join(requirements) + "\n", encoding="utf-8")
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "-r", req_file])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "torch==2.6.0",
     "--index-url", "https://download.pytorch.org/whl/cu124"])

env = os.environ.copy()
env.update({
    "ER_DATA_DIR": str(DATA),
    "ER_CACHE_DIR": str(CACHE),
    "ER_OUTPUT_DIR": str(GENERATED),
    "PYTHONPATH": f"{SRC}:{BER}" + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""),
    "PYTHONUNBUFFERED": "1",
})

# Fail fast before any expensive work.
preflight = r'''
import numpy, scipy, scipy.sparse.linalg, polars, sklearn, lightgbm, torch
assert numpy.__version__ == "2.2.6", numpy.__version__
assert scipy.__version__ == "1.15.3", scipy.__version__
assert polars.__version__ == "1.44.2", polars.__version__
assert sklearn.__version__ == "1.7.2", sklearn.__version__
assert lightgbm.__version__ == "4.7.0", lightgbm.__version__
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.cuda.is_available()
x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
y = torch.topk(x @ x.T, 40, dim=1).values
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print("PRE-FLIGHT PASSED", numpy.__version__, scipy.__version__, torch.__version__)
'''
run([PYTHON, "-c", preflight], env=env)

# Protect the leaderboard file before doing expensive work.
source_match = REPO / "output" / "matching_results.tsv"
assert sha256(source_match) == EXPECTED_MATCH_SHA
if not FINAL_MATCH.exists():
    tmp_match = FINAL_MATCH.with_suffix(".tsv.tmp")
    shutil.copyfile(source_match, tmp_match)
    os.replace(tmp_match, FINAL_MATCH)
assert sha256(FINAL_MATCH) == EXPECTED_MATCH_SHA
print("Protected matching SHA256:", EXPECTED_MATCH_SHA, flush=True)

progress = load_progress()


def stage(name, command):
    if name in progress["completed_stages"]:
        print("SKIP COMPLETED:", name, flush=True)
        return
    free_gb = shutil.disk_usage("/kaggle/working").free / 1024**3
    if free_gb < 7:
        raise RuntimeError(f"Only {free_gb:.1f} GB free before {name}; checkpoint preserved, stopping safely")
    save_event(progress, name, "started")
    run(command, cwd=SRC, env=env)
    progress["completed_stages"].append(name)
    save_event(progress, name, "completed")
    print("CHECKPOINTED:", name, flush=True)


stage("prepare_data", [PYTHON, SRC / "prepare_data.py"])
stage("build_normalized", [PYTHON, SRC / "build_normalized.py", "train", "test"])
stage("indic_dictionary", [PYTHON, SRC / "indic_dictionary.py"])
stage("apply_indic_dict", [PYTHON, SRC / "apply_indic_dict.py", "train", "test"])

# candidates.py writes each chunk to .tmp and atomically renames it; existing
# completed chunks are skipped automatically after any restart.
stage("generate_candidates", [PYTHON, SRC / "candidates.py", "train", "test"])
stage("v2_prepare", [PYTHON, SRC / "v2" / "v2_prepare.py"])

# Export in a fresh interpreter and write atomically to /kaggle/working.
export_script = TEMP / "export_candidates.py"
export_script.write_text(r'''
import os
from pathlib import Path
import polars as pl
from train import id_tables
from predict import write_grouped

cache = Path(os.environ["ER_CACHE_DIR"])
dest = Path("/kaggle/working/candidate_pairs.tsv")
tmp = Path("/kaggle/working/candidate_pairs.tsv.tmp")
s1, q = id_tables("test")
files = sorted((cache / "v2test_cands").glob("*.parquet"))
if not files:
    raise RuntimeError("No v2 test candidate parquet files found")
pairs = pl.concat([pl.read_parquet(f, columns=["s1_idx", "q_idx"], memory_map=False) for f in files])
expected = 99_695_890
if pairs.height != expected:
    raise RuntimeError(f"Expected {expected:,} pairs, found {pairs.height:,}")
n, nonempty = write_grouped(s1["entity_id"], q["entity_id"], pairs,
                            "candidate_entity_ids", tmp)
os.replace(tmp, dest)
print(f"Exported {dest}: rows={n:,}, nonempty={nonempty:,}")
''', encoding="utf-8")
stage("export_candidate_pairs", [PYTHON, export_script])

match_sha = sha256(FINAL_MATCH)
cand_sha = sha256(FINAL_CAND)
print("matching_results.tsv SHA256:", match_sha, flush=True)
print("candidate_pairs.tsv SHA256:", cand_sha, flush=True)
print("expected candidate SHA256:", EXPECTED_CAND_SHA, flush=True)
assert match_sha == EXPECTED_MATCH_SHA, "matching_results.tsv changed"
assert cand_sha == EXPECTED_CAND_SHA, "candidate_pairs.tsv is not byte-identical to the expected artifact"

progress["final"] = {
    "matching_sha256": match_sha,
    "candidate_sha256": cand_sha,
    "verified": True,
    "time": time.time(),
}
atomic_json(PROGRESS, progress)
print("SUCCESS: BOTH OUTPUT FILES VERIFIED", flush=True)
