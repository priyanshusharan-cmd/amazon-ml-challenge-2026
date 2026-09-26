import json
import base64
import io
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.state_manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("KaggleDeployer")


class KaggleDeployer:
    """
    Kaggle CLI Deployment & Artifact Synchronization Engine.
    Handles Git status check, Kaggle authentication verification, kernel push, and artifact downloads.
    """
    def __init__(self):
        self.state_manager = StateManager(config.PROGRESS_FILE, config.MANIFEST_FILE)
        self.kaggle_exe = self._find_kaggle_cli()

    def _find_kaggle_cli(self) -> Path:
        venv_bin = "Scripts" if os.name == "nt" else "bin"
        executable = "kaggle.exe" if os.name == "nt" else "kaggle"
        venv_kaggle = config.BASE_DIR / ".venv" / venv_bin / executable
        if venv_kaggle.exists():
            return venv_kaggle
        sys_kaggle = shutil.which("kaggle")
        if sys_kaggle:
            return Path(sys_kaggle)
        raise FileNotFoundError("Kaggle CLI executable not found in .venv or PATH. Run 'make setup'.")

    def verify_kaggle_authentication(self) -> bool:
        """Verifies Kaggle API credentials by invoking 'kaggle competitions list'."""
        logger.info("[Kaggle CLI] Verifying API authentication...")
        try:
            res = subprocess.run([str(self.kaggle_exe), "competitions", "list"], capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                logger.info("[Kaggle CLI] Authentication VERIFIED for user '%s'.", config.KAGGLE_USERNAME)
                return True
            else:
                logger.error("[Kaggle CLI] Authentication FAILED: %s", res.stdout.strip())
                return False
        except Exception as e:
            logger.error("[Kaggle CLI] Authentication check error: %s", e)
            return False

    def verify_git_cleanliness(self) -> bool:
        """Verifies local Git status before triggering remote run."""
        logger.info("[Git Check] Inspecting local repository state...")
        try:
            res = subprocess.run(["git", "-C", str(config.BASE_DIR), "status", "--porcelain"], capture_output=True, text=True)
            if not res.stdout.strip():
                logger.info("[Git Check] Repository is CLEAN. Ready for Kaggle deployment.")
                return True
            else:
                logger.warning("[Git Check] Uncommitted changes detected:\n%s", res.stdout)
                return True  # Soft warning, allow proceeding
        except Exception as e:
            logger.warning("[Git Check] Could not check git status: %s", e)
            return True

    def create_kernel_metadata(self) -> Path:
        """Generate a self-contained notebook snapshot and its Kaggle metadata."""
        meta_dir = config.BASE_DIR / "notebooks"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "kernel-metadata.json"

        # Kaggle receives a notebook, not the local repository checkout. Embed the
        # exact project source snapshot so the uploaded notebook runs the real code.
        bundle_io = io.BytesIO()
        with zipfile.ZipFile(bundle_io, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in (config.BASE_DIR / "src").glob("*.py"):
                bundle.write(path, f"src/{path.name}")
            validator_root = config.BASE_DIR / "info" / "data" / "student_resource"
            for path in validator_root.rglob("*"):
                if path.is_file() and ("utils" in path.parts or path.name in {"README.md", "Documentation_template.md"}):
                    bundle.write(path, str(Path("info/data/student_resource") / path.relative_to(validator_root)))
        bundle_b64 = base64.b64encode(bundle_io.getvalue()).decode("ascii")
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["# Amazon ML Challenge 2026 — Full Pipeline\n", "Runs the bundled project source against the attached private dataset and writes submission artifacts to `/kaggle/working`.\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                    "import base64, io, os, zipfile, sys, shutil\n",
                    "from pathlib import Path\n",
                    f"BUNDLE_B64 = {bundle_b64!r}\n",
                    "PROJECT = Path('/kaggle/working/amazon_ml_pipeline')\n",
                    "PROJECT.mkdir(parents=True, exist_ok=True)\n",
                    "with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as zf: zf.extractall(PROJECT)\n",
                    "os.environ['AMAZON_ML_BASE_DIR'] = str(PROJECT)\n",
                    "os.environ['AMAZON_ML_DATASET_DIR'] = '/kaggle/input/amazon-ml-2026-dataset'\n",
                    "sys.path.insert(0, str(PROJECT))\n",
                    "assert (Path(os.environ['AMAZON_ML_DATASET_DIR']) / 'train' / 'train_source1.tsv').exists(), 'Attached dataset layout is missing train/train_source1.tsv'\n",
                    "os.chdir(PROJECT)\n",
                    "import subprocess\n",
                    "subprocess.run([sys.executable, '-m', 'src.run_pipeline', '--stage', 'all'], cwd=PROJECT, check=True)\n",
                    "for name in ('matching_results.tsv', 'candidate_pairs.tsv'):\n",
                    "    shutil.copy2(PROJECT / 'output' / name, Path('/kaggle/working') / name)\n",
                    "for name in ('model.pkl', 'optimal_threshold.json'):\n",
                    "    p = PROJECT / 'artifacts' / 'models' / name\n",
                    "    if p.exists(): shutil.copy2(p, Path('/kaggle/working') / name)\n",
                    "print('Submission outputs written to /kaggle/working')\n"
                ]}
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
            "nbformat": 4, "nbformat_minor": 5
        }
        with open(meta_dir / "kaggle_runner.ipynb", "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False)

        meta_content = {
            "id": f"{config.KAGGLE_USERNAME}/{config.KAGGLE_KERNEL_SLUG}",
            "title": "Amazon ML Challenge 2026 — GBDT Entity Resolution",
            "code_file": "kaggle_runner.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": [f"{config.KAGGLE_USERNAME}/{config.KAGGLE_DATASET_SLUG}"],
            "competition_sources": [],
            "kernel_sources": []
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2)
            
        logger.info("[Kaggle CLI] Metadata created at %s", meta_path)
        return meta_path

    def deploy_kernel(self, dry_run: bool = True):
        """
        Triggers Kaggle CLI kernel push.
        If dry_run is True, verifies readiness without launching an actual remote compute run.
        """
        if not self.verify_kaggle_authentication():
            raise PermissionError("Kaggle API authentication failed. Verify %USERPROFILE%\\.kaggle\\kaggle.json.")

        self.verify_git_cleanliness()
        meta_path = self.create_kernel_metadata()

        if dry_run:
            logger.info("[DRY RUN MODE] Kaggle deployment workflow verified successfully. Remote job NOT launched.")
            return {"status": "DRY_RUN_SUCCESS", "kernel_id": f"{config.KAGGLE_USERNAME}/{config.KAGGLE_KERNEL_SLUG}"}

        logger.info("[Kaggle CLI] Pushing notebook to Kaggle...")
        res = subprocess.run([str(self.kaggle_exe), "kernels", "push", "-p", str(meta_path.parent)], capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("[Kaggle CLI] Kernel successfully pushed! %s", res.stdout.strip())
        else:
            logger.error("[Kaggle CLI] Kernel push failed: %s", res.stderr.strip())
            raise RuntimeError(f"Kaggle kernel push failed: {res.stderr.strip() or res.stdout.strip()}")

    def download_artifacts(self, download_dir: Optional[Path] = None):
        """Downloads trained model.pkl and predictions from Kaggle kernel outputs and routes them to workspace folders."""
        models_dir = Path(download_dir or config.MODELS_DIR)
        output_dir = config.OUTPUT_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[Kaggle CLI] Pulling output artifacts from kernel '%s/%s'...", config.KAGGLE_USERNAME, config.KAGGLE_KERNEL_SLUG)
        res = subprocess.run([
            str(self.kaggle_exe), "kernels", "output",
            f"{config.KAGGLE_USERNAME}/{config.KAGGLE_KERNEL_SLUG}",
            "-p", str(models_dir)
        ], capture_output=True, text=True)

        if res.returncode == 0:
            logger.info("[Kaggle CLI] Raw output artifacts downloaded to %s", models_dir)
            matching_src = models_dir / "matching_results.tsv"
            candidate_src = models_dir / "candidate_pairs.tsv"
            missing_remote = [str(p.name) for p in (matching_src, candidate_src) if not p.exists()]
            if missing_remote:
                raise FileNotFoundError(f"Kaggle run output is missing required artifacts: {', '.join(missing_remote)}")
            
            shutil.copy2(matching_src, output_dir / matching_src.name)
            shutil.copy2(candidate_src, output_dir / candidate_src.name)
            logger.info("[Artifact Router] Copied submission TSVs into %s", output_dir)
        else:
            raise RuntimeError(f"Kaggle output download failed: {res.stdout.strip() or res.stderr.strip()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle CLI Deployment & Artifact Synchronization Engine")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--push", action="store_true", help="Push kernel to Kaggle Cloud and start remote execution")
    group.add_argument("--download-only", action="store_true", help="Download output artifacts from existing Kaggle run without pushing")
    group.add_argument("--dry-run", action="store_true", help="Verify deployment configuration without pushing or downloading")
    args = parser.parse_args()

    deployer = KaggleDeployer()
    if args.download_only:
        deployer.download_artifacts()
    elif args.push:
        deployer.deploy_kernel(dry_run=False)
    else:
        deployer.deploy_kernel(dry_run=True)
