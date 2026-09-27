import json
import base64
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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


def _data_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


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

    def create_kernel_metadata(self, mode: str = "hybrid") -> Path:
        """Generate a self-contained notebook snapshot and its Kaggle metadata."""
        meta_dir = config.BASE_DIR / "notebooks"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "kernel-metadata.json"

        hybrid_artifacts = []
        hybrid_dataset_slug = None
        if mode == "hybrid":
            for p in config.FEATURES_DIR.glob("*.parquet"):
                hybrid_artifacts.extend([p, p.with_name(f"{p.name}.meta.json")])
            tc = config.BLOCKED_DIR / "test_candidates.parquet"
            if tc.exists():
                hybrid_artifacts.extend([tc, tc.with_name(f"{tc.name}.meta.json")])
            for mf in [config.PROGRESS_FILE, config.MANIFEST_FILE]:
                if mf.exists():
                    hybrid_artifacts.append(mf)
            
            hybrid_artifacts = [p for p in hybrid_artifacts if p.exists()]
            total_size = sum(p.stat().st_size for p in hybrid_artifacts)
            logger.info("[Hybrid Mode] Found %d artifacts totaling %.2f MB", len(hybrid_artifacts), total_size / (1024*1024))
            
            if total_size > 90 * 1024 * 1024:
                hybrid_dataset_slug = f"{config.KAGGLE_USERNAME}/amazon-ml-2026-hybrid-artifacts"
                logger.info("[Hybrid Mode] Bundle exceeds 90MB Kaggle Kernel limit. Falling back to Kaggle Dataset upload...")
                import tempfile
                with tempfile.TemporaryDirectory(prefix="kaggle-dataset-") as temp_dir:
                    td = Path(temp_dir)
                    for src in hybrid_artifacts:
                        dst = td / src.relative_to(config.ARTIFACTS_DIR)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(src, dst)
                    dataset_meta = {
                        "title": "Amazon ML 2026 Hybrid Artifacts",
                        "id": hybrid_dataset_slug,
                        "licenses": [{"name": "CC0-1.0"}]
                    }
                    import json
                    with open(td / "dataset-metadata.json", "w", encoding="utf-8") as f:
                        json.dump(dataset_meta, f)
                    import subprocess
                    res = subprocess.run([str(self.kaggle_exe), "datasets", "status", hybrid_dataset_slug], capture_output=True, text=True)
                    if res.returncode == 0:
                        logger.info("Updating existing Kaggle dataset...")
                        subprocess.run([str(self.kaggle_exe), "datasets", "version", "-m", "Auto-update hybrid artifacts", "-p", str(td)], check=True)
                    else:
                        logger.info("Creating new Kaggle dataset...")
                        subprocess.run([str(self.kaggle_exe), "datasets", "create", "-p", str(td)], check=True)
                    logger.info("Dataset upload triggered.")
                hybrid_artifacts = []
            else:
                logger.info("[Hybrid Mode] Artifacts are within limits. Bundling directly in Notebook ZIP.")

        import io
        import zipfile
        import base64
        bundle_io = io.BytesIO()
        with zipfile.ZipFile(bundle_io, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in (config.BASE_DIR / "src").glob("*.py"):
                bundle.write(path, f"src/{path.name}")
            validator_root = config.BASE_DIR / "info" / "data" / "student_resource"
            if validator_root.exists():
                for path in validator_root.rglob("*"):
                    if path.is_file() and ("utils" in path.parts or path.name in {"README.md", "Documentation_template.md"}):
                        bundle.write(path, str(Path("info/data/student_resource") / path.relative_to(validator_root)))
            if mode == "hybrid" and hybrid_artifacts:
                for path in hybrid_artifacts:
                    bundle.write(path, f"artifacts/{path.relative_to(config.ARTIFACTS_DIR)}")

        bundle_b64 = base64.b64encode(bundle_io.getvalue()).decode("ascii")
        
        # Build Notebook
        nb_cells = [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# Amazon ML Challenge 2026 — Full Pipeline (Mode: {mode})\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import base64, io, os, zipfile, sys, shutil\n",
                "from pathlib import Path\n",
                f"BUNDLE_B64 = {bundle_b64!r}\n",
                "PROJECT = Path('/kaggle/working/amazon_ml_pipeline')\n",
                "PROJECT.mkdir(parents=True, exist_ok=True)\n",
                "with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as zf: zf.extractall(PROJECT)\n",
                "os.environ['AMAZON_ML_BASE_DIR'] = str(PROJECT)\n",
                f"os.environ['AMAZON_ML_DATASET_DIR'] = '/kaggle/input/{config.KAGGLE_DATASET_SLUG.split('/')[-1]}'\n",
                "sys.path.insert(0, str(PROJECT))\n",
            ]}
        ]
        
        if mode == "hybrid" and hybrid_dataset_slug and not hybrid_artifacts:
            nb_cells[1]["source"].append(
                "hybrid_ds_path = Path('/kaggle/input/amazon-ml-2026-hybrid-artifacts')\n"
                "if hybrid_ds_path.exists():\n"
                "    for src in hybrid_ds_path.rglob('*'):\n"
                "        if src.is_file():\n"
                "            dst = PROJECT / 'artifacts' / src.relative_to(hybrid_ds_path)\n"
                "            dst.parent.mkdir(parents=True, exist_ok=True)\n"
                "            shutil.copy2(src, dst)\n"
                "    print('Copied hybrid artifacts from dataset mounted volume.')\n"
            )

        nb_cells[1]["source"].extend([
            "os.chdir(PROJECT)\n",
            "import subprocess\n",
            "subprocess.run([sys.executable, '-m', 'src.run_pipeline', '--stage', 'all'], cwd=PROJECT, check=True)\n",
            "for name in ('matching_results.tsv', 'candidate_pairs.tsv'):\n",
            "    src_f = PROJECT / 'output' / name\n",
            "    if src_f.exists():\n",
            "        shutil.copy2(src_f, Path('/kaggle/working') / name)\n",
            "for name in ('model.pkl', 'optimal_threshold.json'):\n",
            "    p = PROJECT / 'artifacts' / 'models' / name\n",
            "    if p.exists():\n",
            "        shutil.copy2(p, Path('/kaggle/working') / name)\n",
            "print('Submission outputs written to /kaggle/working')\n"
        ])

        notebook = {
            "cells": nb_cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
            "nbformat": 4, "nbformat_minor": 5
        }
        
        with open(meta_dir / "kaggle_runner.ipynb", "w", encoding="utf-8") as f:
            import json
            json.dump(notebook, f, ensure_ascii=False)

        dataset_sources = [f"{config.KAGGLE_USERNAME}/{config.KAGGLE_DATASET_SLUG}"] if '/' not in config.KAGGLE_DATASET_SLUG else [config.KAGGLE_DATASET_SLUG]
        if mode == "hybrid" and hybrid_dataset_slug and not hybrid_artifacts:
            dataset_sources.append(hybrid_dataset_slug)

        meta_content = {
            "id": f"{config.KAGGLE_USERNAME}/{config.KAGGLE_KERNEL_SLUG}",
            "title": "Amazon ML Challenge 2026 — GBDT Entity Resolution",
            "code_file": "kaggle_runner.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": dataset_sources,
            "competition_sources": [],
            "kernel_sources": []
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2)

        StateManager.write_artifact_metadata(meta_dir / "kaggle_runner.ipynb", len(notebook["cells"]), meta={"kind": "generated_kernel_notebook"})
        StateManager.write_artifact_metadata(meta_path, 1, meta={"kind": "kaggle_kernel_metadata"})
            
        logger.info("[Kaggle CLI] Metadata created at %s", meta_path)
        return meta_path

    def deploy_kernel(self, dry_run: bool = True, mode: str = "hybrid"):
        """
        Triggers Kaggle CLI kernel push.
        If dry_run is True, verifies readiness without launching an actual remote compute run.
        """
        if not self.verify_kaggle_authentication():
            raise PermissionError("Kaggle API authentication failed. Verify %USERPROFILE%\\.kaggle\\kaggle.json.")

        self.verify_git_cleanliness()
        meta_path = self.create_kernel_metadata(mode=mode)

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
        with tempfile.TemporaryDirectory(prefix="kaggle-kernel-output-") as temp_dir:
            res = subprocess.run([
                str(self.kaggle_exe), "kernels", "output",
                f"{config.KAGGLE_USERNAME}/{config.KAGGLE_KERNEL_SLUG}",
                "-p", temp_dir
            ], capture_output=True, text=True)
            if res.returncode:
                raise RuntimeError(f"Kaggle output download failed: {res.stdout.strip() or res.stderr.strip()}")
            downloaded = Path(temp_dir)
            required = ["matching_results.tsv", "candidate_pairs.tsv", "model.pkl", "optimal_threshold.json"]
            missing = [name for name in required if not (downloaded / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Kaggle run output is missing required artifacts: {', '.join(missing)}")

            for name in required:
                source = downloaded / name
                destination = output_dir / name if name.endswith(".tsv") else models_dir / name
                shutil.copy2(source, destination)
                rows = _data_rows(destination) if name.endswith(".tsv") else 1
                StateManager.write_artifact_metadata(destination, rows, meta={"source": "kaggle_kernel_output"})
        logger.info("[Artifact Router] Copied predictions to %s and model artifacts to %s", output_dir, models_dir)

    def show_kernel_status(self) -> str:
        kernel_id = f"{config.KAGGLE_USERNAME}/{config.KAGGLE_KERNEL_SLUG}"
        result = subprocess.run([str(self.kaggle_exe), "kernels", "status", kernel_id], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Kaggle kernel status failed: {result.stderr.strip() or result.stdout.strip()}")
        print(result.stdout.strip())
        return result.stdout.strip()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle CLI Deployment & Artifact Synchronization Engine")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--push", action="store_true", help="Push kernel to Kaggle Cloud and start remote execution")
    group.add_argument("--download-only", action="store_true", help="Download output artifacts from existing Kaggle run without pushing")
    group.add_argument("--download", action="store_true", help="Alias for --download-only")
    group.add_argument("--status", action="store_true", help="Show the current Kaggle kernel run status")
    group.add_argument("--dry-run", action="store_true", help="Verify deployment configuration without pushing or downloading")
    parser.add_argument("--mode", choices=["hybrid", "cloud"], default="hybrid", help="Execution mode (default: hybrid)")
    args = parser.parse_args()

    deployer = KaggleDeployer()
    if args.download_only or args.download:
        deployer.download_artifacts()
    elif args.status:
        deployer.show_kernel_status()
    elif args.push:
        deployer.deploy_kernel(dry_run=False, mode=args.mode)
    else:
        deployer.deploy_kernel(dry_run=True, mode=args.mode)
