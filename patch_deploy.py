import sys
import re

with open("scripts/deploy.py", "r") as f:
    content = f.read()

# Replace the argparse at the bottom
new_argparse = """    import argparse
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
        deployer.deploy_kernel(dry_run=True, mode=args.mode)"""

content = re.sub(r'    import argparse\n.*deployer\.deploy_kernel\(dry_run=True\)', new_argparse, content, flags=re.DOTALL)

# Update deploy_kernel signature
content = content.replace(
    'def deploy_kernel(self, dry_run: bool = True):',
    'def deploy_kernel(self, dry_run: bool = True, mode: str = "hybrid"):'
)

# Pass mode to create_kernel_metadata
content = content.replace(
    'meta_path = self.create_kernel_metadata()',
    'meta_path = self.create_kernel_metadata(mode=mode)'
)

# Update create_kernel_metadata
create_meta = """    def create_kernel_metadata(self, mode: str = "hybrid") -> Path:
        \"\"\"Generate a self-contained notebook snapshot and its Kaggle metadata.\"\"\"
        meta_dir = config.BASE_DIR / "notebooks"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "kernel-metadata.json"

        # Artifact gathering for hybrid mode
        hybrid_artifacts = []
        if mode == "hybrid":
            for p in config.FEATURES_DIR.glob("*.parquet"):
                hybrid_artifacts.extend([p, p.with_name(f"{p.name}.meta.json")])
            tc = config.BLOCKED_DIR / "test_candidates.parquet"
            if tc.exists():
                hybrid_artifacts.extend([tc, tc.with_name(f"{tc.name}.meta.json")])
            for mf in [config.PROGRESS_FILE, config.MANIFEST_FILE]:
                if mf.exists():
                    hybrid_artifacts.append(mf)
            
            # Ensure only existing files are in the list
            hybrid_artifacts = [p for p in hybrid_artifacts if p.exists()]
            total_size = sum(p.stat().st_size for p in hybrid_artifacts)
            logger.info("[Hybrid Mode] Found %d artifacts totaling %.2f MB", len(hybrid_artifacts), total_size / (1024*1024))
            
            hybrid_dataset_slug = f"{config.KAGGLE_USERNAME}/amazon-ml-2026-hybrid-artifacts"
            
            if total_size > 90 * 1024 * 1024:
                logger.info("[Hybrid Mode] Bundle exceeds 90MB Kaggle Kernel limit. Falling back to Kaggle Dataset upload...")
                # Create Kaggle Dataset programmatically
                with tempfile.TemporaryDirectory(prefix="kaggle-dataset-") as temp_dir:
                    td = Path(temp_dir)
                    for src in hybrid_artifacts:
                        dst = td / src.relative_to(config.ARTIFACTS_DIR)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                    
                    # Create dataset metadata
                    dataset_meta = {
                        "title": "Amazon ML 2026 Hybrid Artifacts",
                        "id": hybrid_dataset_slug,
                        "licenses": [{"name": "CC0-1.0"}]
                    }
                    with open(td / "dataset-metadata.json", "w", encoding="utf-8") as f:
                        json.dump(dataset_meta, f)
                    
                    # Check if dataset exists
                    res = subprocess.run([str(self.kaggle_exe), "datasets", "status", hybrid_dataset_slug], capture_output=True, text=True)
                    if res.returncode == 0:
                        logger.info("Updating existing Kaggle dataset...")
                        subprocess.run([str(self.kaggle_exe), "datasets", "version", "-m", "Auto-update hybrid artifacts", "-p", str(td)], check=True)
                    else:
                        logger.info("Creating new Kaggle dataset...")
                        subprocess.run([str(self.kaggle_exe), "datasets", "create", "-p", str(td)], check=True)
                    logger.info("Dataset upload triggered.")
                
                # We will not bundle the artifacts in the ZIP
                hybrid_artifacts = []
            else:
                hybrid_dataset_slug = None
                logger.info("[Hybrid Mode] Artifacts are within limits. Bundling directly in Notebook ZIP.")

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
        
        # Notebook generation
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": [f"# Amazon ML Challenge 2026 — Full Pipeline (Mode: {mode})\\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                    "import base64, io, os, zipfile, sys, shutil\\n",
                    "from pathlib import Path\\n",
                    f"BUNDLE_B64 = {bundle_b64!r}\\n",
                    "PROJECT = Path('/kaggle/working/amazon_ml_pipeline')\\n",
                    "PROJECT.mkdir(parents=True, exist_ok=True)\\n",
                    "with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as zf: zf.extractall(PROJECT)\\n",
                    "os.environ['AMAZON_ML_BASE_DIR'] = str(PROJECT)\\n",
                    f"os.environ['AMAZON_ML_DATASET_DIR'] = '/kaggle/input/{config.KAGGLE_DATASET_SLUG.split('/')[-1]}'\\n",
                    "sys.path.insert(0, str(PROJECT))\\n",
                ]}
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}},
            "nbformat": 4, "nbformat_minor": 5
        }
        
        # Copy external dataset artifacts into PROJECT if dataset fallback was used
        if mode == "hybrid" and hybrid_dataset_slug and not hybrid_artifacts:
            notebook["cells"][1]["source"].append(
                "import shutil\\n"
                f"hybrid_ds_path = Path('/kaggle/input/amazon-ml-2026-hybrid-artifacts')\\n"
                "if hybrid_ds_path.exists():\\n"
                "    for src in hybrid_ds_path.rglob('*'):\\n"
                "        if src.is_file():\\n"
                "            dst = PROJECT / 'artifacts' / src.relative_to(hybrid_ds_path)\\n"
                "            dst.parent.mkdir(parents=True, exist_ok=True)\\n"
                "            shutil.copy2(src, dst)\\n"
                "    print('Copied hybrid artifacts from dataset mounted volume.')\\n"
            )

        notebook["cells"][1]["source"].extend([
            "os.chdir(PROJECT)\\n",
            "import subprocess\\n",
            "subprocess.run([sys.executable, '-m', 'src.run_pipeline', '--stage', 'all'], cwd=PROJECT, check=True)\\n",
            "for name in ('matching_results.tsv', 'candidate_pairs.tsv'):\\n",
            "    src_f = PROJECT / 'output' / name\\n",
            "    if src_f.exists():\\n",
            "        shutil.copy2(src_f, Path('/kaggle/working') / name)\\n",
            "for name in ('model.pkl', 'optimal_threshold.json'):\\n",
            "    p = PROJECT / 'artifacts' / 'models' / name\\n",
            "    if p.exists():\\n",
            "        shutil.copy2(p, Path('/kaggle/working') / name)\\n",
            "print('Submission outputs written to /kaggle/working')\\n"
        ])
        
        with open(meta_dir / "kaggle_runner.ipynb", "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False)

        dataset_sources = [f"{config.KAGGLE_USERNAME}/{config.KAGGLE_DATASET_SLUG}"] if '/' not in config.KAGGLE_DATASET_SLUG else [config.KAGGLE_DATASET_SLUG]
        # Append hybrid dataset if we created it
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
        return meta_path"""

import re
# Find create_kernel_metadata definition and replace it
content = re.sub(
    r'    def create_kernel_metadata\(self\).*?return meta_path',
    create_meta,
    content,
    flags=re.DOTALL
)

with open("scripts/deploy.py", "w") as f:
    f.write(content)
