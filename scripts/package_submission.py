import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging
import os
import argparse
import zipfile
from pathlib import Path

from src.config import config
from src.state_manager import StateManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PackageSubmission")


def _data_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def create_submission_zip(team_name: str = "antigravity_team", force: bool = False) -> Path:
    """
    Creates the official submission ZIP archive conforming strictly to contest requirements:
    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md
    """
    zip_filename = config.BASE_DIR / f"{team_name}_submission.zip"
    required_outputs = [config.OUTPUT_DIR / "matching_results.tsv", config.OUTPUT_DIR / "candidate_pairs.tsv"]
    missing = [path for path in required_outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Cannot package submission; required TSV artifacts are missing or empty: " + ", ".join(map(str, missing)))
    if StateManager.should_skip(zip_filename, force=force):
        logger.info("Existing submission package %s; skipping (use --force to rebuild).", zip_filename)
        return zip_filename
    logger.info("Creating official submission zip package at %s...", zip_filename)

    tmp_zip = zip_filename.with_name(f"{zip_filename.name}.tmp")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        for tsv in ["matching_results.tsv", "candidate_pairs.tsv"]:
            p = config.OUTPUT_DIR / tsv
            zf.write(p, f"output/{tsv}")
            logger.info("  + Added output/%s", tsv)

        # 2. Code files
        src_dir = config.BASE_DIR / "src"
        for py_file in src_dir.glob("*.py"):
            zf.write(py_file, f"code/business_entity_resolution/src/{py_file.name}")
            
        req_file = config.BASE_DIR / "requirements.txt"
        if req_file.exists():
            zf.write(req_file, "code/business_entity_resolution/requirements.txt")
            
        readme_file = config.BASE_DIR / "info" / "data" / "student_resource" / "README.md"
        if readme_file.exists():
            zf.write(readme_file, "code/business_entity_resolution/README.md")

        # 3. Documentation template
        doc_file = config.BASE_DIR / "info" / "data" / "student_resource" / "Documentation_template.md"
        if doc_file.exists():
            zf.write(doc_file, "Documentation_template.md")

    os.replace(tmp_zip, zip_filename)
    rows = sum(_data_rows(path) for path in required_outputs)
    StateManager.write_artifact_metadata(zip_filename, rows, meta={"team_name": team_name, "contents": [str(p.name) for p in required_outputs]})
    logger.info("Submission package ZIP created successfully: %s (%.2f MB)", zip_filename, zip_filename.stat().st_size / (1024*1024))
    return zip_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package validated submission TSVs and project code")
    parser.add_argument("--team-name", default="antigravity_team")
    parser.add_argument("--force", action="store_true", help="Rebuild the ZIP even if it already exists")
    options = parser.parse_args()
    create_submission_zip(team_name=options.team_name, force=options.force)
