"""Build <team>_submission.zip for a chosen output folder (default: the recommended v4 variant).
usage: python make_submission_v4.py <team_name> [output_subdir=v4_c4fr_frp_t90]
Layout: output/{matching_results.tsv, candidate_pairs.tsv}, code/business_entity_resolution/{src/**, README.md, requirements.txt},
Documentation_template.md. The candidate file comes from the SAME folder as the matching file (same run)."""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(team, sub="v4_c4fr_frp_t90"):
    out_dir = ROOT / "output" / sub if sub != "." else ROOT / "output"
    files = {out_dir / "matching_results.tsv": "output/matching_results.tsv",
             out_dir / "candidate_pairs.tsv": "output/candidate_pairs.tsv",
             ROOT / "Documentation_template.md": "Documentation_template.md",
             ROOT / "business_entity_resolution" / "README.md": "code/business_entity_resolution/README.md",
             ROOT / "business_entity_resolution" / "requirements.txt": "code/business_entity_resolution/requirements.txt"}
    src = ROOT / "business_entity_resolution" / "src"
    for py in sorted(src.rglob("*.py")):
        files[py] = f"code/business_entity_resolution/src/{py.relative_to(src).as_posix()}"
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    zp = ROOT / f"{team}_submission_{sub.replace('/', '_')}.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for s, a in files.items():
            z.write(s, a)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None
    print(f"wrote {zp} ({zp.stat().st_size / 1e6:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "team", sys.argv[2] if len(sys.argv) > 2 else "v4_c4fr_frp_t90")
