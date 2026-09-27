"""Merge independently blocked country candidate Parquets into global artifacts."""

import argparse
import os
from pathlib import Path

import polars as pl

from src.config import config
from src.state_manager import StateManager


def merge_mode(mode: str, countries: list[str], output: Path, force: bool = False) -> None:
    inputs = [config.BLOCKED_DIR / country / f"{mode}_candidates.parquet" for country in countries]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing partition candidate files: " + ", ".join(map(str, missing)))
    if StateManager.should_skip(output, force=force):
        print(f"Existing merged candidate artifact {output}; skipping (use --force to rebuild).")
        return

    schemas = [pl.read_parquet_schema(path) for path in inputs]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(f"Candidate Parquet schemas differ for {mode}: {dict(zip(inputs, schemas))}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    pl.scan_parquet([str(path) for path in inputs]).sink_parquet(tmp, compression="snappy")
    os.replace(tmp, output)
    row_count = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in inputs)
    StateManager.write_artifact_metadata(
        output, row_count,
        meta={"mode": mode, "countries": countries, "inputs": [str(path) for path in inputs]},
    )
    print(f"Merged {len(inputs)} {mode} partitions ({row_count:,} rows) -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", required=True, help="Country subdirectories, e.g. India US")
    parser.add_argument("--modes", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--force", action="store_true", help="Rebuild merged artifacts even if outputs exist")
    args = parser.parse_args()
    for mode in args.modes:
        merge_mode(mode, args.countries, config.BLOCKED_DIR / f"{mode}_candidates.parquet", force=args.force)


if __name__ == "__main__":
    main()
