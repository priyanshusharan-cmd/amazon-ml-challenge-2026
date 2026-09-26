"""Merge independently blocked country candidate Parquets into global artifacts."""

import argparse
import os
from pathlib import Path

import polars as pl

from src.config import config


def merge_mode(mode: str, countries: list[str], output: Path) -> None:
    inputs = [config.BLOCKED_DIR / country / f"{mode}_candidates.parquet" for country in countries]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing partition candidate files: " + ", ".join(map(str, missing)))

    schemas = [pl.read_parquet_schema(path) for path in inputs]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(f"Candidate Parquet schemas differ for {mode}: {dict(zip(inputs, schemas))}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    pl.concat([pl.read_parquet(path) for path in inputs], how="vertical").write_parquet(tmp, compression="snappy")
    os.replace(tmp, output)
    print(f"Merged {len(inputs)} {mode} partitions ({sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in inputs):,} rows) -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", required=True, help="Country subdirectories, e.g. India US")
    parser.add_argument("--modes", nargs="+", choices=("train", "val"), default=("train", "val"))
    args = parser.parse_args()
    for mode in args.modes:
        merge_mode(mode, args.countries, config.BLOCKED_DIR / f"{mode}_candidates.parquet")


if __name__ == "__main__":
    main()
