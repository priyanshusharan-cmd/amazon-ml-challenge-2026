"""Add direct role/text features to best-candidate batches, with bounded query reads.

Usage::
    python experiments/v3/enrich_rows.py v2train --limit-files 1
    python experiments/v3/enrich_rows.py v2train

Reads RUN/rows_<split>/*.parquet and writes the same names under roles_<split>.
Each output contains q_idx/s1_idx and direct_features.FEATURES only.
S1 text is read once. Queries are gathered by physical parquet row group using a
shared bounded LRU, so sparse patch batches never load the entire query corpus.
Completed outputs are reused only when code, text inputs, base row input, output
checksum, and row count match. Writes and manifest updates are atomic per file.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import os
from pathlib import Path
import time

from common import CACHE_DIR, RUN, atomic_json, atomic_parquet, memory, pl, sha, role_gate, role_dir
from direct_features import FEATURES, add_features

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


def _check_headroom(min_gb: float = 1.75) -> None:
    available = psutil.virtual_memory().available / 2**30
    if available < min_gb:
        raise RuntimeError(
            f"Safe stop: only {available:.2f} GiB RAM free before loading a text block. "
            "Completed enrichment files can be resumed."
        )


class RowGroupCache:
    """One byte-limited LRU shared by all query text readers."""

    def __init__(self, max_mb: float = 192):
        if max_mb <= 0:
            raise ValueError("query cache size must be positive")
        self.max_bytes = int(max_mb * 2**20)
        self.bytes = 0
        self.groups: OrderedDict[tuple, pa.Table] = OrderedDict()
        self.loads = 0

    def get(self, reader: "IndexedParquet", group: int) -> pa.Table:
        key = (str(reader.path), tuple(reader.columns), group)
        if key in self.groups:
            table = self.groups.pop(key)
            self.groups[key] = table
            return table
        _check_headroom()
        table = reader.file.read_row_group(group, columns=reader.columns, use_threads=False)
        self.loads += 1
        # An unusually large single group is returned directly, never cached.
        if table.nbytes > self.max_bytes:
            self.groups.clear()
            self.bytes = 0
            return table
        while self.groups and self.bytes + table.nbytes > self.max_bytes:
            _, old = self.groups.popitem(last=False)
            self.bytes -= old.nbytes
        self.groups[key] = table
        self.bytes += table.nbytes
        return table


class IndexedParquet:
    """Gather arbitrary row positions while reading only relevant row groups."""

    def __init__(self, path: Path, columns: list[str], cache: RowGroupCache):
        self.path = Path(path)
        self.columns = columns
        self.cache = cache
        self.file = pq.ParquetFile(self.path)
        self.n = self.file.metadata.num_rows
        self.ends = np.cumsum([
            self.file.metadata.row_group(i).num_rows for i in range(self.file.metadata.num_row_groups)
        ], dtype=np.int64)
        self.starts = np.r_[np.int64(0), self.ends[:-1]]

    def take(self, indices: np.ndarray) -> pl.DataFrame:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("row indices must be one-dimensional")
        if not len(indices):
            schema = self.file.schema_arrow
            empty = pa.Table.from_arrays(
                [pa.array([], type=schema.field(c).type) for c in self.columns], names=self.columns
            )
            return pl.from_arrow(empty)
        if indices.min() < 0 or indices.max() >= self.n:
            raise IndexError(f"query index is outside {self.path.name}: expected [0, {self.n})")
        group_ids = np.searchsorted(self.ends, indices, side="right")
        parts = []
        for group in np.unique(group_ids):
            positions = np.flatnonzero(group_ids == group)
            offsets = indices[positions] - self.starts[group]
            table = self.cache.get(self, int(group)).take(pa.array(offsets, type=pa.int64()))
            parts.append(pl.from_arrow(table).with_columns(pl.Series("_gather_order", positions)))
        return pl.concat(parts).sort("_gather_order").drop("_gather_order")


class QueryTexts:
    def __init__(self, split: str, cache_mb: float = 192, cache_dir: Path = CACHE_DIR):
        self.cache = RowGroupCache(cache_mb)
        self.readers = []
        for source in ("source2", "source3"):
            raw = IndexedParquet(cache_dir / f"{split}_{source}.parquet",
                                 ["business_name", "business_address"], self.cache)
            norm = IndexedParquet(cache_dir / f"{split}_{source}_norm.parquet",
                                  ["name_core", "addr_norm"], self.cache)
            if raw.n != norm.n:
                raise RuntimeError(f"raw/normalized row counts differ for {split}/{source}")
            self.readers.append((raw, norm))
        self.n_s2 = self.readers[0][0].n
        self.n = self.n_s2 + self.readers[1][0].n

    def close(self):
        for raw, norm in self.readers:
            raw.file.close()
            norm.file.close()
        self.cache.groups.clear()
        self.cache.bytes = 0

    def take(self, q_idx: np.ndarray) -> pl.DataFrame:
        q_idx = np.asarray(q_idx, dtype=np.int64)
        if q_idx.ndim != 1:
            raise ValueError("q_idx must be one-dimensional")
        if len(q_idx) and (q_idx.min() < 0 or q_idx.max() >= self.n):
            raise IndexError("q_idx is outside the source2/source3 corpus")
        pieces = []
        for source_index, (raw, norm) in enumerate(self.readers):
            positions = np.flatnonzero(q_idx < self.n_s2 if source_index == 0 else q_idx >= self.n_s2)
            if not len(positions):
                continue
            indices = q_idx[positions] - (self.n_s2 if source_index == 1 else 0)
            r, n = raw.take(indices), norm.take(indices)
            pieces.append(r.hstack(n).rename({
                "business_name": "q_name", "business_address": "q_addr",
                "name_core": "q_core", "addr_norm": "q_anorm",
            }).with_columns(pl.Series("_gather_order", positions)))
        if not pieces:
            return pl.DataFrame(schema={c: pl.String for c in ("q_name", "q_addr", "q_core", "q_anorm")})
        return pl.concat(pieces).sort("_gather_order").drop("_gather_order")


def _s1_texts(split: str) -> pl.DataFrame:
    memory(3)
    raw = pl.read_parquet(CACHE_DIR / f"{split}_source1.parquet",
                          columns=["business_name", "business_address"])
    norm = pl.read_parquet(CACHE_DIR / f"{split}_source1_norm.parquet", columns=["name_core", "addr_norm"])
    if raw.height != norm.height:
        raise RuntimeError("S1 raw and normalized tables have different row counts")
    return raw.hstack(norm).rename({
        "business_name": "s_name", "business_address": "s_addr",
        "name_core": "s_core", "addr_norm": "s_anorm",
    })


def _acquire_lock(path: Path) -> None:
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            owner = psutil.Process(old["pid"])
            live = abs(owner.create_time() - old["process_started"]) < 1
        except (json.JSONDecodeError, KeyError, psutil.Error, TypeError):
            live = False
        if live:
            raise RuntimeError(f"Enrichment is already owned by live process {old['pid']}")
        path.unlink()
    payload = json.dumps({"pid": os.getpid(), "process_started": psutil.Process().create_time()})
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _signature(split: str, row_manifest: dict) -> dict:
    here = Path(__file__).resolve().parent
    files = [here / n for n in ("enrich_rows.py", "direct_features.py", "common.py")]
    text_paths = [CACHE_DIR / f"{split}_{src}{suffix}.parquet"
                  for src in ("source1", "source2", "source3") for suffix in ("", "_norm")]
    hashes = {}
    for path in files + text_paths:
        hashes[str(path)] = sha(path)
    if split == "v2stress":
        reuse_manifest = role_dir("v2train") / "manifest.json"
        if reuse_manifest.exists():
            hashes[str(reuse_manifest)] = sha(reuse_manifest)
    return {"version": 2, "split": split, "features": FEATURES, "gate": "C3 p1 in [.001,.999]",
            "base_inputs": row_manifest["inputs"], "sha256": hashes}


def enrich(split: str, limit_files: int | None = None, batch_rows: int = 25000,
           query_cache_mb: float = 192) -> dict:
    if batch_rows < 1 or (limit_files is not None and limit_files < 1):
        raise ValueError("batch_rows and limit_files must be positive")
    source = RUN / f"rows_{split}"
    source_manifest = source / "manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Build best-pair rows first: {source_manifest}")
    source_state = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not source_state.get("complete") and limit_files is None:
        raise RuntimeError("Base row cache is incomplete; wait for build_rows.py or use --limit-files for a smoke run")
    files = [source / name for name in sorted(source_state.get("files", {}))]
    if not files:
        raise RuntimeError(f"No completed base row files are recorded in {source_manifest}")
    selected = files[:limit_files] if limit_files is not None else files
    dest = role_dir(split)
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / ".lock"
    _acquire_lock(lock)
    try:
        print(f"Checking enrichment code and text-input checksums for {split}", flush=True)
        signature = _signature(split, source_state)
        manifest_path = dest / "manifest.json"
        if manifest_path.exists():
            state = json.loads(manifest_path.read_text(encoding="utf-8"))
            if state.get("inputs") != signature:
                raise RuntimeError("Enrichment code/text/base inputs changed; preserve this cache and use a new output directory")
        else:
            state = {"inputs": signature, "files": {}, "complete": False}
            atomic_json(manifest_path, state)
        pending = []
        total = 0
        for path in selected:
            expected = source_state["files"][path.name]
            digest = sha(path)
            if digest != expected["sha"]:
                raise RuntimeError(f"Base row file checksum does not match its manifest: {path.name}")
            output = dest / path.name
            done = state["files"].get(path.name, {})
            if (done.get("input_sha") == digest and done.get("source_rows") == expected["rows"]
                    and output.exists() and done.get("sha") == sha(output)):
                total += done["rows"]
                continue
            pending.append((path, output, digest, expected["rows"]))
        print(f"{split}: {len(pending)} files need enrichment, {total:,} rows already verified", flush=True)
        if pending:
            s1 = _s1_texts(split)
            queries = QueryTexts(split, query_cache_mb)
            print(f"S1 text ready; query cache bounded at {query_cache_mb:.0f} MiB; {memory()}", flush=True)
            for file_i, (path, output, digest, expected_rows) in enumerate(pending, 1):
                memory(2.25)
                start = time.perf_counter()
                rows = pl.read_parquet(path)
                if rows.height != expected_rows:
                    raise RuntimeError(f"Base row count changed for {path.name}")
                if any(c in rows.columns for c in FEATURES):
                    raise RuntimeError("Base row cache already includes direct role features")
                rows = rows.filter(role_gate())
                pieces = []
                pending_rows = rows
                if split == "v2stress":
                    # Direct features are identical for an unchanged (query, S1)
                    # pair. Reuse only those pairs, never the context columns.
                    stem = path.stem.rsplit("_", 1)[0]
                    reuse_files = sorted(role_dir("v2train").glob(stem + "_*.parquet"))
                    if reuse_files:
                        reused = pl.concat([pl.read_parquet(p) for p in reuse_files])
                        reused = rows.select("q_idx", "s1_idx").join(reused, on=["q_idx", "s1_idx"], how="inner", validate="1:1")
                        pending_rows = rows.join(reused.select("q_idx", "s1_idx"), on=["q_idx", "s1_idx"], how="anti")
                        pieces.append(reused)
                for a in range(0, pending_rows.height, batch_rows):
                    _check_headroom(2.0)
                    b = pending_rows.slice(a, batch_rows)
                    qi, si = b["q_idx"].to_numpy(), b["s1_idx"].to_numpy()
                    if len(si) and (si.min() < 0 or si.max() >= s1.height):
                        raise IndexError(f"s1_idx outside source1 in {path.name}")
                    texts = queries.take(qi).hstack(s1[si])
                    features = add_features(texts)
                    if features.height != b.height or features.null_count().sum_horizontal().sum():
                        raise RuntimeError(f"Invalid direct features in {path.name}")
                    pieces.append(b.select("q_idx", "s1_idx").hstack(features))
                    del b, texts, features
                if pieces:
                    enriched = pl.concat(pieces)
                else:
                    enriched = rows.select("q_idx", "s1_idx").with_columns([pl.Series(c, [], dtype=pl.Float32) for c in FEATURES])
                atomic_parquet(output, enriched)
                state["files"][path.name] = {
                    "input_sha": digest, "sha": sha(output), "rows": enriched.height,
                    "source_rows": expected_rows,
                    "seconds": round(time.perf_counter() - start, 2),
                }
                state["complete"] = False
                atomic_json(manifest_path, state)
                total += enriched.height
                del rows, pieces, enriched
                gc.collect()
                print(f"{split} [{file_i}/{len(pending)}] {path.name}: {total:,} rows; "
                      f"{state['files'][path.name]['seconds']:.1f}s; "
                      f"query row-group reads {queries.cache.loads}; {memory()}", flush=True)
        state["rows"] = sum(v["rows"] for v in state["files"].values())
        state["complete"] = bool(source_state.get("complete") and len(selected) == len(files)
                                 and all(p.name in state["files"] for p in files))
        atomic_json(manifest_path, state)
        return state
    finally:
        if "queries" in locals():
            queries.close()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["v2train", "v2test", "v2stress"])
    parser.add_argument("--limit-files", type=int, default=None, help="Enrich only the first N base files for a smoke run")
    parser.add_argument("--batch-rows", type=int, default=25000, help="Maximum pairs per direct-feature call")
    parser.add_argument("--query-cache-mb", type=float, default=192, help="Shared maximum Arrow query row-group cache size")
    args = parser.parse_args()
    enrich(args.split, args.limit_files, args.batch_rows, args.query_cache_mb)
