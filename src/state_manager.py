import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, List

class StateManager:
    """
    Restartable pipeline state and integrity manager.
    Atomically checkpoints progress to progress.json and manifest.json.
    """
    def __init__(self, progress_path: Path, manifest_path: Path):
        self.progress_path = Path(progress_path)
        self.manifest_path = Path(manifest_path)
        self._ensure_init()
        
    def _ensure_init(self):
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.progress_path.exists():
            initial_progress = {
                'stage_1_data_loading': {'status': 'PENDING'},
                'stage_2_normalization': {'status': 'PENDING'},
                'stage_3_blocking': {'status': 'PENDING'},
                'stage_4_feature_engineering': {'status': 'PENDING'},
                'stage_5_training': {'status': 'PENDING'},
                'stage_6_inference': {'status': 'PENDING'}
            }
            self._atomic_write(self.progress_path, initial_progress)
            
        if not self.manifest_path.exists():
            initial_manifest = {
                'created_at': time.time(),
                'sources': {},
                'artifacts': {}
            }
            self._atomic_write(self.manifest_path, initial_manifest)

    def _atomic_write(self, path: Path, data: Dict[str, Any]):
        tmp_path = path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load_progress(self) -> Dict[str, Any]:
        with open(self.progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def is_stage_completed(self, stage_name: str) -> bool:
        prog = self.load_progress()
        return prog.get(stage_name, {}).get('status') == 'COMPLETED'

    def mark_in_progress(self, stage_name: str, meta: Optional[Dict[str, Any]] = None):
        prog = self.load_progress()
        prog[stage_name] = {
            'status': 'IN_PROGRESS',
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'metadata': meta or {}
        }
        self._atomic_write(self.progress_path, prog)

    def mark_completed(self, stage_name: str, meta: Optional[Dict[str, Any]] = None):
        prog = self.load_progress()
        prog[stage_name] = {
            'status': 'COMPLETED',
            'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'metadata': meta or {}
        }
        self._atomic_write(self.progress_path, prog)

    # --- Source Fingerprinting & Chunk Registry (Adjustment 2) ---

    @staticmethod
    def compute_file_fingerprint(path: Path) -> Dict[str, Any]:
        p = Path(path)
        stat = p.stat()
        hasher = hashlib.md5()
        with open(p, 'rb') as f:
            chunk = f.read(65536)
            hasher.update(chunk)
        return {
            'path': str(p.resolve()),
            'size_bytes': stat.st_size,
            'mtime': stat.st_mtime,
            'md5_head_64k': hasher.hexdigest()
        }

    def register_source_file(self, source_key: str, path: Path) -> Dict[str, Any]:
        manifest = self.load_manifest()
        if 'sources' not in manifest:
            manifest['sources'] = {}
            
        current_fp = self.compute_file_fingerprint(path)
        existing = manifest['sources'].get(source_key)
        
        # Check if file has changed
        if existing and (
            existing.get('size_bytes') != current_fp['size_bytes'] or
            existing.get('md5_head_64k') != current_fp['md5_head_64k']
        ):
            manifest['sources'][source_key] = {
                **current_fp,
                'registered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'total_rows_processed': 0,
                'completed_chunks': []
            }
        elif not existing:
            manifest['sources'][source_key] = {
                **current_fp,
                'registered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'total_rows_processed': 0,
                'completed_chunks': []
            }
            
        self._atomic_write(self.manifest_path, manifest)
        return manifest['sources'][source_key]

    def is_chunk_completed(self, source_key: str, chunk_idx: int) -> bool:
        manifest = self.load_manifest()
        source_info = manifest.get('sources', {}).get(source_key, {})
        return chunk_idx in source_info.get('completed_chunks', [])

    def reset_source_progress(self, source_key: str) -> Dict[str, Any]:
        manifest = self.load_manifest()
        source_info = manifest.get("sources", {}).get(source_key)
        if source_info is None:
            return {}
        old_paths = [
            Path(value)
            for paths in source_info.get("partitions", {}).values()
            for value in paths
        ]
        for path in old_paths:
            path.unlink(missing_ok=True)
            path.with_name(f"{path.name}.meta.json").unlink(missing_ok=True)
        source_info["completed_chunks"] = []
        source_info["total_rows_processed"] = 0
        source_info["partitions"] = {}
        self._atomic_write(self.manifest_path, manifest)
        return source_info

    def verify_chunk_artifacts(self, source_key: str, chunk_idx: int) -> None:
        """Fail clearly if a completed ingestion chunk lost one of its registered outputs."""
        source_info = self.load_manifest().get("sources", {}).get(source_key, {})
        marker = f"_part_{chunk_idx:05d}.parquet"
        registered = [
            Path(value)
            for paths in source_info.get("partitions", {}).values()
            for value in paths
            if marker in value
        ]
        missing = [path for path in registered if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Ingestion checkpoint {source_key} chunk {chunk_idx} references missing output(s): "
                + ", ".join(map(str, missing))
                + ". Restore the missing partitions or clear that source checkpoint before rerunning."
            )

    def mark_chunk_completed(self, source_key: str, chunk_idx: int, rows_in_chunk: int, partition_files: Optional[Dict[str, str]] = None):
        manifest = self.load_manifest()
        if 'sources' not in manifest:
            manifest['sources'] = {}
        if source_key not in manifest['sources']:
            manifest['sources'][source_key] = {'completed_chunks': [], 'total_rows_processed': 0}
            
        src = manifest['sources'][source_key]
        if 'completed_chunks' not in src:
            src['completed_chunks'] = []
            
        if chunk_idx not in src['completed_chunks']:
            src['completed_chunks'].append(chunk_idx)
            src['completed_chunks'].sort()
            src['total_rows_processed'] = src.get('total_rows_processed', 0) + rows_in_chunk
            
        if partition_files:
            if 'partitions' not in src:
                src['partitions'] = {}
            for country, p_path in partition_files.items():
                if country not in src['partitions']:
                    src['partitions'][country] = []
                if p_path not in src['partitions'][country]:
                    src['partitions'][country].append(p_path)
                    
        self._atomic_write(self.manifest_path, manifest)

    @staticmethod
    def get_git_commit() -> str:
        try:
            import subprocess
            repo_root = Path(__file__).resolve().parent.parent
            res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def get_config_hash() -> str:
        try:
            from src.config import config
            from dataclasses import asdict
            values = {
                key: value for key, value in asdict(config).items()
                if not isinstance(value, Path)
            }
            cfg = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
            return hashlib.sha256(cfg.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "unknown"

    @classmethod
    def should_skip(cls, artifact_path: Path, force: bool = False) -> bool:
        if force:
            return False
        p = Path(artifact_path)
        if p.is_file():
            return p.stat().st_size > 0
        return p.is_dir() and any(p.iterdir())

    @staticmethod
    def require_artifact(artifact_path: Path, description: str = "Required artifact") -> Path:
        p = Path(artifact_path)
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f"{description} is missing or empty: {p}")
        return p

    @classmethod
    def write_artifact_metadata(cls, file_path: Path, row_count: int, meta: Optional[Dict[str, Any]] = None) -> Path:
        """Write the standard adjacent checkpoint sidecar without touching the manifest."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Cannot checkpoint missing artifact: {p}")
        entry = {
            "path": str(p.resolve()),
            "size_bytes": p.stat().st_size,
            "row_count": int(row_count),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": cls.get_git_commit(),
            "config_hash": cls.get_config_hash(),
            "metadata": meta or {},
        }
        meta_path = p.with_name(f"{p.name}.meta.json")
        StateManager._write_sidecar(meta_path, entry)
        return meta_path

    @staticmethod
    def _write_sidecar(path: Path, entry: Dict[str, Any]) -> None:
        tmp = path.with_name(f"{path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(entry, stream, indent=2)
        os.replace(tmp, path)

    def record_artifact(self, name: str, file_path: Path, row_count: int, meta: Optional[Dict[str, Any]] = None):
        manifest = self.load_manifest()
        if "artifacts" not in manifest:
            manifest["artifacts"] = {}
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Cannot checkpoint missing artifact: {p}")
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        commit_hash = self.get_git_commit()
        cfg_hash = self.get_config_hash()
        artifact_entry = {
            "path": str(p.resolve()),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "row_count": row_count,
            "timestamp": now_str,
            "updated_at": now_str,
            "git_commit": commit_hash,
            "config_hash": cfg_hash,
            "metadata": meta or {}
        }
        manifest["artifacts"][name] = artifact_entry
        self._atomic_write(self.manifest_path, manifest)

        # Write lightweight metadata sidecar file beside generated artifact
        meta_path = p.with_name(f"{p.name}.meta.json")
        self._write_sidecar(meta_path, artifact_entry)
