import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

class StateManager:
    def __init__(self, progress_path: Path, manifest_path: Path):
        self.progress_path = progress_path
        self.manifest_path = manifest_path
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
            initial_manifest = {'created_at': time.time(), 'artifacts': {}}
            self._atomic_write(self.manifest_path, initial_manifest)

    def _atomic_write(self, path: Path, data: Dict[str, Any]):
        tmp_path = path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load_progress(self) -> Dict[str, Any]:
        with open(self.progress_path, 'r', encoding='utf-8') as f:
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

    def record_artifact(self, name: str, file_path: Path, row_count: int, meta: Optional[Dict[str, Any]] = None):
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        manifest['artifacts'][name] = {
            'path': str(file_path),
            'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
            'row_count': row_count,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'metadata': meta or {}
        }
        self._atomic_write(self.manifest_path, manifest)
