"""Reservoir sampling store for IVF training vectors."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from common_utils import get_total_memory_bytes


class TrainingSampleStore:
    def __init__(self, root_dir: str | Path, max_vectors: int = 25000):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.sample_path = self.root_dir / "training_sample.npy"
        self.meta_path = self.root_dir / "training_sample_meta.json"
        self.stats_path = self.root_dir / "training_sample_stats.json"
        self.max_vectors = max_vectors

        self._loaded = False
        self._vectors: List[np.ndarray] = []
        self._meta: List[Dict[str, Any]] = []
        self._total_seen = 0
        self._embedding_dim: Optional[int] = None
        self._model_name: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.sample_path.exists():
            try:
                vectors = np.load(self.sample_path)
                if vectors.size:
                    self._vectors = [vectors[i] for i in range(vectors.shape[0])]
                    self._embedding_dim = vectors.shape[1]
            except Exception:
                self._vectors = []
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text())
            except Exception:
                self._meta = []
        if self.stats_path.exists():
            try:
                stats = json.loads(self.stats_path.read_text())
                self._total_seen = int(stats.get("total_seen", 0))
                self._embedding_dim = self._embedding_dim or stats.get("embedding_dim")
                self._model_name = stats.get("model_name")
            except Exception:
                pass

        # Ensure meta length matches vectors if files got out of sync
        if self._meta and len(self._meta) != len(self._vectors):
            self._meta = self._meta[: len(self._vectors)]
        if not self._meta:
            self._meta = [{} for _ in self._vectors]

    def add(self, vector: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_loaded()
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._embedding_dim is None:
            self._embedding_dim = int(vec.shape[0])
        if self._model_name is None and meta:
            self._model_name = meta.get("model")

        self._total_seen += 1
        if len(self._vectors) < self.max_vectors:
            self._vectors.append(vec)
            self._meta.append(meta or {})
            return

        j = random.randint(0, self._total_seen - 1)
        if j < self.max_vectors:
            self._vectors[j] = vec
            self._meta[j] = meta or {}

    def add_batch(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        self._ensure_loaded()
        for vec, meta in zip(vectors, metas):
            self.add(vec, meta)

    def save(self) -> None:
        self._ensure_loaded()
        if self._vectors:
            vectors = np.stack(self._vectors, axis=0).astype(np.float32)
        else:
            dim = self._embedding_dim or 0
            vectors = np.empty((0, dim), dtype=np.float32)
        np.save(self.sample_path, vectors)
        self.meta_path.write_text(json.dumps(self._meta, indent=2))
        stats = {
            "total_seen": self._total_seen,
            "count": len(self._vectors),
            "max_vectors": self.max_vectors,
            "embedding_dim": self._embedding_dim,
            "model_name": self._model_name,
        }
        self.stats_path.write_text(json.dumps(stats, indent=2))

    def load(self) -> Dict[str, Any]:
        self._ensure_loaded()
        if self._vectors:
            vectors = np.stack(self._vectors, axis=0).astype(np.float32)
        else:
            dim = self._embedding_dim or 0
            vectors = np.empty((0, dim), dtype=np.float32)
        return {
            "vectors": vectors,
            "meta": list(self._meta),
            "stats": {
                "total_seen": self._total_seen,
                "count": len(self._vectors),
                "max_vectors": self.max_vectors,
                "embedding_dim": self._embedding_dim,
                "model_name": self._model_name,
            },
        }

    def clear(self) -> None:
        self._vectors = []
        self._meta = []
        self._total_seen = 0
        self._embedding_dim = None
        self._model_name = None
        self._loaded = True
        for path in (self.sample_path, self.meta_path, self.stats_path):
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass


def resolve_training_sample_max() -> int:
    value = str(os.getenv("CODE_SEARCH_TRAIN_SAMPLE_MAX", "") or "").strip()
    if value:
        try:
            return int(value)
        except Exception:
            pass

    total_bytes = get_total_memory_bytes()
    total_gb = (float(total_bytes) / (1024 ** 3)) if total_bytes > 0 else 0.0
    if total_gb <= 0:
        return 25000
    if total_gb <= 16:
        return 12000
    if total_gb <= 24:
        return 18000
    return 25000
