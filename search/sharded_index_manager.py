"""Sharded FAISS index manager."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

from search.indexer import CodeIndexManager
from search.shard_manifest import ShardManifest


class ShardedIndexManager:
    """Manage multiple FAISS shards under a single project."""

    def __init__(self, index_root: str):
        self.index_root = Path(index_root)
        self.shards_root = self.index_root / "shards"
        self.manifest_path = self.index_root / "manifest.json"
        self.shards_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._managers: Dict[str, CodeIndexManager] = {}

        self._target_shard_bytes = int(
            os.getenv("CODE_SEARCH_SHARD_TARGET_BYTES", "536870912") or 536870912
        )

        self._manifest = self._load_or_init_manifest()
        self._active_shard_id = self._ensure_active_shard()

    @property
    def active_shard_id(self) -> str:
        return self._active_shard_id

    def active_manager(self) -> CodeIndexManager:
        return self._get_manager(self._active_shard_id)

    def _load_or_init_manifest(self) -> ShardManifest:
        if self.manifest_path.exists():
            return ShardManifest.load(self.manifest_path)
        manifest = ShardManifest(
            version=1,
            project_path="",
            embedding_dimension=0,
            index_type="flat",
            shard_count=0,
            shards=[],
        )
        manifest.save(self.manifest_path)
        return manifest

    def _ensure_active_shard(self) -> str:
        with self._lock:
            if self._manifest.shards:
                return self._manifest.shards[-1]["id"]
            return self._create_new_shard()

    def _create_new_shard(self) -> str:
        shard_id = f"shard_{len(self._manifest.shards):03d}"
        shard_path = self.shards_root / shard_id
        shard_path.mkdir(parents=True, exist_ok=True)
        self._manifest.shards.append(
            {
                "id": shard_id,
                "path": str(Path("shards") / shard_id),
                "vector_count": 0,
                "index_bytes": 0,
                "metadata_bytes": 0,
            }
        )
        self._manifest.shard_count = len(self._manifest.shards)
        self._manifest.save(self.manifest_path)
        return shard_id

    def _get_manager(self, shard_id: str) -> CodeIndexManager:
        with self._lock:
            manager = self._managers.get(shard_id)
            if manager is not None:
                return manager
            shard_path = self.shards_root / shard_id
            manager = CodeIndexManager(str(shard_path))
            self._managers[shard_id] = manager
            return manager

    def _maybe_rollover(self, current_shard_bytes: int) -> None:
        if current_shard_bytes <= self._target_shard_bytes:
            return
        self._active_shard_id = self._create_new_shard()

    def add_embeddings(self, embedding_results: List) -> None:
        manager = self.active_manager()
        manager.add_embeddings(embedding_results)

        stats = manager.get_stats()
        current_bytes = int(stats.get("storage_size", 0))
        self._update_manifest_for_shard(self._active_shard_id, stats)
        self._maybe_rollover(current_bytes)

    def remove_file_chunks(self, file_path: str) -> int:
        removed = 0
        for shard in self._manifest.shards:
            shard_id = shard["id"]
            manager = self._get_manager(shard_id)
            removed += manager.remove_file_chunks(file_path)
        return removed

    def save_index(self, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        for shard in self._manifest.shards:
            manager = self._get_manager(shard["id"])
            manager.save_index(extra_metadata=extra_metadata)
        self._manifest.save(self.manifest_path)

    def get_stats(self) -> Dict[str, Any]:
        total_chunks = 0
        storage_size = 0
        for shard in self._manifest.shards:
            shard_id = shard["id"]
            manager = self._get_manager(shard_id)
            stats = manager.get_stats()
            total_chunks += int(stats.get("total_chunks", 0))
            storage_size += int(stats.get("storage_size", 0))
            self._update_manifest_for_shard(shard_id, stats)
        return {
            "total_chunks": total_chunks,
            "storage_size": storage_size,
            "shard_count": len(self._manifest.shards),
        }

    def _update_manifest_for_shard(self, shard_id: str, stats: Dict[str, Any]) -> None:
        for shard in self._manifest.shards:
            if shard["id"] == shard_id:
                shard["vector_count"] = int(stats.get("total_chunks", 0))
                shard["index_bytes"] = int(stats.get("storage_size", 0))
                metadata_path = self.shards_root / shard_id / "metadata.db"
                shard["metadata_bytes"] = metadata_path.stat().st_size if metadata_path.exists() else 0
                break
        self._manifest.shard_count = len(self._manifest.shards)
        self._manifest.save(self.manifest_path)
