"""Sharded FAISS index manager."""

from __future__ import annotations

import heapq
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common_utils import get_available_memory_bytes
from search.indexer import CodeIndexManager
from search.shard_manifest import ShardManifest


def merge_top_k(results: List[List[Tuple[str, float, Dict[str, Any]]]], k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Merge ranked shard results into a global top-k list."""
    merged: List[Tuple[str, float, Dict[str, Any]]] = []
    for shard_results in results:
        merged.extend(shard_results)
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged[:k]


class ShardedIndexManager:
    """Manage multiple FAISS shards under a single project."""

    def __init__(self, index_root: str):
        self.index_root = Path(index_root)
        self.shards_root = self.index_root / "shards"
        self.manifest_path = self.index_root / "manifest.json"
        self.shards_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._managers: Dict[str, CodeIndexManager] = {}
        self._loaded_shards: Dict[str, int] = {}
        self._lru: List[str] = []

        self._target_shard_bytes = int(
            os.getenv("CODE_SEARCH_SHARD_TARGET_BYTES", "536870912") or 536870912
        )
        self._max_bytes = self._compute_budget_bytes()

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
            self._mark_loaded(shard_id, manager.get_storage_bytes())
            return manager

    def _maybe_rollover(self, current_shard_bytes: int) -> None:
        if current_shard_bytes <= self._target_shard_bytes:
            return
        self._active_shard_id = self._create_new_shard()

    def _compute_budget_bytes(self) -> int:
        cap_gb = float(os.getenv("CODE_SEARCH_SHARD_MEMORY_CAP_GB", "13") or 13)
        cap_bytes = int(cap_gb * 1024 ** 3)
        available = get_available_memory_bytes()
        if available <= 0:
            return cap_bytes
        return min(cap_bytes, int(available * 0.75))

    def _mark_loaded(self, shard_id: str, bytes_used: int) -> None:
        self._loaded_shards[shard_id] = int(bytes_used)
        if shard_id in self._lru:
            self._lru.remove(shard_id)
        self._lru.append(shard_id)
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        if not self._loaded_shards:
            return
        budget = self._max_bytes or self._compute_budget_bytes()
        while sum(self._loaded_shards.values()) > budget and self._lru:
            victim = self._lru.pop(0)
            self._loaded_shards.pop(victim, None)
            manager = self._managers.pop(victim, None)
            if manager is not None:
                self._release_manager(manager)

    def _release_manager(self, manager: CodeIndexManager) -> None:
        try:
            if getattr(manager, "_metadata_db", None):
                manager._metadata_db.close()
            if getattr(manager, "_id_map_db", None):
                manager._id_map_db.close()
        except Exception:
            pass
        try:
            manager._index = None
        except Exception:
            pass

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

    def search(
        self,
        query_embedding,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        shard_ids = [shard["id"] for shard in self._manifest.shards]
        if not shard_ids:
            return []

        def _search_shard(shard_id: str) -> List[Tuple[str, float, Dict[str, Any]]]:
            manager = self._get_manager(shard_id)
            return manager.search(query_embedding, k=k, filters=filters)

        results: List[List[Tuple[str, float, Dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=min(4, len(shard_ids))) as pool:
            for shard_results in pool.map(_search_shard, shard_ids):
                results.append(shard_results)

        return merge_top_k(results, k)

    def iter_all_chunks(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        for shard in self._manifest.shards:
            manager = self._get_manager(shard["id"])
            for key, entry in manager.metadata_db.items():
                meta = entry.get("metadata") if isinstance(entry, dict) else None
                if not isinstance(meta, dict):
                    continue
                cid = entry.get("chunk_id") or str(key)
                yield cid, meta

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        for shard in self._manifest.shards:
            manager = self._get_manager(shard["id"])
            found = manager.get_chunk_by_id(chunk_id)
            if found:
                return found
        return None

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
