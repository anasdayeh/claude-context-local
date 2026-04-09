"""Sharded FAISS index manager."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common_utils import get_available_memory_bytes, get_total_memory_bytes
from search.indexer import CodeIndexManager
from search.hybrid import normalize_fts_query, rrf_fuse
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
        self.stats_path = self.index_root / "stats.json"
        self.storage_dir = str(self.index_root)
        self.shards_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._managers: Dict[str, CodeIndexManager] = {}
        self._loaded_shards: Dict[str, int] = {}
        self._lru: List[str] = []

        self._target_shard_bytes = int(
            os.getenv("CODE_SEARCH_SHARD_TARGET_BYTES", "536870912") or 536870912
        )
        self._max_bytes: Optional[int] = None

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

    def _get_manager(self, shard_id: str, enforce_budget: bool = True) -> CodeIndexManager:
        with self._lock:
            manager = self._managers.get(shard_id)
            if manager is not None:
                return manager
            shard_path = self.shards_root / shard_id
            manager = CodeIndexManager(str(shard_path))
            self._managers[shard_id] = manager
            self._mark_loaded(shard_id, manager.get_storage_bytes(), enforce_budget=enforce_budget)
            return manager

    def _maybe_rollover(self, current_shard_bytes: int) -> None:
        if current_shard_bytes <= self._target_shard_bytes:
            return
        self._active_shard_id = self._create_new_shard()

    def _compute_budget_bytes(self) -> int:
        raw_cap = str(os.getenv("CODE_SEARCH_SHARD_MEMORY_CAP_GB", "") or "").strip()
        if raw_cap:
            try:
                cap_gb = float(raw_cap)
            except Exception:
                cap_gb = 6.0
        else:
            total_bytes = get_total_memory_bytes()
            total_gb = (float(total_bytes) / (1024 ** 3)) if total_bytes > 0 else 0.0
            cap_gb = max(2.0, min(12.0, total_gb * 0.35 if total_gb > 0 else 6.0))
        cap_bytes = int(cap_gb * 1024 ** 3)
        available = get_available_memory_bytes()
        if available <= 0:
            return cap_bytes
        return min(cap_bytes, int(available * 0.75))

    def _mark_loaded(self, shard_id: str, bytes_used: int, enforce_budget: bool = True) -> None:
        self._loaded_shards[shard_id] = int(bytes_used)
        if shard_id in self._lru:
            self._lru.remove(shard_id)
        self._lru.append(shard_id)
        if enforce_budget:
            self._enforce_budget()

    def _enforce_budget(self) -> None:
        if not self._loaded_shards:
            return
        budget = self._max_bytes if self._max_bytes is not None else self._compute_budget_bytes()
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

    def add_embeddings(self, embedding_results: List, update_stats: bool = True) -> None:
        manager = self.active_manager()
        manager.add_embeddings(embedding_results, update_stats=update_stats)

        if update_stats:
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

        groups = self._build_shard_groups()
        all_results: List[List[Tuple[str, float, Dict[str, Any]]]] = []

        for group in groups:
            self._load_shard_group(group)

            def _search_shard(shard_id: str) -> List[Tuple[str, float, Dict[str, Any]]]:
                manager = self._get_manager(shard_id, enforce_budget=False)
                return manager.search(query_embedding, k=k, filters=filters)

            raw_workers = str(os.getenv("CODE_SEARCH_SHARD_SEARCH_WORKERS", "2") or "2").strip()
            try:
                worker_limit = max(1, int(raw_workers))
            except Exception:
                worker_limit = 2
            with ThreadPoolExecutor(max_workers=min(worker_limit, len(group))) as pool:
                for shard_results in pool.map(_search_shard, group):
                    all_results.append(shard_results)

        return merge_top_k(all_results, k)

    def search_hybrid(
        self,
        query_text: str,
        query_embedding,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        rrf_k: int = 60,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        dense_limit = dense_k or k
        sparse_limit = sparse_k or k
        dense_results = self.search(query_embedding, k=dense_limit, filters=filters)
        dense_ids = [cid for cid, _sim, _meta in dense_results]

        normalized_query = normalize_fts_query(query_text)
        sparse_ids: List[str] = []
        if normalized_query:
            for shard in self._manifest.shards:
                manager = self._get_manager(shard["id"], enforce_budget=False)
                for cid, _score in manager.fts_search(normalized_query, k=sparse_limit):
                    sparse_ids.append(cid)

        fused = rrf_fuse(dense_ids, sparse_ids, rrf_k=rrf_k, top_k=k)
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for cid, score in fused:
            meta = self.get_chunk_by_id(cid)
            if meta is None:
                continue
            results.append((cid, score, meta))
        if filters:
            results = self.active_manager().apply_filters(results, filters)
            results.sort(key=lambda item: item[1], reverse=True)
            results = results[:k]
        return results

    def fts_ready(self) -> bool:
        for shard in self._manifest.shards:
            manager = self._get_manager(shard["id"], enforce_budget=False)
            if not manager.fts_ready():
                return False
        return True

    def ensure_fts_async(self) -> None:
        for shard in self._manifest.shards:
            manager = self._get_manager(shard["id"], enforce_budget=False)
            manager.ensure_fts_async()

    def _estimate_shard_bytes(self, shard_id: str) -> int:
        for shard in self._manifest.shards:
            if shard["id"] == shard_id:
                estimate = int(shard.get("index_bytes", 0)) + int(shard.get("metadata_bytes", 0))
                if estimate > 0:
                    return estimate
        # Fallback to on-disk sizes
        shard_path = self.shards_root / shard_id
        total = 0
        for name in ("code.index", "metadata.db", "id_map.db"):
            p = shard_path / name
            if p.exists():
                total += p.stat().st_size
        return total

    def _build_shard_groups(self) -> List[List[str]]:
        budget = self._max_bytes if self._max_bytes is not None else self._compute_budget_bytes()
        if budget <= 0:
            return [[shard["id"] for shard in self._manifest.shards]]

        groups: List[List[str]] = []
        current: List[str] = []
        current_bytes = 0

        for shard in self._manifest.shards:
            shard_id = shard["id"]
            shard_bytes = max(self._estimate_shard_bytes(shard_id), 1)
            if current and current_bytes + shard_bytes > budget:
                groups.append(current)
                current = [shard_id]
                current_bytes = shard_bytes
                continue
            current.append(shard_id)
            current_bytes += shard_bytes

        if current:
            groups.append(current)

        return groups

    def _load_shard_group(self, group: List[str]) -> None:
        # Load required shards without enforcing budget per-shard.
        for shard_id in group:
            self._get_manager(shard_id, enforce_budget=False)
            self._mark_loaded(shard_id, self._estimate_shard_bytes(shard_id), enforce_budget=False)
        budget = self._max_bytes if self._max_bytes is not None else self._compute_budget_bytes()
        if len(group) == 1 and self._estimate_shard_bytes(group[0]) > budget:
            return
        # Evict shards not in the group if over budget.
        self._enforce_budget()

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
        self._write_root_stats(extra_metadata=extra_metadata)

    def get_stats(self) -> Dict[str, Any]:
        if self.stats_path.exists():
            try:
                return json.loads(self.stats_path.read_text())
            except Exception:
                pass
        return self._compute_root_stats()

    def _compute_root_stats(self) -> Dict[str, Any]:
        total_chunks = 0
        storage_size = 0
        file_paths = set()
        chunk_types: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}
        index_meta: Dict[str, Any] = {}
        training_count = 0
        training_seen = 0
        training_max = 0

        for shard in self._manifest.shards:
            shard_id = shard["id"]
            manager = self._get_manager(shard_id)
            stats = manager.get_stats()
            total_chunks += int(stats.get("total_chunks", 0))
            storage_size += int(stats.get("storage_size", 0))
            self._update_manifest_for_shard(shard_id, stats)
            if not index_meta:
                for key in ("index_type", "metric", "embedding_dim", "trained", "nlist", "nprobe"):
                    if key in stats:
                        index_meta[key] = stats.get(key)
            if "training_sample_count" in stats:
                training_count += int(stats.get("training_sample_count", 0))
            if "training_sample_total_seen" in stats:
                training_seen += int(stats.get("training_sample_total_seen", 0))
            if "training_sample_max" in stats:
                training_max = max(training_max, int(stats.get("training_sample_max", 0)))

            for _, meta in manager.metadata_db.items():
                meta = meta.get("metadata") if isinstance(meta, dict) else None
                if not isinstance(meta, dict):
                    continue
                path = meta.get("relative_path") or meta.get("file_path")
                if path:
                    file_paths.add(path)
                ctype = meta.get("chunk_type")
                if ctype:
                    chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
                tags = meta.get("tags") or []
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])

        stats = {
            "total_chunks": total_chunks,
            "files_indexed": len(file_paths),
            "chunk_types": chunk_types,
            "top_tags": top_tags,
            "storage_size": storage_size,
            "fts_rows": sum(self._get_manager(shard["id"])._get_fts_row_count() for shard in self._manifest.shards),
            "shard_count": len(self._manifest.shards),
        }
        if total_chunks == 0 and (file_paths or chunk_types or tag_counts):
            stats["sanity_warning"] = (
                "Index has metadata but zero vectors; FAISS index is missing or stale."
            )
            stats["sanity_suggestion"] = (
                "Reindex the project to rebuild the FAISS vectors."
            )
        if training_count:
            stats["training_sample_count"] = training_count
            stats["training_sample_total_seen"] = training_seen
            stats["training_sample_max"] = training_max
        stats.update(index_meta)
        return stats

    def _write_root_stats(self, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        stats = self._compute_root_stats()
        if extra_metadata:
            stats.update(extra_metadata)
        try:
            self.stats_path.write_text(json.dumps(stats, indent=2))
        except Exception:
            pass

    def repair_manifest_from_shards(self) -> Dict[str, Any]:
        """Rebuild manifest.json from existing shard directories."""
        shard_dirs = sorted(self.shards_root.glob("shard_*"))
        if not shard_dirs:
            return {"repaired": False, "reason": "no_shards"}

        shards: List[Dict[str, Any]] = []
        embedding_dim = 0
        index_type = "flat"
        project_path = self._manifest.project_path if self._manifest else ""

        for shard_dir in shard_dirs:
            shard_id = shard_dir.name
            manager = self._get_manager(shard_id)
            stats = manager.get_stats()
            if not embedding_dim:
                try:
                    embedding_dim = int(stats.get("embedding_dim") or 0)
                except Exception:
                    embedding_dim = 0
            if index_type == "flat":
                candidate = stats.get("index_type")
                if candidate:
                    index_type = candidate

            shards.append(
                {
                    "id": shard_id,
                    "path": str(Path("shards") / shard_id),
                    "vector_count": int(stats.get("total_chunks", 0)),
                    "index_bytes": int(stats.get("storage_size", 0)),
                    "metadata_bytes": (shard_dir / "metadata.db").stat().st_size
                    if (shard_dir / "metadata.db").exists()
                    else 0,
                }
            )

        self._manifest = ShardManifest(
            version=1,
            project_path=project_path,
            embedding_dimension=embedding_dim,
            index_type=index_type,
            shard_count=len(shards),
            shards=shards,
        )
        self._manifest.save(self.manifest_path)
        self._active_shard_id = shards[-1]["id"]
        self._write_root_stats()

        return {
            "repaired": True,
            "shard_count": len(shards),
            "shards": [s["id"] for s in shards],
        }

    def clear_index(self) -> None:
        for shard in self._manifest.shards:
            manager = self._get_manager(shard["id"])
            manager.clear_index()
        # Remove shard directories
        for shard_dir in self.shards_root.glob("shard_*"):
            for file in shard_dir.iterdir():
                try:
                    file.unlink()
                except Exception:
                    pass
        self._managers = {}
        self._loaded_shards = {}
        self._lru = []
        self._manifest = ShardManifest(
            version=1,
            project_path=self._manifest.project_path,
            embedding_dimension=self._manifest.embedding_dimension,
            index_type=self._manifest.index_type,
            shard_count=0,
            shards=[],
        )
        self._manifest.save(self.manifest_path)
        # Ensure a fresh active shard exists after a clear to avoid empty manifests.
        self._active_shard_id = self._create_new_shard()
        # Clear or regenerate cached root stats to avoid reporting stale counts.
        try:
            self.stats_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        self._write_root_stats()

    def _update_manifest_for_shard(self, shard_id: str, stats: Dict[str, Any]) -> None:
        # Fill in manifest-level metadata when available.
        project_path = stats.get("project_path")
        if project_path and not self._manifest.project_path:
            self._manifest.project_path = project_path

        embedding_dim = stats.get("embedding_dim")
        if embedding_dim and not self._manifest.embedding_dimension:
            try:
                self._manifest.embedding_dimension = int(embedding_dim)
            except Exception:
                pass

        index_type = stats.get("index_type")
        if index_type and self._manifest.index_type in {"", "flat"}:
            self._manifest.index_type = index_type

        for shard in self._manifest.shards:
            if shard["id"] == shard_id:
                shard["vector_count"] = int(stats.get("total_chunks", 0))
                shard["index_bytes"] = int(stats.get("storage_size", 0))
                metadata_path = self.shards_root / shard_id / "metadata.db"
                shard["metadata_bytes"] = metadata_path.stat().st_size if metadata_path.exists() else 0
                break
        self._manifest.shard_count = len(self._manifest.shards)
        self._manifest.save(self.manifest_path)
