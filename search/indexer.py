"""Vector index management with FAISS and metadata storage."""

import os
import json
import logging
import hashlib
import sqlite3
import time
import threading
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlitedict import SqliteDict
from embeddings.embedder import EmbeddingResult
from chunking.code_chunk import CodeChunk

# Reduce OpenMP/BLAS thread contention for stability.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_FAISS_MODULE = None


def _get_faiss():
    global _FAISS_MODULE
    if _FAISS_MODULE is None:
        import faiss as _faiss

        try:
            _faiss.omp_set_num_threads(1)
        except Exception:
            pass
        _FAISS_MODULE = _faiss
    return _FAISS_MODULE


class CodeIndexManager:
    """Manages FAISS vector index and metadata storage for code chunks."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.storage_dir / "code.index"
        self.metadata_path = self.storage_dir / "metadata.db"
        self.id_map_path = self.storage_dir / "id_map.db"
        self.file_map_path = self.storage_dir / "file_map.db"
        self.stats_path = self.storage_dir / "stats.json"
        
        # Initialize components
        self._index = None
        self._metadata_db = None
        self._id_map_db = None
        self._file_map_db = None
        self._training_sample = None
        self._logger = logging.getLogger(__name__)
        self._on_gpu = False
        self._legacy_index_map = None
        self._fts_lock = threading.Lock()
        self._fts_building = False

    def _resolve_fts_content(self, result: "EmbeddingResult") -> str:
        """Return a reusable content string (non-empty) for FTS insertion."""
        metadata = result.metadata or {}
        content = metadata.get("content") or ""
        if content and content.strip():
            return content
        chunk_content = result.chunk.content if result.chunk else ""
        if chunk_content and chunk_content.strip():
            return chunk_content
        name = metadata.get("name")
        if name:
            return str(name)
        chunk_type = metadata.get("chunk_type") or metadata.get("type")
        if chunk_type:
            return str(chunk_type)
        path = metadata.get("relative_path") or metadata.get("file_path")
        if path:
            return str(path)
        return ""

    @property
    def index(self):
        """Lazy loading of FAISS index."""
        if self._index is None:
            self._load_index()
        return self._index
    
    @property
    def metadata_db(self):
        """Lazy loading of metadata database."""
        if self._metadata_db is None:
            self._metadata_db = self._open_sqlitedict(self.metadata_path)
        return self._metadata_db

    @property
    def id_map_db(self):
        """Lazy loading of chunk_id -> int_id map."""
        if self._id_map_db is None:
            self._id_map_db = self._open_sqlitedict(self.id_map_path)
        return self._id_map_db

    @property
    def file_map_db(self):
        """Lazy loading of file_path -> [int_id] map."""
        if self._file_map_db is None:
            self._file_map_db = self._open_sqlitedict(self.file_map_path)
        return self._file_map_db

    @property
    def training_sample(self):
        """Lazy loading of training sample store."""
        if self._training_sample is None:
            from search.training_sample import TrainingSampleStore, resolve_training_sample_max
            max_vectors = resolve_training_sample_max()
            if max_vectors <= 0:
                return None
            self._training_sample = TrainingSampleStore(self.storage_dir, max_vectors=max_vectors)
        return self._training_sample

    def _fts_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.metadata_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=1000;")
        return conn

    def _ensure_fts_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "chunk_id UNINDEXED, path, content, tokenize='unicode61 tokenchars _'"
            ");"
        )

    def fts_upsert(self, chunk_id: str, path: str, content: str) -> None:
        if not chunk_id or not path or not content:
            return
        try:
            with self._fts_lock:
                with self._fts_connect() as conn:
                    self._ensure_fts_table(conn)
                    conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
                    conn.execute(
                        "INSERT INTO chunks_fts(chunk_id, path, content) VALUES (?, ?, ?)",
                        (chunk_id, path, content),
                    )
        except Exception:
            pass

    def fts_upsert_many(self, entries: List[Tuple[str, str, str]]) -> None:
        if not entries:
            return
        cleaned = [(cid, path, content) for cid, path, content in entries if cid and path and content]
        if not cleaned:
            return
        try:
            with self._fts_lock:
                with self._fts_connect() as conn:
                    self._ensure_fts_table(conn)
                    conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(cid,) for cid, _, _ in cleaned])
                    conn.executemany(
                        "INSERT INTO chunks_fts(chunk_id, path, content) VALUES (?, ?, ?)",
                        cleaned,
                    )
        except Exception:
            pass

    def fts_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if not query:
            return []
        try:
            with self._fts_connect() as conn:
                self._ensure_fts_table(conn)
                rows = conn.execute(
                    "SELECT chunk_id, bm25(chunks_fts) AS score "
                    "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                    (query, k),
                ).fetchall()
            return [(row[0], float(row[1])) for row in rows]
        except Exception:
            return []

    def fts_delete_by_path(self, path: str) -> None:
        if not path:
            return
        try:
            with self._fts_lock:
                with self._fts_connect() as conn:
                    self._ensure_fts_table(conn)
                    conn.execute("DELETE FROM chunks_fts WHERE path = ?", (path,))
        except Exception:
            pass

    def fts_ready(self) -> bool:
        try:
            with self._fts_connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts' LIMIT 1"
                ).fetchone()
                if not row:
                    return False
                if conn.execute("SELECT 1 FROM chunks_fts LIMIT 1").fetchone():
                    return True
        except Exception:
            return False

        try:
            for _ in self.metadata_db.items():
                return False
        except Exception:
            return False
        return True

    def _get_fts_row_count(self) -> int:
        try:
            with self._fts_connect() as conn:
                self._ensure_fts_table(conn)
                row = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def build_fts_from_metadata(self) -> None:
        try:
            with self._fts_lock:
                with self._fts_connect() as conn:
                    self._ensure_fts_table(conn)
                    conn.execute("DELETE FROM chunks_fts")
                    batch: List[Tuple[str, str, str]] = []
                    for key, entry in self.metadata_db.items():
                        meta = entry.get("metadata") if isinstance(entry, dict) else None
                        if not isinstance(meta, dict):
                            continue
                        chunk_id = entry.get("chunk_id") or meta.get("chunk_id") or str(key)
                        path = meta.get("relative_path") or meta.get("file_path")
                        content = meta.get("content")
                        if not content:
                            content = meta.get("content_preview")
                        if not content:
                            content = meta.get("name")
                        if not content:
                            content = meta.get("chunk_type") or meta.get("type")
                        if not chunk_id or not path or not content:
                            continue
                        batch.append((chunk_id, path, content))
                        if len(batch) >= 500:
                            conn.executemany(
                                "INSERT INTO chunks_fts(chunk_id, path, content) VALUES (?, ?, ?)",
                                batch,
                            )
                            batch.clear()
                    if batch:
                        conn.executemany(
                            "INSERT INTO chunks_fts(chunk_id, path, content) VALUES (?, ?, ?)",
                            batch,
                        )
        except Exception:
            pass

    def ensure_fts_async(self) -> None:
        with self._fts_lock:
            if self._fts_building or self.fts_ready():
                return
            self._fts_building = True

        def _run():
            try:
                self.build_fts_from_metadata()
            finally:
                with self._fts_lock:
                    self._fts_building = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def apply_filters(
        self,
        results: List[Tuple[str, float, Dict[str, Any]]],
        filters: Dict[str, Any],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self._apply_filters(results, filters)

    def _open_sqlitedict(self, path: Path) -> SqliteDict:
        """Open a SqliteDict with basic corruption recovery."""
        def _open():
            return SqliteDict(
                str(path),
                autocommit=False,
                journal_mode="WAL",
                outer_stack=True,
            )

        try:
            self._cleanup_orphaned_wal(path)
            return _open()
        except sqlite3.Error as e:
            message = str(e).lower()
            if "disk i/o" in message or "malformed" in message or "not a database" in message:
                self._logger.warning(f"SQLite error opening {path}: {e}. Backing up and recreating.")
                self._backup_sqlite_files(path)
                return _open()
            raise

    def _metadata_exists(self) -> bool:
        """Return True if any metadata entries exist."""
        try:
            for _ in self.metadata_db.items():
                return True
        except Exception:
            return False
        return False

    def _apply_sanity_warning(self, stats: Dict[str, Any]) -> None:
        """Attach sanity warning when metadata exists but no vectors are indexed."""
        try:
            total_chunks = int(stats.get("total_chunks", 0))
        except Exception:
            total_chunks = 0
        if total_chunks > 0:
            return

        metadata_hint = bool(stats.get("files_indexed") or stats.get("chunk_types") or stats.get("top_tags"))
        metadata_exists = metadata_hint or self._metadata_exists()
        if not metadata_exists:
            return

        stats["sanity_warning"] = (
            "Index has metadata but zero vectors; FAISS index is missing or stale."
        )
        stats["sanity_suggestion"] = (
            "Reindex the project to rebuild the FAISS vectors."
        )

    def _backup_sqlite_files(self, path: Path) -> None:
        """Move sqlite db and wal/shm to a timestamped backup."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            file_path = Path(f"{path}{suffix}")
            if file_path.exists():
                backup = file_path.with_suffix(file_path.suffix + f".corrupt-{timestamp}")
                try:
                    file_path.rename(backup)
                except Exception:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass

    def _cleanup_orphaned_wal(self, path: Path) -> None:
        """Remove orphaned WAL/SHM if base DB is missing."""
        if path.exists():
            return
        for suffix in ("-wal", "-shm"):
            orphan = Path(f"{path}{suffix}")
            if orphan.exists():
                try:
                    orphan.unlink()
                except Exception:
                    pass
    
    def _load_index(self):
        """Load existing FAISS index or create new one."""
        if self.index_path.exists():
            faiss = _get_faiss()
            self._logger.info(f"Loading existing index from {self.index_path}")
            self._index = faiss.read_index(str(self.index_path))
            # If GPU support is available, optionally move to GPU for runtime speed
            self._maybe_move_index_to_gpu()

            if not isinstance(self._index, faiss.IndexIDMap2):
                self._logger.warning(
                    "Loaded legacy FAISS index without ID mapping. "
                    "Reindex is recommended for reliable deletions."
                )

            # Warn if legacy metadata format is detected
            try:
                for key in self.metadata_db.keys():
                    if not str(key).isdigit():
                        self._logger.warning(
                            "Legacy metadata format detected. "
                            "Please reindex to migrate to ID-mapped metadata."
                        )
                        self._build_legacy_index_map()
                        break
            except Exception:
                pass
            
        else:
            self._logger.info("Creating new index")
            # Create a new index - we'll initialize it when we get the first embedding
            self._index = None
            # id maps are stored in sqlite

    def is_legacy_index(self) -> bool:
        """Check whether the current index is legacy (no IDMap2 wrapper)."""
        if self.index is None:
            return False
        faiss = _get_faiss()
        return not isinstance(self.index, faiss.IndexIDMap2)

    def create_index(self, embedding_dimension: int, index_type: str = "flat"):
        """Create a new FAISS index with ID mapping."""
        faiss = _get_faiss()
        if index_type == "flat":
            base_index = faiss.IndexFlatIP(embedding_dimension)
            self._index = faiss.IndexIDMap2(base_index)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dimension)
            n_centroids = min(100, max(10, embedding_dimension // 8))
            base_index = faiss.IndexIVFFlat(quantizer, embedding_dimension, n_centroids)
            self._index = faiss.IndexIDMap2(base_index)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        self._logger.info(f"Created {index_type} index with IDMap2")
        self._maybe_move_index_to_gpu()

    def add_embeddings(self, embedding_results: List[EmbeddingResult], update_stats: bool = True) -> None:
        """Add embeddings to the index."""
        if not embedding_results:
            return

        # Initialize index if needed
        if self._index is None:
            embedding_dim = len(embedding_results[0].embedding)
            self.create_index(embedding_dim, "flat")

        embeddings = np.array([r.embedding for r in embedding_results], dtype=np.float32)
        faiss = _get_faiss()
        faiss.normalize_L2(embeddings)

        ids = np.array([self._get_or_create_int_id(r.chunk_id) for r in embedding_results], dtype=np.int64)

        # Remove existing IDs to avoid duplicates
        try:
            self._index.remove_ids(ids)
        except Exception as e:
            self._logger.warning(f"Failed to remove existing IDs before re-add: {e}")

        # Add to FAISS index with explicit IDs
        self._index.add_with_ids(embeddings, ids)
        
        # Store metadata and update id map / file map
        sample_metas: List[Dict[str, Any]] = []
        fts_entries: List[Tuple[str, str, str]] = []
        for i, result in enumerate(embedding_results):
            int_id = int(ids[i])
            prev_entry = self.metadata_db.get(str(int_id))
            if prev_entry:
                prev_meta = prev_entry.get("metadata") if isinstance(prev_entry, dict) else None
                prev_key = self._normalize_file_key(prev_meta)
                if prev_key:
                    self._remove_ids_from_file_map(prev_key, {int_id})

            self.metadata_db[str(int_id)] = {
                'chunk_id': result.chunk_id,
                'metadata': result.metadata
            }
            path = result.metadata.get("relative_path") or result.metadata.get("file_path")
            content = self._resolve_fts_content(result)
            if path and content:
                fts_entries.append((result.chunk_id, path, content))
            new_key = self._normalize_file_key(result.metadata)
            if new_key:
                self._add_ids_to_file_map(new_key, {int_id})

            meta = result.metadata or {}
            sample_metas.append(
                {
                    "relative_path": meta.get("relative_path"),
                    "file_path": meta.get("file_path"),
                    "chunk_type": meta.get("chunk_type"),
                    "model": result.model_name or meta.get("model"),
                    "tags": meta.get("tags"),
                }
            )
        
        self._logger.info(f"Added {len(embedding_results)} embeddings to index")

        sample_store = self.training_sample
        if sample_store is not None:
            try:
                sample_store.add_batch(embeddings, sample_metas)
            except Exception:
                pass

        # Commit metadata in a single transaction for performance
        try:
            self.metadata_db.commit()
            self.id_map_db.commit()
            self.file_map_db.commit()
        except Exception:
            # If commit is unavailable for some reason, continue without failing
            pass

        self.fts_upsert_many(fts_entries)
        
        # Update statistics
        if update_stats:
            self._update_stats()

    def _gpu_is_available(self) -> bool:
        """Check if GPU FAISS support is available and GPUs are present."""
        faiss = _get_faiss()
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                return False
            get_num_gpus = getattr(faiss, 'get_num_gpus', None)
            if get_num_gpus is None:
                return False
            return get_num_gpus() > 0
        except Exception:
            return False

    def _maybe_move_index_to_gpu(self) -> None:
        """Move the current index to GPU if supported. No-op if already on GPU or unsupported."""
        if self._index is None or self._on_gpu:
            return
        if not self._gpu_is_available():
            return
        try:
            # Move index to all GPUs for faster add/search
            self._index = faiss.index_cpu_to_all_gpus(self._index)
            self._on_gpu = True
            self._logger.info("FAISS index moved to GPU(s)")
        except Exception as e:
            self._logger.warning(f"Failed to move FAISS index to GPU, continuing on CPU: {e}")

    def _stable_int_id(self, chunk_id: str) -> int:
        """Create a stable 63-bit integer ID from a chunk_id."""
        digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        return value & 0x7FFFFFFFFFFFFFFF

    def _build_legacy_index_map(self) -> None:
        """Build index_id -> chunk_id map for legacy indexes."""
        try:
            legacy_map = {}
            for chunk_id, entry in self.metadata_db.items():
                index_id = entry.get("index_id")
                if isinstance(index_id, int):
                    legacy_map[index_id] = chunk_id
            self._legacy_index_map = legacy_map
            self._logger.info(f"Legacy index map built with {len(legacy_map)} entries")
        except Exception as e:
            self._logger.warning(f"Failed to build legacy index map: {e}")

    def _get_or_create_int_id(self, chunk_id: str) -> int:
        """Return a stable integer ID for a given chunk_id, handling collisions."""
        existing = self.id_map_db.get(chunk_id)
        if existing is not None:
            return int(existing)

        candidate = self._stable_int_id(chunk_id)
        salt = 0
        while True:
            existing_entry = self.metadata_db.get(str(int(candidate)))
            if not existing_entry or existing_entry.get("chunk_id") == chunk_id:
                break
            salt += 1
            candidate = self._stable_int_id(f"{chunk_id}#{salt}")

        self.id_map_db[chunk_id] = int(candidate)
        return int(candidate)

    def _lookup_int_id(self, chunk_id: str) -> Optional[int]:
        """Lookup an int ID for a chunk_id."""
        value = self.id_map_db.get(chunk_id)
        if value is None:
            return None
        return int(value)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings."""
        if self.index is None or self.index.ntotal == 0:
            return []

        # Ensure query is normalized for cosine similarity
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss = _get_faiss()
        faiss.normalize_L2(query_embedding)

        # Search the index (widen if filtering)
        search_k = k
        if filters:
            # Arbitrary expansion to give room for filtering
            search_k = min(max(k * 20, k + 50), self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            int_id = int(idx)
            metadata_entry = self.metadata_db.get(str(int_id))
            if not metadata_entry:
                if self._legacy_index_map:
                    legacy_chunk_id = self._legacy_index_map.get(int_id)
                    if legacy_chunk_id:
                        legacy_entry = self.metadata_db.get(legacy_chunk_id)
                        if legacy_entry:
                            results.append((legacy_chunk_id, float(dist), legacy_entry.get("metadata", {})))
                continue

            chunk_id = metadata_entry.get("chunk_id")
            metadata = metadata_entry.get("metadata", {})
            if not chunk_id:
                continue

            results.append((chunk_id, float(dist), metadata))

        # Apply filters if needed
        if filters:
            results = self._apply_filters(results, filters)
            results = results[:k]

        return results

    def _apply_filters(
        self,
        results: List[Tuple[str, float, Dict[str, Any]]],
        filters: Dict[str, Any]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Apply filters to search results with robust glob support."""
        filtered = []
        file_patterns = filters.get('file_pattern')
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
            
        chunk_type = filters.get('chunk_type')
        tags = filters.get('tags')

        for chunk_id, similarity, metadata in results:
            # File pattern filtering (supports globs)
            if file_patterns:
                path = metadata.get('relative_path') or metadata.get('file_path')
                if not path:
                    continue
                
                # Normalize path for matching (standardize separators)
                norm_path = path.replace('\\', '/')
                
                match = False
                for pattern in file_patterns:
                    # Normalize pattern
                    norm_pattern = pattern.replace('\\', '/')
                    # 1. Direct match
                    if fnmatch.fnmatch(norm_path, norm_pattern): 
                        match = True; break
                    # 2. Match as sub-path (unanchored)
                    if not norm_pattern.startswith('/') and not norm_pattern.startswith('./'):
                        if fnmatch.fnmatch(norm_path, "*/" + norm_pattern):
                            match = True; break
                    # 3. Match on filename
                    if fnmatch.fnmatch(Path(norm_path).name, norm_pattern):
                        match = True; break
                    # 4. Fallback: substring match for plain patterns
                    if not any(ch in norm_pattern for ch in "*?[]"):
                        if norm_pattern in norm_path:
                            match = True; break
                if not match:
                    continue
            
            # Chunk type filtering
            if chunk_type and metadata.get('chunk_type') != chunk_type:
                continue
            
            # Tags filtering
            if tags:
                metadata_tags = metadata.get('tags') or []
                if not any(tag in metadata_tags for tag in tags):
                    continue
                    
            filtered.append((chunk_id, similarity, metadata))

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r') as f:
                    stats = json.load(f)
                    stats.update(self._get_index_metadata())
                    stats.update(self._get_training_sample_stats())
                    # Always refresh storage size from disk for accuracy.
                    stats["storage_size"] = self.get_storage_bytes()
                    self._apply_sanity_warning(stats)
                    return stats
            except Exception:
                pass

        stats = {
            'total_chunks': self.index.ntotal if self.index else 0,
            'files_indexed': self._count_indexed_files(),
            'storage_size': self.get_storage_bytes(),
        }
        stats.update(self._get_index_metadata())
        stats.update(self._get_training_sample_stats())
        self._apply_sanity_warning(stats)
        return stats

    def get_storage_bytes(self) -> int:
        """Return total on-disk bytes for index + metadata."""
        total = 0
        sample_paths = [
            self.storage_dir / "training_sample.npy",
            self.storage_dir / "training_sample_meta.json",
            self.storage_dir / "training_sample_stats.json",
        ]
        for path in (
            self.index_path,
            self.metadata_path,
            self.id_map_path,
            self.file_map_path,
            self.stats_path,
            *sample_paths,
        ):
            if path.exists():
                total += path.stat().st_size
        return total

    def get_index_size(self) -> int:
        """Return number of vectors in the index."""
        return self.index.ntotal if self.index else 0

    def _update_stats(self, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Recalculate and save index statistics."""
        current_stats = {}
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r') as f:
                    current_stats = json.load(f)
            except Exception:
                pass

        files_indexed, chunk_types, top_tags = self._collect_metadata_stats()
        stats = {
            'total_chunks': self.index.ntotal if self.index else 0,
            'files_indexed': files_indexed,
            'chunk_types': chunk_types,
            'top_tags': top_tags,
            'storage_size': self.get_storage_bytes(),
            'fts_rows': self._get_fts_row_count(),
            'last_updated': time.time()
        }
        stats.update(self._get_index_metadata())
        stats.update(self._get_training_sample_stats())
        self._apply_sanity_warning(stats)
        if extra_metadata:
            current_stats.update(extra_metadata)
        current_stats.update(stats)

        try:
            with open(self.stats_path, 'w') as f:
                json.dump(current_stats, f, indent=2)
        except Exception as e:
            self._logger.error(f"Failed to update stats: {e}")

    def _get_index_metadata(self) -> Dict[str, Any]:
        """Return metadata about the underlying FAISS index."""
        if self._index is None:
            return {}

        faiss = _get_faiss()
        base = self._index
        try:
            if isinstance(base, faiss.IndexIDMap2):
                base = base.index
        except Exception:
            pass

        meta: Dict[str, Any] = {}
        try:
            meta["index_type"] = self._classify_index_type(base)
        except Exception:
            meta["index_type"] = base.__class__.__name__

        try:
            meta["embedding_dim"] = int(getattr(base, "d"))
        except Exception:
            pass

        metric_type = getattr(base, "metric_type", None)
        if metric_type is not None:
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                meta["metric"] = "ip"
            elif metric_type == faiss.METRIC_L2:
                meta["metric"] = "l2"
            else:
                meta["metric"] = str(metric_type)

        if hasattr(base, "is_trained"):
            meta["trained"] = bool(base.is_trained)
        else:
            meta["trained"] = True

        if hasattr(base, "nlist"):
            try:
                meta["nlist"] = int(base.nlist)
            except Exception:
                pass
        if hasattr(base, "nprobe"):
            try:
                meta["nprobe"] = int(base.nprobe)
            except Exception:
                pass

        return meta

    def _get_training_sample_stats(self) -> Dict[str, Any]:
        stats_path = self.storage_dir / "training_sample_stats.json"
        if not stats_path.exists():
            return {}
        try:
            payload = json.loads(stats_path.read_text())
            return {
                "training_sample_count": int(payload.get("count", 0)),
                "training_sample_total_seen": int(payload.get("total_seen", 0)),
                "training_sample_max": int(payload.get("max_vectors", 0)),
            }
        except Exception:
            return {}

    def _classify_index_type(self, base) -> str:
        if hasattr(base, "nlist"):
            return "ivf"
        try:
            faiss = _get_faiss()
            if isinstance(base, faiss.IndexFlat):
                return "flat"
        except Exception:
            pass
        return base.__class__.__name__

    def _count_indexed_files(self) -> int:
        """Count unique files represented in metadata."""
        try:
            file_paths = set()
            for entry in self.metadata_db.values():
                meta = entry.get("metadata") if isinstance(entry, dict) else None
                if not isinstance(meta, dict):
                    continue
                path = meta.get("relative_path") or meta.get("file_path")
                if path:
                    file_paths.add(path)
            return len(file_paths)
        except Exception:
            return 0

    def _collect_metadata_stats(self) -> tuple[int, Dict[str, int], Dict[str, int]]:
        """Collect files indexed, chunk type counts, and top tags."""
        file_paths = set()
        chunk_types: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}
        try:
            for entry in self.metadata_db.values():
                meta = entry.get("metadata") if isinstance(entry, dict) else None
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
        except Exception:
            return len(file_paths), {}, {}

        top_tags = dict(
            sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        )
        return len(file_paths), chunk_types, top_tags

    def save_index(self, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save index and metadata to disk."""
        if self._index:
            faiss = _get_faiss()
            index_to_save = self._index
            if self._on_gpu:
                index_to_save = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(index_to_save, str(self.index_path))
        
        if self._metadata_db:
            self._metadata_db.commit()
        
        if self._id_map_db:
            self._id_map_db.commit()

        if self._file_map_db:
            self._file_map_db.commit()

        if self._training_sample is not None:
            try:
                self._training_sample.save()
            except Exception:
                pass
            
        self._update_stats(extra_metadata=extra_metadata)

    def clear_index(self) -> None:
        """Completely clear the index and all metadata."""
        self._index = None
        self._on_gpu = False
        self._legacy_index_map = None
        
        if self._metadata_db:
            self._metadata_db.close()
            self._metadata_db = None
            
        if self._id_map_db:
            self._id_map_db.close()
            self._id_map_db = None

        if self._file_map_db:
            self._file_map_db.close()
            self._file_map_db = None

        if self._training_sample is not None:
            try:
                self._training_sample.clear()
            except Exception:
                pass
            self._training_sample = None
            
        extra_paths = [
            self.storage_dir / "training_sample.npy",
            self.storage_dir / "training_sample_meta.json",
            self.storage_dir / "training_sample_stats.json",
        ]
        for p in [self.index_path, self.metadata_path, self.id_map_path, self.file_map_path, self.stats_path, *extra_paths]:
            if p.exists():
                p.unlink()
            # Also cleanup WAL/SHM
            for suffix in ("-wal", "-shm"):
                aux = Path(f"{p}{suffix}")
                if aux.exists():
                    aux.unlink()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Optimized metadata retrieval by chunk_id."""
        int_id = self._lookup_int_id(chunk_id)
        if int_id is None:
            return None
        
        entry = self.metadata_db.get(str(int_id))
        if entry:
            return entry.get("metadata")
        return None

    def remove_file_chunks(self, file_path: str) -> int:
        """Remove all chunks associated with a file path."""
        if not self.index:
            return 0
            
        ids_to_remove = []
        norm_target = Path(file_path).as_posix()
        fts_chunk_ids: List[str] = []
        fts_paths: set[str] = set()

        mapped_ids = self.file_map_db.get(norm_target)
        if mapped_ids:
            ids_to_remove = [int(i) for i in mapped_ids]
        else:
            # Fallback: linear scan of metadata
            for int_id_str, entry in self.metadata_db.items():
                meta = entry.get("metadata", {})
                path = meta.get("relative_path") or meta.get("file_path")
                if not path:
                    continue
                norm_path = Path(path).as_posix()
                if norm_path == norm_target or norm_path.endswith(norm_target) or norm_target.endswith(norm_path):
                    ids_to_remove.append(int(int_id_str))
                
        if not ids_to_remove:
            return 0
            
        try:
            for iid in ids_to_remove:
                entry = self.metadata_db.get(str(iid))
                if not entry:
                    continue
                chunk_id = entry.get("chunk_id")
                if chunk_id:
                    fts_chunk_ids.append(chunk_id)
                meta = entry.get("metadata", {})
                key = self._normalize_file_key(meta)
                if key:
                    fts_paths.add(key)

            self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            for iid in ids_to_remove:
                # Remove from metadata and id_map
                iid_str = str(iid)
                # Find the chunk_id for this int_id to cleanup id_map
                entry = self.metadata_db.get(iid_str)
                if entry:
                    chunk_id = entry.get("chunk_id")
                    if chunk_id:
                        del self.id_map_db[chunk_id]
                    meta = entry.get("metadata", {})
                    key = self._normalize_file_key(meta)
                    if key:
                        self._remove_ids_from_file_map(key, {iid})
                del self.metadata_db[iid_str]

            if mapped_ids:
                self.file_map_db.pop(norm_target, None)

            try:
                self.metadata_db.commit()
                self.id_map_db.commit()
                self.file_map_db.commit()
            except Exception:
                pass

            self._fts_delete_entries(fts_chunk_ids, fts_paths | {norm_target})
                
            return len(ids_to_remove)
        except Exception as e:
            self._logger.error(f"Failed to remove IDs: {e}")
            return 0

    def _fts_delete_entries(self, chunk_ids: List[str], paths: set[str]) -> None:
        if not chunk_ids and not paths:
            return
        try:
            with self._fts_lock:
                with self._fts_connect() as conn:
                    self._ensure_fts_table(conn)
                    if chunk_ids:
                        conn.executemany(
                            "DELETE FROM chunks_fts WHERE chunk_id = ?",
                            [(chunk_id,) for chunk_id in chunk_ids],
                        )
                    if paths:
                        conn.executemany(
                            "DELETE FROM chunks_fts WHERE path = ?",
                            [(path,) for path in sorted(paths)],
                        )
        except Exception:
            pass

    def _normalize_file_key(self, meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(meta, dict):
            return None
        path = meta.get("relative_path") or meta.get("file_path")
        if not path:
            return None
        return Path(path).as_posix()

    def _add_ids_to_file_map(self, key: str, ids: set[int]) -> None:
        existing = self.file_map_db.get(key) or []
        merged = set(int(i) for i in existing)
        merged.update(int(i) for i in ids)
        self.file_map_db[key] = sorted(merged)

    def _remove_ids_from_file_map(self, key: str, ids: set[int]) -> None:
        existing = self.file_map_db.get(key)
        if not existing:
            return
        remaining = [int(i) for i in existing if int(i) not in ids]
        if remaining:
            self.file_map_db[key] = remaining
        else:
            try:
                del self.file_map_db[key]
            except Exception:
                pass
