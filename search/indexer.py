"""Vector index management with FAISS and metadata storage."""

import os
import json
import logging
import hashlib
import sqlite3
import time
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sqlitedict import SqliteDict
from embeddings.embedder import EmbeddingResult
from chunking.code_chunk import CodeChunk

# Reduce OpenMP/BLAS thread contention for stability.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass


class CodeIndexManager:
    """Manages FAISS vector index and metadata storage for code chunks."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.storage_dir / "code.index"
        self.metadata_path = self.storage_dir / "metadata.db"
        self.id_map_path = self.storage_dir / "id_map.db"
        self.stats_path = self.storage_dir / "stats.json"
        
        # Initialize components
        self._index = None
        self._metadata_db = None
        self._id_map_db = None
        self._logger = logging.getLogger(__name__)
        self._on_gpu = False
        self._legacy_index_map = None
        
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
        return not isinstance(self.index, faiss.IndexIDMap2)

    def create_index(self, embedding_dimension: int, index_type: str = "flat"):
        """Create a new FAISS index with ID mapping."""
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
        faiss.normalize_L2(embeddings)

        ids = np.array([self._get_or_create_int_id(r.chunk_id) for r in embedding_results], dtype=np.int64)

        # Remove existing IDs to avoid duplicates
        try:
            self._index.remove_ids(ids)
        except Exception as e:
            self._logger.warning(f"Failed to remove existing IDs before re-add: {e}")

        # Add to FAISS index with explicit IDs
        self._index.add_with_ids(embeddings, ids)
        
        # Store metadata and update id map
        for i, result in enumerate(embedding_results):
            int_id = int(ids[i])
            self.metadata_db[str(int_id)] = {
                'chunk_id': result.chunk_id,
                'metadata': result.metadata
            }
        
        self._logger.info(f"Added {len(embedding_results)} embeddings to index")
        
        # Commit metadata in a single transaction for performance
        try:
            self.metadata_db.commit()
            self.id_map_db.commit()
        except Exception:
            # If commit is unavailable for some reason, continue without failing
            pass
        
        # Update statistics
        if update_stats:
            self._update_stats()

    def _gpu_is_available(self) -> bool:
        """Check if GPU FAISS support is available and GPUs are present."""
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
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'total_chunks': self.index.ntotal if self.index else 0,
            'storage_size': self.index_path.stat().st_size if self.index_path.exists() else 0
        }

    def _update_stats(self) -> None:
        """Recalculate and save index statistics."""
        stats = {
            'total_chunks': self.index.ntotal if self.index else 0,
            'last_updated': time.time()
        }
        try:
            with open(self.stats_path, 'w') as f:
                json.dump(stats, f)
        except Exception as e:
            self._logger.error(f"Failed to update stats: {e}")

    def save_index(self) -> None:
        """Save index and metadata to disk."""
        if self._index:
            # Ensure we save the CPU version if it's on GPU
            index_to_save = self._index
            if self._on_gpu:
                index_to_save = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(index_to_save, str(self.index_path))
        
        if self._metadata_db:
            self._metadata_db.commit()
        
        if self._id_map_db:
            self._id_map_db.commit()
            
        self._update_stats()

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
            
        for p in [self.index_path, self.metadata_path, self.id_map_path, self.stats_path]:
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
        # This is a linear scan of metadata - okay for small/medium repos.
        # For very large ones, we'd need a secondary index (file_path -> ids).
        for int_id_str, entry in self.metadata_db.items():
            meta = entry.get("metadata", {})
            path = meta.get("relative_path") or meta.get("file_path")
            if path == file_path:
                ids_to_remove.append(int(int_id_str))
                
        if not ids_to_remove:
            return 0
            
        try:
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
                del self.metadata_db[iid_str]
                
            return len(ids_to_remove)
        except Exception as e:
            self._logger.error(f"Failed to remove IDs: {e}")
            return 0
