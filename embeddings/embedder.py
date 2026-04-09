"""Main embedding logic for handling code and queries."""

import logging
import gc
import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np

from chunking.code_chunk import CodeChunk
from embeddings.embedding_models_register import AVAILIABLE_MODELS
from common_utils import get_available_memory_bytes

class EmbeddingResult:
    """Result of embedding generation for a chunk."""

    def __init__(
        self,
        chunk: Optional[CodeChunk] = None,
        embedding: Optional[np.ndarray] = None,
        model_name: str = "",
        tokens: int = 0,
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if embedding is None:
            raise ValueError("embedding is required")
        self.chunk = chunk
        self.embedding = embedding
        self.model_name = model_name
        self.tokens = tokens
        self._chunk_id_override = chunk_id
        self._metadata_override = metadata

    @property
    def chunk_id(self) -> str:
        """Generate a stable unique ID for this chunk."""
        if self._chunk_id_override:
            return self._chunk_id_override
        if not self.chunk:
            return ""
        import hashlib
        raw_id = f"{self.chunk.relative_path}:{self.chunk.name}:{self.chunk.start_line}:{self.chunk.chunk_type}"
        return hashlib.md5(raw_id.encode()).hexdigest()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Convert chunk into storage-ready metadata dictionary."""
        if self._metadata_override is not None:
            return self._metadata_override
        if not self.chunk:
            return {}
        content = self.chunk.content or ""
        preview = content
        preview_limit_raw = str(os.getenv("CODE_SEARCH_CONTENT_PREVIEW_CHARS", "320") or "320").strip()
        try:
            preview_limit = max(0, int(preview_limit_raw))
        except Exception:
            preview_limit = 320
        if preview_limit > 0 and len(preview) > preview_limit:
            preview = preview[:preview_limit] + "..."
        metadata = {
            "name": self.chunk.name,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk.chunk_type,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "relative_path": self.chunk.relative_path,
            "file_path": self.chunk.file_path,
            "parent_name": self.chunk.parent_name,
            "tags": self.chunk.tags,
            "content": content,
            "content_preview": preview,
            "folder_structure": self.chunk.folder_structure,
            "model": self.model_name
        }
        if self.chunk.extra_metadata:
            metadata.update(self.chunk.extra_metadata)
        return metadata

class CodeEmbedder:
    """Handles embedding generation for code chunks and search queries using semantic models."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """Initialize code embedder."""
        self._logger = logging.getLogger(__name__)
        self._status = "not_loaded"
        self._last_error: Optional[str] = None

        # Normalize model name if using known aliases
        if model_name in AVAILIABLE_MODELS:
            model_class = AVAILIABLE_MODELS[model_name]
        else:
            model_class = AVAILIABLE_MODELS.get(model_name)
            if not model_class:
                for k, v in AVAILIABLE_MODELS.items():
                    if k.endswith(model_name) or model_name.endswith(k):
                        model_class = v
                        model_name = k
                        break
            
            if not model_class:
                available = sorted(AVAILIABLE_MODELS.keys())
                raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        self.model_name = model_name
        try:
            self._model = model_class(
                model_name,
                device=device,
                cache_dir=cache_dir
            )
        except Exception as e:
            msg = f"Failed to load model '{model_name}': {e}"
            self._logger.error(msg)
            self._status = "failed"
            self._last_error = msg
            raise RuntimeError(msg) from e
        self._min_free_ram_bytes = self._resolve_min_free_ram_bytes()

    @property
    def raw_model(self):
        """Access the underlying SentenceTransformer model for introspection."""
        return getattr(self._model, "model", self._model)

    def _resolve_prompt_name(self, is_query: bool) -> Optional[str]:
        """Resolve generic prompt names to model-specific ones via introspection."""
        prompts = getattr(self.raw_model, "prompts", {})
        if not prompts:
            return None
        if is_query:
            return "query" if "query" in prompts else None
        else:
            return "document" if "document" in prompts else None

    def _encode_documents(self, texts: List[str]) -> np.ndarray:
        """Encode documents using robust wrapper methods."""
        encode_kwargs = {"show_progress_bar": False}
        try:
            if hasattr(self._model, "encode_document"):
                embeddings = self._model.encode_document(texts, **encode_kwargs)
            else:
                prompt_name = self._resolve_prompt_name(is_query=False)
                embeddings = self._model.encode(
                    texts,
                    prompt_name=prompt_name,
                    **encode_kwargs,
                )
            self._status = "ready"
            self._last_error = None
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            self._mark_failure(exc)
            raise

    def _encode_queries(self, texts: List[str]) -> np.ndarray:
        """Encode queries using robust wrapper methods."""
        encode_kwargs = {"show_progress_bar": False}
        try:
            if hasattr(self._model, "encode_query"):
                embeddings = self._model.encode_query(texts, **encode_kwargs)
            else:
                prompt_name = self._resolve_prompt_name(is_query=True)
                embeddings = self._model.encode(
                    texts,
                    prompt_name=prompt_name,
                    **encode_kwargs,
                )
            self._status = "ready"
            self._last_error = None
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            self._mark_failure(exc)
            raise

    def create_embedding_content(self, chunk: CodeChunk, max_chars: int = 2048) -> str:
        """Create formatted content string for embedding."""
        parts = []
        name = chunk.name or "unknown"
        chunk_type = chunk.chunk_type or "unknown"
        
        parts.append(f"Name: {name}")
        parts.append(f"Type: {chunk_type}")
        
        if getattr(chunk, 'parent_name', None):
             parts.append(f"Context: {chunk.parent_name}")

        tags = chunk.tags or []
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        docstring = chunk.docstring or ""
        overhead = sum(len(p) + 1 for p in parts) + 10 
        
        remaining_budget = max_chars - overhead
        docstring_len = len(docstring)
        
        if remaining_budget <= 20:
             return f"Name: {name}\n{(chunk.content or '')[:max_chars//2]}"

        docstring_budget = min(docstring_len, int(remaining_budget * 0.3))
        if docstring_len < remaining_budget * 0.5:
             docstring_budget = docstring_len
             
        if docstring:
             parts.append(f"Docstring: {docstring[:docstring_budget]}")
             
        current_used = sum(len(p) + 1 for p in parts)
        code_budget = max(0, max_chars - current_used)
        
        content = chunk.content or ""
        header = "\n".join(parts)
        
        if not content:
            return header
            
        full_text = f"{header}\n{content}"
        
        if len(full_text) > max_chars:
            allowed_content_len = max(0, max_chars - len(header) - 1)
            return f"{header}\n{content[:allowed_content_len]}"
            
        return full_text

    def embed_chunks(self, chunks: List[CodeChunk], batch_size: int = 32) -> List[EmbeddingResult]:
        """Generate embeddings for code chunks in batches."""
        if not chunks:
            return []

        if not batch_size or batch_size <= 0:
            batch_size = 32

        texts = [self.create_embedding_content(chunk) for chunk in chunks]
        results = []
        total = len(texts)
        
        self._logger.info(f"Generating embeddings for {total} chunks (batch_size={batch_size})")

        i = 0
        adaptive_batch = batch_size
        while i < total:
            current_batch = min(adaptive_batch, total - i)
            batch_texts = texts[i : i + current_batch]
            batch_chunks = chunks[i : i + current_batch]

            while True:
                if self._is_memory_pressure():
                    if current_batch > 1:
                        next_batch = max(1, current_batch // 2)
                        self._notify_progress(
                            f"Memory-pressure backoff: batch {current_batch}->{next_batch}"
                        )
                        self._logger.warning(
                            "Low available RAM detected; reducing batch size from %d to %d",
                            current_batch,
                            next_batch,
                        )
                        self._clear_device_cache()
                        adaptive_batch = next_batch
                        current_batch = min(adaptive_batch, total - i)
                        batch_texts = texts[i : i + current_batch]
                        batch_chunks = chunks[i : i + current_batch]
                        continue
                    self._clear_device_cache()

                try:
                    self._logger.info(
                        f"Loop {i}: text_len={len(batch_texts)} chunk_len={len(batch_chunks)}"
                    )
                    batch_embeddings = self._encode_documents(batch_texts)
                    self._logger.info(f"Loop {i}: embed_len={len(batch_embeddings)}")
                    break
                except Exception as e:
                    if not self._is_oom_error(e):
                        self._logger.error(f"Batch encoding failed at index {i}: {e}")
                        raise

                    if current_batch > 1:
                        next_batch = max(1, current_batch // 2)
                        self._notify_progress(
                            f"OOM backoff: batch {current_batch}->{next_batch}"
                        )
                        self._logger.warning(
                            "OOM during embedding; reducing batch size from %d to %d",
                            current_batch,
                            next_batch,
                        )
                        self._clear_device_cache()
                        adaptive_batch = next_batch
                        current_batch = min(adaptive_batch, total - i)
                        batch_texts = texts[i : i + current_batch]
                        batch_chunks = chunks[i : i + current_batch]
                        continue

                    if self._force_cpu_fallback():
                        self._notify_progress("OOM: retrying on CPU fallback")
                        self._logger.warning(
                            "OOM at batch_size=1; retrying on CPU fallback"
                        )
                        self._clear_device_cache()
                        continue

                    self._logger.error(f"Batch encoding failed at index {i}: {e}")
                    raise

            batch_results = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                batch_results.append(EmbeddingResult(
                    chunk=chunk,
                    embedding=embedding,
                    model_name=self.model_name
                ))
            self._logger.info(f"Loop {i}: zipped_results={len(batch_results)}")
            results.extend(batch_results)
            i += current_batch

        self._logger.info(f"Embedding generation completed. Results: {len(results)}")
        return results

    def _notify_progress(self, message: str) -> None:
        callback = getattr(self, "_progress_callback", None)
        if not callback:
            return
        try:
            callback(message)
        except Exception:
            pass

    def _is_oom_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "out of memory",
                "oom",
                "mps backend out of memory",
                "cuda out of memory",
            )
        )

    def _clear_device_cache(self) -> None:
        gc.collect()
        try:
            import torch
        except Exception:
            return

        try:
            torch_cuda = getattr(torch, "cuda", None)
            cuda_is_available = getattr(torch_cuda, "is_available", lambda: False)()
            if cuda_is_available:
                try:
                    torch_cuda.empty_cache()
                except Exception:
                    pass
            else:
                mps_backend = getattr(torch.backends, "mps", None)
                if (
                    mps_backend is not None
                    and getattr(mps_backend, "is_available", lambda: False)()
                    and getattr(mps_backend, "is_built", lambda: False)()
                ):
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
        except Exception:
            pass

    def _force_cpu_fallback(self) -> bool:
        model = getattr(self, "_model", None)
        if model is None:
            return False

        changed = False
        if hasattr(model, "_device"):
            if getattr(model, "_device", None) != "cpu":
                model._device = "cpu"
                changed = True
        if hasattr(model, "_fallback_attempted"):
            model._fallback_attempted = True
        if hasattr(model, "_reset_model"):
            try:
                model._reset_model()
                changed = True
            except Exception:
                return False
        return changed

    def _resolve_min_free_ram_bytes(self) -> int:
        raw = ""
        try:
            import os
            raw = str(os.getenv("CODE_SEARCH_MIN_FREE_RAM_GB", "") or "").strip()
        except Exception:
            raw = ""
        if raw:
            try:
                value = float(raw)
                if value > 0:
                    return int(value * 1024 ** 3)
            except Exception:
                pass
        # Default to a conservative floor if unset.
        return int(1.5 * 1024 ** 3)

    def _is_memory_pressure(self) -> bool:
        threshold = getattr(self, "_min_free_ram_bytes", None)
        if not isinstance(threshold, int) or threshold <= 0:
            threshold = self._resolve_min_free_ram_bytes()
            self._min_free_ram_bytes = threshold
        available = get_available_memory_bytes()
        if available <= 0:
            return False
        return available < threshold

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode_queries([query])[0]

    def embed_document(self, text: str) -> np.ndarray:
        return self._encode_documents([text])[0]

    def get_model_info(self) -> Dict[str, Any]:
        info = {}
        try:
            info = self._model.get_model_info()
        except Exception as exc:
            self._mark_failure(exc)
            info = {"status": "failed", "error": str(exc)}

        if not isinstance(info, dict):
            info = {"status": self._status}

        status = info.get("status") or self._status
        if status == "loaded":
            status = "ready"
        info["status"] = status
        info.setdefault("model_name", self.model_name)
        info.setdefault("error", self._last_error)
        return info

    def health_status(self) -> Dict[str, Any]:
        info = self.get_model_info()
        return {
            "status": info.get("status", self._status),
            "backend": info.get("backend"),
            "device": info.get("device"),
            "error": info.get("error"),
            "model_name": info.get("model_name", self.model_name),
        }

    def is_available(self) -> bool:
        return self.health_status().get("status") == "ready"

    def warmup(self, probe: str = "healthcheck") -> bool:
        try:
            self.embed_query(probe)
            return True
        except Exception:
            return False

    def cleanup(self):
        if hasattr(self, '_model'):
            self._model.cleanup()

    def _mark_failure(self, exc: Exception) -> None:
        self._status = "failed"
        self._last_error = str(exc)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
