"""SentenceTransformer model implementation."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.embedding_model import EmbeddingModel


class SentenceTransformerModel(EmbeddingModel):
    """SentenceTransformer wrapper with robustness features."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
        backend: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        query_instruction: Optional[str] = None,
    ):
        """Initialize SentenceTransformer model.

        model_kwargs: extra kwargs forwarded to SentenceTransformer(model_kwargs=...)
            on every load path (primary + torch fallback) — e.g.
            attn_implementation="eager" (Qwen macOS SDPA NaN fix) and torch_dtype.
        query_instruction: if set, prepended to each query text in encode_query
            (Qwen instruction-aware retrieval); documents are never prefixed.
        """
        super().__init__(device)
        self._requested_device = self._device
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self._extra_model_kwargs = dict(model_kwargs or {})
        self.query_instruction = query_instruction
        self._logger = logging.getLogger(__name__)
        
        # State tracking
        env_backend = str(os.environ.get("CODE_SEARCH_EMBED_BACKEND", "") or "").strip().lower()
        requested_backend = (backend or env_backend or "torch").strip().lower()
        if requested_backend not in {"torch", "onnx"}:
            requested_backend = "torch"
        self._requested_backend = requested_backend
        self.backend = requested_backend
        self._model_loaded = False
        self._fallback_attempted = False
        self._load_error: Optional[str] = None
        self._fallback_events: List[Dict[str, str]] = []

    def _effective_device_for_backend(self, backend: str) -> str:
        if backend == "onnx" and self._device == "mps":
            # ONNX Runtime on Apple Silicon typically runs via CPU/CoreML providers.
            return "cpu"
        return self._device

    def _apply_torch_thread_limits(self) -> None:
        if self.backend != "torch":
            return
        raw_threads = str(os.environ.get("CODE_SEARCH_TORCH_NUM_THREADS", "") or "").strip()
        raw_interop = str(os.environ.get("CODE_SEARCH_TORCH_INTEROP_THREADS", "") or "").strip()
        try:
            if raw_threads:
                threads = max(1, int(raw_threads))
                torch.set_num_threads(threads)
        except Exception:
            pass
        try:
            if raw_interop:
                interop = max(1, int(raw_interop))
                torch.set_num_interop_threads(interop)
        except Exception:
            pass

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if not self._model_loaded:
            try:
                self.__dict__["model"] = self._load_model()
                self._model_loaded = True
                self._load_error = None
            except Exception as exc:
                self._load_error = str(exc)
                raise
        return self.__dict__["model"]

    def _load_model(self) -> SentenceTransformer:
        """Load the model with backend fallback logic."""
        model_source = self.model_name
        if self._is_model_cached():
            local_path = self._find_local_model_dir()
            if local_path:
                model_source = str(local_path)
        
        backend = self.backend
        # Seed with per-model kwargs (e.g. attn_implementation="eager", torch_dtype)
        # so they reach BOTH the primary load below and the torch fallback.
        model_kwargs = dict(self._extra_model_kwargs)

        if backend == "onnx":
            # Configure ONNX kwargs based on env/defaults
            # This matches sbert.net behavior for backend="onnx"
            provider = os.environ.get("ST_ONNX_PROVIDER")
            file_name = os.environ.get("ST_ONNX_FILE_NAME")
            export_flag = os.environ.get("ST_ONNX_EXPORT")
            if provider:
                model_kwargs["provider"] = provider
            if file_name:
                model_kwargs["file_name"] = file_name
            if export_flag is not None:
                model_kwargs["export"] = export_flag.lower() in {"1", "true", "yes"}

        try:
            self._apply_torch_thread_limits()
            model = SentenceTransformer(
                model_source,
                cache_folder=self.cache_dir,
                device=self._effective_device_for_backend(backend),
                trust_remote_code=self.trust_remote_code,
                backend=backend,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            return self._maybe_quantize_onnx(model)
        except Exception as e:
            if backend == "onnx":
                self._logger.warning(
                    f"Failed to load ONNX backend, falling back to PyTorch: {e}"
                )
                self.backend = "torch" # Update state
                self._record_fallback("onnx", "torch", "onnx_error")
                self._apply_torch_thread_limits()
                return SentenceTransformer(
                    model_source,
                    cache_folder=self.cache_dir,
                    device=self._effective_device_for_backend("torch"),
                    trust_remote_code=self.trust_remote_code,
                    model_kwargs=model_kwargs if model_kwargs else None,
                )
            raise

    def _maybe_quantize_onnx(self, model: SentenceTransformer) -> SentenceTransformer:
        """Optionally export and load a quantized ONNX model."""
        if self.backend != "onnx":
            return model

        quantize_flag = os.environ.get("ST_ONNX_QUANTIZE", "").lower() in {"1", "true", "yes"}
        if not quantize_flag:
            return model

        try:
            from sentence_transformers.backend import export_dynamic_quantized_onnx_model
        except Exception as e:
            self._logger.warning(f"ONNX quantization not available: {e}")
            return model

        quant_config = os.environ.get("ST_ONNX_QUANT_CONFIG", "arm64")
        file_suffix = os.environ.get("ST_ONNX_QUANT_SUFFIX")

        cache_root = Path(self.cache_dir) if self.cache_dir else Path.cwd()
        model_key = self.model_name.replace("/", "__")
        quant_dir = cache_root / "onnx_quantized" / model_key
        quant_dir.mkdir(parents=True, exist_ok=True)

        onnx_file = self._find_onnx_file(quant_dir, file_suffix)
        if onnx_file is None:
            try:
                export_dynamic_quantized_onnx_model(
                    model,
                    quantization_config=quant_config,
                    model_name_or_path=str(quant_dir),
                    file_suffix=file_suffix,
                )
            except Exception as e:
                self._logger.warning(f"Failed to export quantized ONNX model: {e}")
                return model
            onnx_file = self._find_onnx_file(quant_dir, file_suffix)

        if onnx_file is None:
            self._logger.warning("Quantized ONNX export did not produce a model file")
            return model

        model_kwargs = {
            "file_name": onnx_file,
            "export": False,
        }

        return SentenceTransformer(
            str(quant_dir),
            cache_folder=self.cache_dir,
            device=self._device,
            trust_remote_code=self.trust_remote_code,
            backend="onnx",
            model_kwargs=model_kwargs,
        )

    def _find_onnx_file(self, quant_dir: Path, file_suffix: Optional[str]) -> Optional[str]:
        """Find an ONNX file in a directory, preferring a known suffix."""
        candidates = []
        onnx_root = quant_dir / "onnx"
        search_dirs = [onnx_root, quant_dir] if onnx_root.exists() else [quant_dir]
        for search_dir in search_dirs:
            candidates.extend(sorted(search_dir.glob("*.onnx")))

        if not candidates:
            return None

        if file_suffix:
            for candidate in candidates:
                if file_suffix in candidate.stem:
                    return str(candidate.relative_to(quant_dir))

        # Default to the first ONNX file
        return str(candidates[0].relative_to(quant_dir))

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode texts using SentenceTransformer with fallback logic."""
        try:
            return self.model.encode(texts, **kwargs)
        except Exception as e:
            if self._fallback_attempted:
                raise

            fallback_reason = str(e)
            self._logger.warning(f"Encode failed ({fallback_reason}). Attempting fallback.")

            # First fallback: ONNX -> torch
            if self.backend == "onnx":
                previous_backend = self.backend
                self.backend = "torch"
                self._record_fallback(previous_backend, "torch", "onnx_error")
                self._fallback_attempted = True
                self._reset_model()
                try:
                    result = self.model.encode(texts, **kwargs)
                    return result
                except Exception:
                    pass

            # Second fallback: MPS -> CPU
            if self._device == "mps":
                reason = "mps_oom" if self._is_oom_error(e) else "mps_error"
                self._device = "cpu"
                self._fallback_attempted = True
                self._reset_model()
                result = self.model.encode(texts, **kwargs)
                self._record_fallback("mps", "cpu", reason)
                return result

            raise

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "oom" in message

    def _record_fallback(self, source: str, target: str, reason: str) -> None:
        event = {"from": source, "to": target, "reason": reason}
        if event not in self._fallback_events:
            self._fallback_events.append(event)

    def get_execution_info(self) -> Dict[str, Any]:
        """Return truthful execution provenance for callers and benchmark artifacts."""
        actual_device = self._device
        if self._model_loaded and "model" in self.__dict__:
            actual_device = str(getattr(self.__dict__["model"], "device", actual_device))
        events = [dict(event) for event in self._fallback_events]
        return {
            "requested_device": self._requested_device,
            "actual_device": actual_device,
            "requested_backend": getattr(self, "_requested_backend", self.backend),
            "actual_backend": self.backend,
            "fallback_events": events,
            "degraded": bool(events),
        }

    def encode_query(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode queries using model-specific method if available."""
        if self.query_instruction:
            # Instruction-aware model (e.g. Qwen) with no usable "query" prompt:
            # prepend the instruction ourselves and encode as plain text, bypassing
            # encode_query's prompt handling. Documents (encode_document) get nothing.
            prefixed = [f"{self.query_instruction}{t}" for t in texts]
            return self.encode(prefixed, **kwargs)
        try:
            m = self.model
            if hasattr(m, "encode_query"):
                return m.encode_query(texts, **kwargs)
            # Fallback for models that support task prompt
            return m.encode(texts, prompt_name="query", **kwargs)
        except Exception:
            # Re-route through robust encode() which includes fallback logic
            # Use task arg which some models support
            return self.encode(texts, **kwargs)

    def encode_document(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode documents using model-specific method if available."""
        try:
            m = self.model
            if hasattr(m, "encode_document"):
                return m.encode_document(texts, **kwargs)
            # Fallback for models that support task prompt
            return m.encode(texts, prompt_name="document", **kwargs)
        except Exception:
            # Re-route through robust encode()
            return self.encode(texts, **kwargs)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self._model_loaded:
            info = {
                "status": "failed" if self._load_error else "not_loaded",
                "model_name": self.model_name,
                "device": self._effective_device_for_backend(self.backend),
                "backend": self.backend,
                "error": self._load_error,
            }
            info.update(self.get_execution_info())
            return info

        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "device": str(self.model.device),
            "backend": self.backend,
            "status": "loaded",
            "error": None,
        }
        info.update(self.get_execution_info())
        return info

    def cleanup(self):
        """Clean up model resources."""
        if not self._model_loaded:
            return

        try:
            # Drop from memory
            if "model" in self.__dict__:
                model = self.__dict__["model"]
                if hasattr(model, 'to'):
                    try:
                        model.to('cpu')
                    except Exception:
                        pass
            
            # Clear caches (guard for interpreter shutdown states)
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

            self._reset_model()
            self._logger.info("Model cleaned up and memory freed")
        except Exception as e:
            self._logger.warning(f"Error during model cleanup: {e}")

    def _reset_model(self) -> None:
        """Clear cached model so it can be reloaded with new settings."""
        self.__dict__.pop("model", None)
        self._model_loaded = False

    def _is_model_cached(self) -> bool:
        """Check if model is cached locally."""
        if not self.cache_dir:
            return False
        try:
            model_key = self.model_name.split('/')[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return False
            for path in cache_root.rglob('config_sentence_transformers.json'):
                parent_str = str(path.parent).lower()
                if model_key in parent_str:
                    return True
            for d in cache_root.glob('**/*'):
                if d.is_dir() and model_key in d.name.lower():
                    if (d / 'config_sentence_transformers.json').exists() or (d / 'README.md').exists():
                        return True
        except Exception:
            return False
        return False

    def _find_local_model_dir(self) -> Optional[str]:
        """Locate the cached model directory."""
        if not self.cache_dir:
            return None
        try:
            model_key = self.model_name.split('/')[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return None
            for path in cache_root.rglob('config_sentence_transformers.json'):
                parent = path.parent
                if model_key in str(parent).lower():
                    return parent
            candidates = [d for d in cache_root.glob('**/*') if d.is_dir() and model_key in d.name.lower()]
            return candidates[0] if candidates else None
        except Exception:
            return None
