"""SentenceTransformer model implementation."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
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
    ):
        """Initialize SentenceTransformer model."""
        super().__init__(device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self._logger = logging.getLogger(__name__)
        
        # State tracking
        self.backend = "torch"
        self._model_loaded = False
        self._fallback_attempted = False

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if not self._model_loaded:
            self.__dict__["model"] = self._load_model()
            self._model_loaded = True
        return self.__dict__["model"]

    def _load_model(self) -> SentenceTransformer:
        """Load the model with backend fallback logic."""
        model_source = self.model_name
        if self._is_model_cached():
            local_path = self._find_local_model_dir()
            if local_path:
                model_source = str(local_path)
        
        backend = self.backend
        model_kwargs = {}
        
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
            return SentenceTransformer(
                model_source,
                cache_folder=self.cache_dir,
                device=self._device,
                trust_remote_code=self.trust_remote_code,
                backend=backend,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
        except Exception as e:
            if backend == "onnx":
                self._logger.warning(
                    f"Failed to load ONNX backend, falling back to PyTorch: {e}"
                )
                self.backend = "torch" # Update state
                return SentenceTransformer(
                    model_source,
                    cache_folder=self.cache_dir,
                    device=self._device,
                    trust_remote_code=self.trust_remote_code,
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
                self.backend = "torch"
                self._fallback_attempted = True
                self._reset_model()
                try:
                    return self.model.encode(texts, **kwargs)
                except Exception:
                    pass

            # Second fallback: MPS -> CPU
            if self._device == "mps":
                self._device = "cpu"
                self._fallback_attempted = True
                self._reset_model()
                return self.model.encode(texts, **kwargs)

            raise

    def encode_query(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode queries using model-specific method if available."""
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
            return {"status": "not_loaded"}

        return {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "device": str(self.model.device),
            "status": "loaded"
        }

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
            
            # Clear caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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
