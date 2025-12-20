"""SentenceTransformer embedding model implementation."""

from typing import Optional, Dict, Any
from pathlib import Path
from functools import cached_property
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from embeddings.embedding_model import EmbeddingModel


class SentenceTransformerModel(EmbeddingModel):
    """SentenceTransformer embedding model with caching and device management."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
        backend: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize SentenceTransformerModel.

        Args:
            model_name: Name of the model to load
            cache_dir: Directory to cache the model
            device: Device to load model on
        """
        super().__init__(device=device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.backend = (backend or os.environ.get("ST_BACKEND", "torch")).lower()
        self.model_kwargs = model_kwargs or {}
        self._model_loaded = False
        self._fallback_attempted = False
        self._logger = logging.getLogger(__name__)

    @cached_property
    def model(self):
        """Load and cache the SentenceTransformer model."""
        self._logger.info(f"Loading model: {self.model_name}")

        # If the model appears to be cached locally, enable offline mode
        local_model_dir = None
        try:
            if self._is_model_cached():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self._logger.info("Model cache detected. Enabling offline mode for faster startup.")
                local_model_dir = self._find_local_model_dir()
                if local_model_dir:
                    self._logger.info(f"Loading model from local cache path: {local_model_dir}")
        except Exception as e:
            self._logger.debug(f"Offline mode detection skipped: {e}")

        try:
            model_source = str(local_model_dir) if local_model_dir else self.model_name
            model = self._load_model(model_source)
            self._logger.info(f"Model loaded successfully on device: {model.device}")
            self._model_loaded = True
            return self._maybe_quantize_onnx(model)
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            raise

    def _load_model(self, model_source: str) -> SentenceTransformer:
        """Load model with backend-specific options."""
        backend = self.backend
        model_kwargs = dict(self.model_kwargs)

        if backend == "onnx":
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
        """Encode texts using SentenceTransformer.

        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments passed to SentenceTransformer.encode()

        Returns:
            Array of embeddings
        """
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
            model = self.model
            model.to('cpu')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del model
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
