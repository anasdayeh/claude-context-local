"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, device: str):
        """Initialize with device resolution."""
        self._device = self._resolve_device(device)

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            **kwargs: Additional model-specific arguments

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass

    def _resolve_device(self, requested: Optional[str]) -> str:
        """Resolve device string."""
        req = (requested or "auto").lower()
        if req in ("auto", "none", ""):
            if torch.cuda.is_available():
                return "cuda"
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        if req.startswith("cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            try:
                return "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
            except Exception:
                return "cpu"
        return "cpu"
