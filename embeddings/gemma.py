"""EmbeddingGemma model implementation."""

from typing import Optional
from embeddings.sentence_transformer import SentenceTransformerModel


class GemmaEmbeddingModel(SentenceTransformerModel):
    """EmbeddingGemma model - specialized SentenceTransformer implementation."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        """Initialize GemmaEmbeddingModel.

        Args:
            cache_dir: Directory to cache the model
            device: Device to load model on ("auto", "cuda", "mps", "cpu")
        """
        super().__init__(
            model_name="google/embeddinggemma-300m",
            cache_dir=cache_dir,
            device=device
        )
