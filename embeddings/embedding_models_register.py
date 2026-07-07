"""Embedding models registry."""
from embeddings.sentence_transformer import SentenceTransformerModel


def _gemma_factory(model_name=None, **kwargs):
    """Factory for EmbeddingGemma — always uses the correct model name and trust_remote_code."""
    return SentenceTransformerModel(
        model_name="google/embeddinggemma-300m",
        trust_remote_code=True,
        **kwargs,
    )


AVAILABLE_MODELS = {
    "google/embeddinggemma-300m": _gemma_factory,
    "embeddinggemma-300m": _gemma_factory,  # Alias
    "all-MiniLM-L6-v2": SentenceTransformerModel,  # Useful default/fallback
}
