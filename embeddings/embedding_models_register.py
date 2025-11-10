"""Embedding models registry."""
from embeddings.gemma import GemmaEmbeddingModel

AVAILIABLE_MODELS = {
    "google/embeddinggemma-300m": GemmaEmbeddingModel,
}
