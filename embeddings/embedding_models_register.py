"""Embedding models registry."""
from embeddings.gemma import GemmaEmbeddingModel
from embeddings.sentence_transformer import SentenceTransformerModel

AVAILIABLE_MODELS = {
    "google/embeddinggemma-300m": GemmaEmbeddingModel,
    "embeddinggemma-300m": GemmaEmbeddingModel, # Alias
    "all-MiniLM-L6-v2": SentenceTransformerModel, # Useful default/fallback
}
