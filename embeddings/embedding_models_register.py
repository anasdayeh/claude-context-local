"""Embedding models registry."""
import torch

from embeddings.sentence_transformer import SentenceTransformerModel


def _gemma_factory(model_name=None, **kwargs):
    """Factory for EmbeddingGemma — always uses the correct model name and trust_remote_code."""
    return SentenceTransformerModel(
        model_name="google/embeddinggemma-300m",
        trust_remote_code=True,
        **kwargs,
    )


# Qwen3-Embedding-4B is instruction-aware on the query side. Documents get NO prefix.
_QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a code search query, retrieve relevant code and "
    "documentation passages that answer the query\nQuery:"
)


def _qwen_factory(model_name=None, **kwargs):
    """Factory for Qwen3-Embedding-4B.

    Pre-wires: eager attention (macOS SDPA NaN fix, sentence-transformers#3498),
    bf16 dtype (keeps the ~8GB model within a 16GB M1's budget), and the query
    instruction. Full-precision safetensors via the existing torch path — no MLX.
    """
    return SentenceTransformerModel(
        model_name="Qwen/Qwen3-Embedding-4B",
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.bfloat16},
        query_instruction=_QWEN_QUERY_INSTRUCTION,
        **kwargs,
    )


AVAILABLE_MODELS = {
    "google/embeddinggemma-300m": _gemma_factory,
    "embeddinggemma-300m": _gemma_factory,  # Alias
    "Qwen/Qwen3-Embedding-4B": _qwen_factory,
    "qwen3-embedding-4b": _qwen_factory,  # Alias
    "all-MiniLM-L6-v2": SentenceTransformerModel,  # Useful default/fallback
}
