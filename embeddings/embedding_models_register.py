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


# BAAI/bge-code-v1 is a code-specialist embedder (last-token pooling). Queries get
# the "<instruct>{task}\n<query>" prefix (verified on the model card); documents get
# none. trust_remote_code is required for its custom model/pooling code.
_BGE_CODE_QUERY_INSTRUCTION = (
    "<instruct>Given a code search query, retrieve relevant code and "
    "documentation that answer the query\n<query>"
)


def _bge_code_factory(model_name=None, **kwargs):
    """Factory for BAAI/bge-code-v1."""
    return SentenceTransformerModel(
        model_name="BAAI/bge-code-v1",
        trust_remote_code=True,
        query_instruction=_BGE_CODE_QUERY_INSTRUCTION,
        **kwargs,
    )


AVAILABLE_MODELS = {
    "google/embeddinggemma-300m": _gemma_factory,
    "embeddinggemma-300m": _gemma_factory,  # Alias
    "Qwen/Qwen3-Embedding-4B": _qwen_factory,
    "qwen3-embedding-4b": _qwen_factory,  # Alias
    "BAAI/bge-code-v1": _bge_code_factory,
    "bge-code-v1": _bge_code_factory,  # Alias
    "all-MiniLM-L6-v2": SentenceTransformerModel,  # Useful default/fallback
}
