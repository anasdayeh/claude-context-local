"""Qwen3-Embedding-4B must be registered as a factory that pre-wires the eager-attn
macOS fix, bf16 dtype, and the query instruction — without loading the model.
"""
import torch

from embeddings.embedding_models_register import AVAILABLE_MODELS
from embeddings.sentence_transformer import SentenceTransformerModel


def test_qwen_registered_under_canonical_and_alias():
    assert "Qwen/Qwen3-Embedding-4B" in AVAILABLE_MODELS
    assert "qwen3-embedding-4b" in AVAILABLE_MODELS


def test_qwen_factory_prewires_config_without_loading():
    factory = AVAILABLE_MODELS["qwen3-embedding-4b"]
    m = factory()  # lazy — constructing must NOT download/load the model
    assert isinstance(m, SentenceTransformerModel)
    assert m.model_name == "Qwen/Qwen3-Embedding-4B"
    assert m.trust_remote_code is True
    assert m._extra_model_kwargs.get("attn_implementation") == "eager"
    assert m._extra_model_kwargs.get("torch_dtype") == torch.bfloat16
    assert m.query_instruction and m.query_instruction.startswith("Instruct:")
    assert not m._model_loaded  # still lazy
