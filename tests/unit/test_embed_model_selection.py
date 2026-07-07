"""The active embedding model is selectable via CODE_SEARCH_EMBED_MODEL (default Gemma).

Switching to Qwen must be an env change (+ a distinct CODE_SEARCH_STORAGE root),
never a code edit.
"""
from mcp_server.code_search_server import _resolve_embed_model_name


def test_default_is_gemma(monkeypatch):
    monkeypatch.delenv("CODE_SEARCH_EMBED_MODEL", raising=False)
    assert _resolve_embed_model_name() == "google/embeddinggemma-300m"


def test_env_overrides_to_qwen(monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_EMBED_MODEL", "qwen3-embedding-4b")
    assert _resolve_embed_model_name() == "qwen3-embedding-4b"


def test_blank_env_falls_back_to_gemma(monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_EMBED_MODEL", "   ")
    assert _resolve_embed_model_name() == "google/embeddinggemma-300m"
