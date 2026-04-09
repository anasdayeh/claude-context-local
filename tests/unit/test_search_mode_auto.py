import pytest

from search.searcher import IntelligentSearcher


class DummyIndexManager:
    def __init__(self):
        self.metadata_db = {}

    def fts_search(self, _query: str, k: int = 5):
        return [("chunk-1", 0.0)][:k]

    def get_chunk_by_id(self, chunk_id: str):
        return {
            "chunk_id": chunk_id,
            "relative_path": "src/example.py",
            "file_path": "src/example.py",
            "content_preview": "def example(): pass",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 1,
            "folder_structure": ["src"],
            "tags": [],
        }

    def apply_filters(self, results, _filters):
        return results


class DummyEmbedder:
    def embed_query(self, query: str):
        return query


def test_auto_mode_uses_hybrid_when_available(monkeypatch):
    searcher = IntelligentSearcher(DummyIndexManager(), DummyEmbedder())
    monkeypatch.setattr(searcher, "_fts_ready", lambda: True)
    monkeypatch.setattr(searcher, "_hybrid_search", lambda *args, **kwargs: ["hybrid"])
    monkeypatch.setattr(searcher, "_semantic_search", lambda *args, **kwargs: ["semantic"])

    assert searcher.search("q", search_mode="auto") == ["hybrid"]


def test_hybrid_falls_back_and_triggers_build(monkeypatch):
    searcher = IntelligentSearcher(DummyIndexManager(), DummyEmbedder())
    triggered = {"called": False}

    monkeypatch.setattr(searcher, "_fts_ready", lambda: False)
    monkeypatch.setattr(searcher, "_ensure_fts_async", lambda: triggered.__setitem__("called", True))
    monkeypatch.setattr(searcher, "_hybrid_search", lambda *args, **kwargs: ["hybrid"])
    monkeypatch.setattr(searcher, "_semantic_search", lambda *args, **kwargs: ["semantic"])

    assert searcher.search("q", search_mode="hybrid") == ["semantic"]
    assert triggered["called"] is True


def test_auto_mode_falls_back_to_fts_when_embeddings_fail(monkeypatch):
    searcher = IntelligentSearcher(DummyIndexManager(), DummyEmbedder())

    monkeypatch.setattr(searcher, "_fts_ready", lambda: True)
    monkeypatch.setattr(searcher, "_hybrid_search", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedder failed")))
    monkeypatch.setattr(searcher, "_build_file_context_cache", lambda *_args, **_kwargs: {})

    results = searcher.search("q", search_mode="auto")

    assert searcher.last_search_mode_used == "fts"
    assert searcher.last_error_summary == "embedder failed"
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


def test_explicit_semantic_does_not_silently_fallback(monkeypatch):
    searcher = IntelligentSearcher(DummyIndexManager(), DummyEmbedder())

    monkeypatch.setattr(searcher, "_semantic_search", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedder failed")))

    with pytest.raises(RuntimeError, match="embedder failed"):
        searcher.search("q", search_mode="semantic")
