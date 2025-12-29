from search.searcher import IntelligentSearcher


class DummyIndexManager:
    pass


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
