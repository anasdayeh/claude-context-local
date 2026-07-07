"""SentenceTransformerModel must carry per-model `model_kwargs` and a query instruction.

Needed for Qwen3-Embedding-4B: it requires model_kwargs={"attn_implementation":"eager"}
(macOS SDPA NaN fix) and a query-side "Instruct: ...\\nQuery:" prefix that documents
must NOT receive. These are threaded via the shared wrapper without a real model load.
"""
import numpy as np

from embeddings.sentence_transformer import SentenceTransformerModel


class _FakeST:
    """Records which encode path was hit and with what texts."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kw):
        self.calls.append(("encode", list(texts), kw))
        return np.zeros((len(texts), 4), dtype=np.float32)

    def encode_query(self, texts, **kw):
        self.calls.append(("encode_query", list(texts), kw))
        return np.zeros((len(texts), 4), dtype=np.float32)

    def encode_document(self, texts, **kw):
        self.calls.append(("encode_document", list(texts), kw))
        return np.zeros((len(texts), 4), dtype=np.float32)


def _with_fake(**kwargs):
    m = SentenceTransformerModel("all-MiniLM-L6-v2", **kwargs)
    fake = _FakeST()
    m.__dict__["model"] = fake       # bypass lazy load
    m._model_loaded = True
    return m, fake


def test_init_stores_model_kwargs_and_instruction():
    m = SentenceTransformerModel(
        "all-MiniLM-L6-v2",
        model_kwargs={"attn_implementation": "eager"},
        query_instruction="Instruct: t\nQuery:",
    )
    assert m._extra_model_kwargs == {"attn_implementation": "eager"}
    assert m.query_instruction == "Instruct: t\nQuery:"


def test_query_instruction_prepended_to_queries_only():
    m, fake = _with_fake(query_instruction="Instruct: t\nQuery:")
    m.encode_query(["find the bug"])
    texts_seen = "".join("".join(c[1]) for c in fake.calls)
    assert "Instruct: t\nQuery:find the bug" in texts_seen, fake.calls

    fake.calls.clear()
    m.encode_document(["def foo(): pass"])
    doc_texts = "".join("".join(c[1]) for c in fake.calls)
    assert "Instruct:" not in doc_texts, fake.calls


def test_no_instruction_keeps_default_query_path():
    m, fake = _with_fake()  # no query_instruction
    m.encode_query(["q"])
    assert fake.calls and fake.calls[0][0] == "encode_query", fake.calls
