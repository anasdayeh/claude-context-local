import numpy as np

from search.searcher import IntelligentSearcher


class _FakeEmbedder:
    def embed_query(self, _query: str):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_document(self, _text: str):
        return np.array([1.0, 0.0], dtype=np.float32)


class _FakeIndexManager:
    def __init__(self):
        self.iter_calls = 0

    def search(self, _query_embedding, _k=5, _filters=None):
        return [
            (
                "c1",
                0.9,
                {
                    "relative_path": "src/a.py",
                    "file_path": "src/a.py",
                    "start_line": 10,
                    "end_line": 20,
                    "chunk_type": "function",
                    "name": "alpha",
                    "content_preview": "def alpha(): pass",
                    "folder_structure": ["src"],
                },
            ),
            (
                "c2",
                0.8,
                {
                    "relative_path": "src/a.py",
                    "file_path": "src/a.py",
                    "start_line": 30,
                    "end_line": 40,
                    "chunk_type": "function",
                    "name": "beta",
                    "content_preview": "def beta(): pass",
                    "folder_structure": ["src"],
                },
            ),
        ]

    def iter_all_chunks(self):
        self.iter_calls += 1
        yield "c1", {
            "relative_path": "src/a.py",
            "start_line": 10,
            "end_line": 20,
            "chunk_type": "function",
            "name": "alpha",
            "content_preview": "def alpha(): pass",
        }
        yield "c2", {
            "relative_path": "src/a.py",
            "start_line": 30,
            "end_line": 40,
            "chunk_type": "function",
            "name": "beta",
            "content_preview": "def beta(): pass",
        }


def test_context_cache_scans_metadata_once_per_query():
    index = _FakeIndexManager()
    searcher = IntelligentSearcher(index, _FakeEmbedder())
    results = searcher.search("alpha", k=2, context_depth=1)
    assert len(results) == 2
    assert index.iter_calls == 1
