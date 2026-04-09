from search.indexer import CodeIndexManager


def test_fts_upsert_many_writes_rows(tmp_path):
    indexer = CodeIndexManager(str(tmp_path))
    entries = [
        ("cid-1", "src/a.py", "def alpha(): return 1"),
        ("cid-2", "src/b.py", "def beta(): return 2"),
    ]
    indexer.fts_upsert_many(entries)

    hits = indexer.fts_search("alpha", k=5)
    assert hits
    assert hits[0][0] == "cid-1"
