"""A failed full index must persist status="failed" to stats.json.

Regression test for the missing-weights failure mode: previously the failure
path only updated resume.json (status="failed") but left stats.json at the
preliminary "indexing" status, so a *failed* index was indistinguishable from
one still in progress — which made a broken embedder look like a hang.
"""
import json

from search.incremental_indexer import IncrementalIndexer
from search.indexer import CodeIndexManager


class _ExplodingChunker:
    """Minimal chunker stub that raises once the full-index loop reaches it,
    i.e. after the preliminary status="indexing" record has been written."""

    def is_supported(self, path):  # called in _full_index's supported-files loop
        raise RuntimeError("boom: simulated indexing failure")


def test_failed_full_index_marks_status_failed(tmp_path):
    # A project with one file so the DAG has something to walk.
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.py").write_text("def f():\n    return 1\n")

    index_dir = tmp_path / "store" / "index"
    index = CodeIndexManager(str(index_dir))

    indexer = IncrementalIndexer(
        index_manager=index,
        embedder=None,  # never reached; failure happens before embedding
        chunker=_ExplodingChunker(),
        storage_dir=str(index_dir),
    )

    result = indexer.incremental_index(
        str(project), project_name="proj", force_full=True, resume=False
    )

    assert result.success is False

    stats_path = index_dir / "stats.json"
    assert stats_path.exists(), "stats.json should exist after a failed index"
    status = json.loads(stats_path.read_text()).get("status")
    assert status == "failed", (
        f"failed index must persist status='failed', got {status!r} "
        "(a lingering 'indexing' status hides real failures)"
    )
