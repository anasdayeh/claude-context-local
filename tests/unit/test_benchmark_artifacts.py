import json

from scripts.bench_artifacts import (
    artifact_is_reusable,
    atomic_write_json,
    atomic_write_text,
    build_run_fingerprint,
    hash_directory,
)


def _fingerprint(tmp_path):
    chunks = tmp_path / "chunks.json"
    queries = tmp_path / "queries.jsonl"
    config = tmp_path / "arms.yaml"
    source = tmp_path / "runner.py"
    if not chunks.exists():
        chunks.write_text('[{"text": "a"}]')
    if not queries.exists():
        queries.write_text('{"query": "a"}\n')
    if not config.exists():
        config.write_text("defaults: {k: 5}\n")
    if not source.exists():
        source.write_text("print('runner')\n")
    return build_run_fingerprint(
        chunk_path=chunks,
        query_path=queries,
        config_path=config,
        arm={"label": "gemma", "model_id": "example/model"},
        source_paths=[source],
    )


def test_artifact_reuse_requires_complete_matching_fingerprint(tmp_path):
    fingerprint = _fingerprint(tmp_path)
    path = tmp_path / "arm.json"
    atomic_write_json(
        path,
        {
            "artifact": {"schema_version": 2, "status": "complete", "fingerprint": fingerprint},
            "summary": {"label": "gemma"},
            "per_query": [],
        },
    )

    reusable, reason = artifact_is_reusable(path, fingerprint)

    assert reusable is True
    assert reason == "match"


def test_changed_query_invalidates_artifact(tmp_path):
    fingerprint = _fingerprint(tmp_path)
    path = tmp_path / "arm.json"
    atomic_write_json(
        path,
        {
            "artifact": {"schema_version": 2, "status": "complete", "fingerprint": fingerprint},
            "summary": {"label": "gemma"},
            "per_query": [],
        },
    )
    (tmp_path / "queries.jsonl").write_text('{"query": "changed"}\n')
    changed = _fingerprint(tmp_path)

    reusable, reason = artifact_is_reusable(path, changed)

    assert reusable is False
    assert reason == "fingerprint_mismatch"


def test_corrupt_or_incomplete_artifact_is_not_reused(tmp_path):
    fingerprint = _fingerprint(tmp_path)
    path = tmp_path / "arm.json"
    path.write_text("not-json")
    assert artifact_is_reusable(path, fingerprint) == (False, "invalid_json")

    path.write_text(json.dumps({"artifact": {"status": "running", "fingerprint": fingerprint}}))
    assert artifact_is_reusable(path, fingerprint) == (False, "incomplete")


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "nested" / "artifact.json"

    atomic_write_json(path, {"value": 1})

    assert json.loads(path.read_text()) == {"value": 1}
    assert list(path.parent.glob(".*.tmp")) == []


def test_atomic_text_and_directory_hash_are_deterministic(tmp_path):
    tree = tmp_path / "repo"
    tree.mkdir()
    (tree / "a.py").write_text("print('a')")
    (tree / "ignored.txt").write_text("ignored")

    first = hash_directory(tree, patterns=["*.py"])
    (tree / "ignored.txt").write_text("changed")
    assert hash_directory(tree, patterns=["*.py"]) == first
    (tree / "a.py").write_text("print('changed')")
    assert hash_directory(tree, patterns=["*.py"]) != first

    report = tmp_path / "report.md"
    atomic_write_text(report, "complete")
    assert report.read_text() == "complete"
    assert list(tmp_path.glob(".*.tmp")) == []
