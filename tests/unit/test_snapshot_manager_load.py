"""Regression: a bad snapshot must fail LOUDLY (ERROR), not silently full-reindex.

Before the fix, load_latest_snapshot() swallowed every load failure at WARNING and
returned None, so a corrupt/stale-format snapshot silently triggered a full re-index
on every run with no visible signal. These tests pin the loud-failure behaviour.
"""
import json
import logging

from merkle.snapshot_manager import SnapshotManager


def test_corrupt_snapshot_logs_error_not_silent(tmp_path, caplog):
    sm = SnapshotManager(tmp_path)
    snap = sm.get_snapshot_path("/some/project")
    snap.write_text("{ this is definitely : not valid json")  # corrupt

    with caplog.at_level(logging.ERROR, logger="merkle.snapshot_manager"):
        dag = sm.load_latest_snapshot("/some/project")

    assert dag is None
    assert any(
        r.levelname == "ERROR" and "snapshot" in r.getMessage().lower()
        for r in caplog.records
    ), "corrupt snapshot must be logged at ERROR, not silently at WARNING"


def test_version_mismatch_logs_error_and_returns_none(tmp_path, caplog):
    sm = SnapshotManager(tmp_path)
    snap = sm.get_snapshot_path("/some/project")
    snap.write_text(json.dumps({"version": "0.9", "dag": {"root_path": "/x", "root_node": {}}}))

    with caplog.at_level(logging.ERROR, logger="merkle.snapshot_manager"):
        dag = sm.load_latest_snapshot("/some/project")

    assert dag is None
    assert any(
        r.levelname == "ERROR" and "version" in r.getMessage().lower()
        for r in caplog.records
    ), "a version mismatch must be logged at ERROR and force a full re-index"


def test_missing_snapshot_returns_none_quietly(tmp_path):
    # No snapshot on disk at all is a normal first-index condition, not an error.
    sm = SnapshotManager(tmp_path)
    assert sm.load_latest_snapshot("/never/indexed") is None
