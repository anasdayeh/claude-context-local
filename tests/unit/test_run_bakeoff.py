from pathlib import Path

import pytest

from scripts.run_bakeoff import BakeoffLock, aggregate_exit_status, wait_for_memory_handoff


def test_required_stage_failure_returns_nonzero():
    assert aggregate_exit_status({"gemma": "ok", "reranker": "FAIL(rc=1)"}) == 1


def test_successful_and_intentional_skip_stages_return_zero():
    assert aggregate_exit_status({"gemma": "ok", "blind+render": "5 arms"}) == 0
    assert aggregate_exit_status({"reranker": "SKIP (no base artifact)"}) == 0


def test_bakeoff_lock_rejects_concurrent_owner(tmp_path):
    path = Path(tmp_path) / "bakeoff.lock"

    with BakeoffLock(path, run_id="first"):
        with pytest.raises(RuntimeError, match="already running"):
            with BakeoffLock(path, run_id="second"):
                pass

    with BakeoffLock(path, run_id="third"):
        assert "third" in path.read_text()


def test_memory_handoff_requires_stable_available_samples():
    samples = iter([1 * 1024**3, 5 * 1024**3, 5 * 1024**3])
    sleeps = []

    result = wait_for_memory_handoff(
        min_available_bytes=4 * 1024**3,
        stable_samples=2,
        min_delay_seconds=0,
        poll_seconds=0.01,
        timeout_seconds=1,
        read_available=lambda: next(samples),
        sleep=sleeps.append,
    )

    assert result["stable"] is True
    assert result["available_gb"] == 5.0
    assert len(sleeps) == 2
