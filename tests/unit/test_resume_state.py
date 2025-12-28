from pathlib import Path


def test_resume_state_save_load_clear(tmp_path):
    from search.resume_state import ResumeState, load_resume_state, save_resume_state, clear_resume_state

    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    assert load_resume_state(index_dir) is None

    state = ResumeState(
        project_path="/repo/path",
        project_id="abc123",
        status="in_progress",
        files_total=10,
        files_completed=3,
        hashes={"src/a.py": "hash1", "src/b.py": "hash2"},
        completed={"src/a.py", "src/b.py"},
    )
    save_resume_state(index_dir, state)

    loaded = load_resume_state(index_dir)
    assert loaded is not None
    assert loaded.project_path == state.project_path
    assert loaded.project_id == state.project_id
    assert loaded.status == "in_progress"
    assert loaded.files_total == 10
    assert loaded.files_completed == 3
    assert loaded.hashes["src/a.py"] == "hash1"
    assert "src/b.py" in loaded.completed

    clear_resume_state(index_dir)
    assert load_resume_state(index_dir) is None
