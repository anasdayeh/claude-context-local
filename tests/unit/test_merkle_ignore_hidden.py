from pathlib import Path


def test_merkle_ignores_hidden_directories(tmp_path):
    from merkle.merkle_dag import MerkleDAG

    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.py").write_text("print('secret')\n")

    (tmp_path / "visible.py").write_text("print('visible')\n")

    dag = MerkleDAG(str(tmp_path))
    dag.build()
    files = set(dag.get_all_files())

    assert "visible.py" in files
    assert ".hidden/secret.py" not in files
