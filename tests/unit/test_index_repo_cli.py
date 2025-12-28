import subprocess
import sys


def test_index_repo_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/index_repo.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
