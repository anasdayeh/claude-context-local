"""index_repo.py must default to INCREMENTAL, matching the MCP surface.

Root cause of silent full-reindexing: the CLI defaulted to full (--incremental was
store_true), and 3 of the 4 reindex_*.sh scripts call index_repo.py without it, so
they full-reindexed every run. These tests pin incremental-by-default with an
explicit --full opt-out.
"""
from scripts.index_repo import _build_parser


def test_defaults_to_incremental():
    args = _build_parser().parse_args(["/tmp/repo"])
    assert args.incremental is True


def test_full_flag_forces_full_reindex():
    args = _build_parser().parse_args(["/tmp/repo", "--full"])
    assert args.incremental is False


def test_incremental_flag_still_accepted():
    args = _build_parser().parse_args(["/tmp/repo", "--incremental"])
    assert args.incremental is True
