#!/usr/bin/env python3
"""Offline indexer that uses the same MCP pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mcp_server.code_search_server import CodeSearchServer


def _setup_logging(verbose: bool, log_file: str | None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = None
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(log_path)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a repo using the MCP pipeline")
    parser.add_argument("path", help="Path to the repository to index")
    parser.add_argument("--project-name", dest="project_name", help="Override project name")
    parser.add_argument(
        "--storage-dir",
        dest="storage_dir",
        help="Base storage directory (overrides CODE_SEARCH_STORAGE)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run incremental indexing (default: full)",
    )
    parser.add_argument(
        "--sharded",
        action="store_true",
        help="Force sharded indexing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-file",
        help="Write logs to a file",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run indexing as a background job and poll progress",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_path = Path(args.path).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        print(f"Invalid path: {repo_path}", file=sys.stderr)
        return 2

    if args.storage_dir:
        os.environ["CODE_SEARCH_STORAGE"] = str(Path(args.storage_dir).expanduser())

    if args.sharded:
        os.environ["CODE_SEARCH_SHARDED_INDEX"] = "1"

    if args.log_file:
        os.environ["CODE_SEARCH_LOG_FILE"] = args.log_file

    _setup_logging(args.verbose, args.log_file)
    logger = logging.getLogger(__name__)

    server = CodeSearchServer()

    if args.background:
        job = server.start_index_job(
            str(repo_path),
            project_name=args.project_name,
            file_patterns=None,
            incremental=args.incremental,
        )
        job_info = job.get("job", {})
        job_id = job_info.get("job_id")
        if not job_id:
            print(job)
            return 1

        logger.info("Started background indexing job: %s", job_id)
        last_seen = 0.0
        while True:
            status = server.get_index_job_status(job_id=job_id)
            job_state = status.get("job", {})
            for event in job_state.get("events", []):
                if event.get("ts", 0) > last_seen:
                    logger.info("%s", event.get("message"))
                    last_seen = event.get("ts", last_seen)
            if job_state.get("status") in {"completed", "failed", "canceled"}:
                print(job_state)
                return 0 if job_state.get("status") == "completed" else 1
            time.sleep(5)

    result = server.index_directory(
        str(repo_path),
        project_name=args.project_name,
        incremental=args.incremental,
    )
    print(result)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
