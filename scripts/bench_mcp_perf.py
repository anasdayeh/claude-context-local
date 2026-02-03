import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = {
        "embedding_throughput_chunks_per_sec": None,
        "indexing_time_seconds": None,
        "search_latency_ms": {"p50": None, "p95": None},
        "meta": {"dry_run": bool(args.dry_run)},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
