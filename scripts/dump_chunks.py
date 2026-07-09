#!/usr/bin/env python3
"""Dump the canonical chunk set for a corpus using the SAME MultiLanguageChunker
the server uses at index time. Every A/B arm (torch server, MLX, GGUF) then embeds
these identical chunks, isolating the embedding model as the only variable — the
fairest possible cross-model retrieval comparison.

Usage:
  uv run python scripts/dump_chunks.py --corpus /path/to/corpus --out benchmarks/chunk_dump.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chunking.multi_language_chunker import MultiLanguageChunker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = Path(args.corpus).expanduser()
    chunker = MultiLanguageChunker(root_path=str(corpus))
    out: list[dict] = []
    n_files = 0
    n_skipped = 0
    for f in sorted(corpus.rglob("*")):
        if not f.is_file():
            continue
        if not chunker.is_supported(str(f)):
            n_skipped += 1
            continue
        n_files += 1
        try:
            chunks = chunker.chunk_file(str(f))
        except Exception as e:  # noqa: BLE001
            print(f"  chunk fail {f}: {e}", file=sys.stderr)
            continue
        for c in chunks:
            rel = getattr(c, "relative_path", None) or str(f)
            out.append({
                "chunk_id": getattr(c, "chunk_id", None) or f"{rel}:{getattr(c, 'start_line', '?')}-{getattr(c, 'end_line', '?')}",
                "path": rel,
                "start_line": getattr(c, "start_line", None),
                "end_line": getattr(c, "end_line", None),
                "name": getattr(c, "name", None),
                "chunk_type": getattr(c, "chunk_type", None),
                "text": getattr(c, "content", None),
            })
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).expanduser().write_text(json.dumps(out, indent=2))
    print(f"dumped {len(out)} chunks from {n_files} supported files "
          f"({n_skipped} unsupported) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
