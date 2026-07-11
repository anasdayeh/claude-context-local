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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chunking.multi_language_chunker import MultiLanguageChunker  # noqa: E402
from embeddings.content_formatter import format_embedding_content  # noqa: E402
from scripts.bench_artifacts import atomic_write_json  # noqa: E402


def serialize_chunk(chunk, *, input_mode: str, max_chars: int) -> dict:
    source = getattr(chunk, "content", None) or ""
    text = source if input_mode == "raw" else format_embedding_content(chunk, max_chars=max_chars)
    rel = getattr(chunk, "relative_path", None) or getattr(chunk, "file_path", None)
    return {
        "chunk_id": getattr(chunk, "chunk_id", None)
        or f"{rel}:{getattr(chunk, 'start_line', '?')}-{getattr(chunk, 'end_line', '?')}",
        "path": rel,
        "start_line": getattr(chunk, "start_line", None),
        "end_line": getattr(chunk, "end_line", None),
        "name": getattr(chunk, "name", None),
        "chunk_type": getattr(chunk, "chunk_type", None),
        "text": text,
        "input_policy": input_mode,
        "source_chars": len(source),
        "embedded_chars": len(text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-mode", choices=("production", "raw"), default="production")
    ap.add_argument("--max-chars", type=int, default=2048)
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
            out.append(serialize_chunk(c, input_mode=args.input_mode, max_chars=args.max_chars))
    atomic_write_json(Path(args.out).expanduser(), out)
    print(f"dumped {len(out)} chunks from {n_files} supported files "
          f"({n_skipped} unsupported) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
