"""Plain-text chunker for non-AST file types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from chunking.code_chunk import CodeChunk


@dataclass
class TextChunk:
    content: str
    start_line: int
    end_line: int


class TextChunker:
    """Fallback chunker for text-like files without tree-sitter support."""

    def __init__(
        self,
        root_path: Optional[str] = None,
        max_lines: int = 200,
        max_chars: int = 8000,
    ) -> None:
        self.root_path = Path(root_path) if root_path else None
        self.max_lines = max_lines
        self.max_chars = max_chars

    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        path = Path(file_path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        if not content.strip():
            return []

        lines = content.splitlines()
        chunks: List[TextChunk] = []

        current: List[str] = []
        start_line = 1
        current_len = 0

        def flush(end_line: int) -> None:
            nonlocal current, start_line, current_len
            if not current:
                return
            chunks.append(
                TextChunk(
                    content="\n".join(current),
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            current = []
            start_line = end_line + 1
            current_len = 0

        for idx, line in enumerate(lines, start=1):
            line_len = len(line) + 1
            would_exceed = (
                current_len + line_len > self.max_chars
                or (idx - start_line + 1) > self.max_lines
            )
            if would_exceed:
                flush(idx - 1)

            current.append(line)
            current_len += line_len

        flush(len(lines))

        folder_parts = []
        if self.root_path:
            try:
                rel_path = path.relative_to(self.root_path)
                folder_parts = list(rel_path.parent.parts)
                relative_path = str(rel_path)
            except ValueError:
                folder_parts = [path.parent.name] if path.parent.name else []
                relative_path = str(path)
        else:
            folder_parts = [path.parent.name] if path.parent.name else []
            relative_path = str(path)

        ext = path.suffix.lower().lstrip(".")
        tags = ["text"]
        if ext:
            tags.append(ext)

        code_chunks: List[CodeChunk] = []
        for i, chunk in enumerate(chunks, start=1):
            code_chunks.append(
                CodeChunk(
                    file_path=str(path),
                    relative_path=relative_path,
                    folder_structure=folder_parts,
                    chunk_type="text",
                    content=chunk.content,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    name=f"{path.name}:{i}",
                    tags=tags,
                )
            )

        return code_chunks
