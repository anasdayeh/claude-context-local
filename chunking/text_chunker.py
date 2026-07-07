"""Plain-text chunker for non-AST file types."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from chunking.code_chunk import CodeChunk


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

        return self.chunk_text(content, str(path))

    def chunk_text(
        self,
        content: str,
        file_path: str,
        *,
        chunk_type: str = "text",
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, object]] = None,
    ) -> List[CodeChunk]:
        path = Path(file_path)

        if not content.strip():
            return []

        lines = content.splitlines()
        folder_parts, relative_path = self._path_metadata(path)
        final_tags = list(tags or self._default_tags(path))

        code_chunks: List[CodeChunk] = []
        current: List[str] = []
        start_line = 1
        current_len = 0
        chunk_index = 1

        def flush(end_line: int) -> None:
            nonlocal current, start_line, current_len, chunk_index
            if not current:
                return
            code_chunks.append(
                CodeChunk(
                    file_path=str(path),
                    relative_path=relative_path,
                    folder_structure=folder_parts,
                    chunk_type=chunk_type,
                    content="\n".join(current),
                    start_line=start_line,
                    end_line=end_line,
                    name=name or f"{path.name}:{chunk_index}",
                    tags=list(final_tags),
                    extra_metadata=dict(extra_metadata or {}),
                )
            )
            chunk_index += 1
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

        return code_chunks

    def _default_tags(self, path: Path) -> List[str]:
        ext = path.suffix.lower().lstrip(".")
        tags = ["text"]
        if ext:
            tags.append(ext)
        return tags

    def _path_metadata(self, path: Path) -> tuple[List[str], str]:
        folder_parts = []
        if self.root_path:
            try:
                rel_path = path.relative_to(self.root_path)
                folder_parts = list(rel_path.parent.parts)
                return folder_parts, str(rel_path)
            except ValueError:
                folder_parts = [path.parent.name] if path.parent.name else []
                return folder_parts, str(path)
        folder_parts = [path.parent.name] if path.parent.name else []
        return folder_parts, str(path)
