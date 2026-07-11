"""Pure production input formatting shared by indexing and benchmarks."""
from __future__ import annotations


def format_embedding_content(chunk, max_chars: int = 2048) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    parts = []
    name = chunk.name or "unknown"
    chunk_type = chunk.chunk_type or "unknown"
    parts.append(f"Name: {name}")
    parts.append(f"Type: {chunk_type}")

    if getattr(chunk, "parent_name", None):
        parts.append(f"Context: {chunk.parent_name}")
    tags = chunk.tags or []
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")

    docstring = chunk.docstring or ""
    overhead = sum(len(part) + 1 for part in parts) + 10
    remaining_budget = max_chars - overhead
    if remaining_budget <= 20:
        return f"Name: {name}\n{(chunk.content or '')[:max_chars // 2]}"[:max_chars]

    docstring_budget = min(len(docstring), int(remaining_budget * 0.3))
    if len(docstring) < remaining_budget * 0.5:
        docstring_budget = len(docstring)
    if docstring:
        parts.append(f"Docstring: {docstring[:docstring_budget]}")

    header = "\n".join(parts)
    content = chunk.content or ""
    if not content:
        return header[:max_chars]
    full_text = f"{header}\n{content}"
    if len(full_text) <= max_chars:
        return full_text
    allowed = max(0, max_chars - len(header) - 1)
    return f"{header}\n{content[:allowed]}"
