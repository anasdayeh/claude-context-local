"""TOML-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class TomlChunker(LanguageChunker):
    """TOML-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("toml")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "pair",
            "table",
            "table_array",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}
        text = self.get_node_text(node, source).strip()

        if node.type == "pair" and "=" in text:
            key = text.split("=", 1)[0].strip()
            if key:
                metadata["name"] = key
                metadata["type"] = "pair"
        elif node.type in {"table", "table_array"}:
            # Headers like [table] or [[table]]
            line = text.split("\n", 1)[0].strip()
            name = line.strip("[]").strip()
            if name:
                metadata["name"] = name
                metadata["type"] = node.type

        return metadata
