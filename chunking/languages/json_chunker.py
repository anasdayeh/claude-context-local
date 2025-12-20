"""JSON-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class JsonChunker(LanguageChunker):
    """JSON-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("json")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "pair",
            "array",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        if node.type == "pair":
            for child in node.children:
                if child.type == "string":
                    key = self.get_node_text(child, source).strip().strip('"')
                    metadata["name"] = key
                    break
        return metadata
