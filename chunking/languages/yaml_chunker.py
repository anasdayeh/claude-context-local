"""YAML-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class YamlChunker(LanguageChunker):
    """YAML-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("yaml")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "block_mapping_pair",
            "flow_mapping_pair",
            "block_sequence_item",
            "flow_sequence_item",
            "block_mapping",
            "flow_mapping",
            "block_sequence",
            "flow_sequence",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        text = self.get_node_text(node, source).strip()
        if node.type in {"block_mapping_pair", "flow_mapping_pair"} and ":" in text:
            key = text.split(":", 1)[0].strip()
            if key:
                metadata["name"] = key
                metadata["type"] = "pair"
        elif node.type in {"block_sequence_item", "flow_sequence_item"}:
            # Best-effort sequence item label
            line = text.split("\n", 1)[0].strip()
            if line.startswith("-"):
                line = line.lstrip("-").strip()
            if line:
                metadata["name"] = line
                metadata["type"] = "item"

        return metadata
