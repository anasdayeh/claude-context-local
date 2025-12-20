"""CSS-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class CssChunker(LanguageChunker):
    """CSS-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("css")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "rule_set",
            "at_rule",
            "media_statement",
            "keyframes_statement",
            "supports_statement",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        if node.type == "rule_set":
            # Try to capture selector text
            for child in node.children:
                if child.type in {"selectors", "selector_list"}:
                    metadata["name"] = self.get_node_text(child, source).strip()
                    break
        elif node.type == "at_rule":
            metadata["name"] = self.get_node_text(node, source).split("{", 1)[0].strip()

        return metadata
