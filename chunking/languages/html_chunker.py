"""HTML-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class HtmlChunker(LanguageChunker):
    """HTML-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("html")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "element",
            "script_element",
            "style_element",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        # Try to extract tag name from start_tag/tag_name
        tag_name = None
        for child in node.children:
            if child.type == "start_tag":
                for tag_child in child.children:
                    if tag_child.type == "tag_name":
                        tag_name = self.get_node_text(tag_child, source)
                        break
            if tag_name:
                break

        if not tag_name:
            for child in node.children:
                if child.type == "tag_name":
                    tag_name = self.get_node_text(child, source)
                    break

        if tag_name:
            metadata["name"] = tag_name

        if node.type == "script_element":
            metadata["type"] = "script"
        elif node.type == "style_element":
            metadata["type"] = "style"
        elif tag_name:
            metadata["type"] = "element"

        return metadata
