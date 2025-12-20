"""Astro-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class AstroChunker(LanguageChunker):
    """Astro-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("astro")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "frontmatter",
            "element",
            "component",
            "script_element",
            "style_element",
            "fragment",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        # Best-effort tag/component name extraction
        tag_name = None
        for child in node.children:
            if child.type in {"tag_name", "identifier", "component_name"}:
                tag_name = self.get_node_text(child, source)
                break

        if tag_name:
            metadata["name"] = tag_name

        if node.type == "frontmatter":
            metadata["type"] = "frontmatter"
        elif node.type == "script_element":
            metadata["type"] = "script"
        elif node.type == "style_element":
            metadata["type"] = "style"
        elif tag_name:
            metadata["type"] = "element"

        return metadata
