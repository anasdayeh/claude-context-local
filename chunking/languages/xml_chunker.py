"""XML-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class XmlChunker(LanguageChunker):
    """XML-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("xml")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "element",
            "processing_instruction",
            "doctype",
            "cdata_section",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

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
            metadata["type"] = "element"

        return metadata
