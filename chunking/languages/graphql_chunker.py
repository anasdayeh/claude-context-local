"""GraphQL-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class GraphqlChunker(LanguageChunker):
    """GraphQL-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__("graphql")

    def _get_splittable_node_types(self) -> Set[str]:
        return {
            "operation_definition",
            "fragment_definition",
            "schema_definition",
            "directive_definition",
            "type_definition",
            "object_type_definition",
            "interface_type_definition",
            "input_object_type_definition",
            "enum_type_definition",
            "union_type_definition",
            "scalar_type_definition",
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        metadata = {"node_type": node.type}

        name = None
        for child in node.children:
            if child.type == "name":
                name = self.get_node_text(child, source)
                break

        if name:
            metadata["name"] = name
            metadata["type"] = node.type

        return metadata
