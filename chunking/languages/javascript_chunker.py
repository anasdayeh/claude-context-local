"""JavaScript-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class JavaScriptChunker(LanguageChunker):
    """JavaScript-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__('javascript')

    def _get_splittable_node_types(self) -> Set[str]:
        """JavaScript-specific splittable node types."""
        return {
            'function_declaration',
            'function',
            'arrow_function',
            'class_declaration',
            'method_definition',
            'generator_function',
            'generator_function_declaration',
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract JavaScript-specific metadata."""
        metadata = {'node_type': node.type}

        # Extract function/class name
        for child in node.children:
            if child.type == 'identifier':
                metadata['name'] = self.get_node_text(child, source)
                break

        # Check for async
        if node.children and self.get_node_text(node.children[0], source) == 'async':
            metadata['is_async'] = True

        # Check for generator
        if 'generator' in node.type:
            metadata['is_generator'] = True

        return metadata
