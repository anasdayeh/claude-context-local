"""TypeScript-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class TypeScriptChunker(LanguageChunker):
    """TypeScript-specific chunker using tree-sitter."""

    def __init__(self, use_tsx: bool = False):
        super().__init__('tsx' if use_tsx else 'typescript')
        self.use_tsx = use_tsx

    def _get_splittable_node_types(self) -> Set[str]:
        """TypeScript-specific splittable node types."""
        return {
            'function_declaration',
            'function',
            'arrow_function',
            'class_declaration',
            'method_definition',
            'generator_function',
            'generator_function_declaration',
            'interface_declaration',
            'type_alias_declaration',
            'enum_declaration',
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract TypeScript-specific metadata."""
        metadata = {'node_type': node.type}

        # Extract name
        for child in node.children:
            if child.type in ['identifier', 'type_identifier']:
                metadata['name'] = self.get_node_text(child, source)
                break

        # Check for async
        if node.children and self.get_node_text(node.children[0], source) == 'async':
            metadata['is_async'] = True

        # Check for export
        if node.children and self.get_node_text(node.children[0], source) == 'export':
            metadata['is_export'] = True

        # Check for generic parameters
        for child in node.children:
            if child.type == 'type_parameters':
                metadata['has_generics'] = True
                break

        return metadata
