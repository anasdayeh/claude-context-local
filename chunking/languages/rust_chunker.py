"""Rust-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class RustChunker(LanguageChunker):
    """Rust-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__('rust')

    def _get_splittable_node_types(self) -> Set[str]:
        """Rust-specific splittable node types."""
        return {
            'function_item',
            'impl_item',
            'struct_item',
            'enum_item',
            'trait_item',
            'mod_item',
            'macro_definition',
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract Rust-specific metadata."""
        metadata = {'node_type': node.type}

        # Extract name (identifier or type_identifier)
        for child in node.children:
            if child.type in ['identifier', 'type_identifier']:
                metadata['name'] = self.get_node_text(child, source)
                break

        # Check for async functions
        if node.type == 'function_item':
            for child in node.children:
                if child.type == 'async' or self.get_node_text(child, source) == 'async':
                    metadata['is_async'] = True
                    break

        # Extract impl type for impl blocks
        if node.type == 'impl_item':
            for child in node.children:
                if child.type in ['type_identifier', 'generic_type']:
                    metadata['impl_type'] = self.get_node_text(child, source)
                    break

        return metadata
