"""Svelte-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class SvelteChunker(LanguageChunker):
    """Svelte-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__('svelte')

    def _get_splittable_node_types(self) -> Set[str]:
        """Svelte-specific splittable node types."""
        return {
            'script_element',
            'style_element',
            'function_declaration',
            'function',
            'arrow_function',
            'class_declaration',
            'method_definition',
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract Svelte-specific metadata."""
        metadata = {'node_type': node.type}

        # Extract script type (module or instance)
        if node.type == 'script_element':
            for child in node.children:
                if child.type == 'start_tag':
                    tag_text = self.get_node_text(child, source)
                    if 'context="module"' in tag_text:
                        metadata['script_type'] = 'module'
                    else:
                        metadata['script_type'] = 'instance'
                    break

        # Extract style scope
        elif node.type == 'style_element':
            for child in node.children:
                if child.type == 'start_tag':
                    tag_text = self.get_node_text(child, source)
                    if 'global' in tag_text:
                        metadata['style_scope'] = 'global'
                    else:
                        metadata['style_scope'] = 'component'
                    break

        # Extract function/class names
        for child in node.children:
            if child.type == 'identifier':
                metadata['name'] = self.get_node_text(child, source)
                break

        return metadata
