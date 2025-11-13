"""Unit tests for JavaScript tree-sitter chunking."""

import pytest

from chunking.languages import JavaScriptChunker


@pytest.mark.unit
class TestJavaScriptChunker:
    """Test JavaScript-specific tree-sitter chunking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        try:
            self.chunker = JavaScriptChunker()
        except ValueError:
            pytest.skip("tree-sitter-javascript not installed")

    def test_function_types(self):
        """Test different JavaScript function types."""
        code = '''
function normalFunction() {
    return 42;
}

const arrowFunction = () => {
    return "arrow";
};

const asyncFunc = async function() {
    await Promise.resolve();
};

class MyClass {
    method() {
        return "method";
    }
}
'''
        chunks = self.chunker.chunk_code(code)

        # Should find various function types
        assert len(chunks) >= 1, "Should find at least one chunk for function types"

        # Check for different node types
        node_types = {c.node_type for c in chunks}
        assert len(node_types) > 0, "Should have at least some chunks with node types"

        # Verify structure
        for chunk in chunks:
            assert hasattr(chunk, 'node_type'), "Chunk should have node_type"
            assert hasattr(chunk, 'content'), "Chunk should have content"

    def test_arrow_function_detection(self):
        """Test detection of arrow functions specifically."""
        code = '''
const simpleArrow = () => 42;
const paramArrow = (x, y) => x + y;
const blockArrow = (x) => {
    const result = x * 2;
    return result;
};
'''
        chunks = self.chunker.chunk_code(code)

        assert len(chunks) > 0, "Should find arrow function chunks"

        # All chunks should be valid
        for chunk in chunks:
            assert isinstance(chunk.content, str), "Content should be string"
            assert len(chunk.content) > 0, "Content should not be empty"
