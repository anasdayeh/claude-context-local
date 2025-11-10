"""Unit tests for MerkleNode class."""

from unittest import TestCase

from merkle.merkle_dag import MerkleNode


class TestMerkleNode(TestCase):
    """Test MerkleNode class."""

    def test_node_creation(self):
        """Test creating a Merkle node."""
        node = MerkleNode(
            path='test.py',
            hash='abc123',
            is_file=True,
            size=100
        )

        assert node.path == 'test.py'
        assert node.hash == 'abc123'
        assert node.is_file is True
        assert node.size == 100
        assert len(node.children) == 0

    def test_node_serialization(self):
        """Test node to/from dict conversion."""
        # Create parent with children
        child1 = MerkleNode('child1.py', 'hash1', True, 50)
        child2 = MerkleNode('child2.py', 'hash2', True, 75)
        parent = MerkleNode('parent', 'parent_hash', False, 0)
        parent.children = [child1, child2]

        # Serialize
        data = parent.to_dict()

        # Verify structure
        assert data['path'] == 'parent'
        assert data['hash'] == 'parent_hash'
        assert data['is_file'] is False
        assert len(data['children']) == 2

        # Deserialize
        restored = MerkleNode.from_dict(data)

        # Verify restoration
        assert restored.path == parent.path
        assert restored.hash == parent.hash
        assert len(restored.children) == 2
        assert restored.children[0].path == 'child1.py'
        assert restored.children[1].path == 'child2.py'
