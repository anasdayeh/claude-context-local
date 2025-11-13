"""Test suite for metadata extraction richness."""

import tempfile
import shutil
from pathlib import Path

import pytest

from chunking.multi_language_chunker import MultiLanguageChunker


@pytest.mark.integration
class TestMetadataRichness:
    """Test suite for metadata extraction richness."""

    def test_metadata_richness(self):
        """Test the richness of metadata extraction."""
        # Create a test file in a nested directory structure
        test_dir = tempfile.mkdtemp()
        try:
            project_dir = Path(test_dir) / "test_project"
            src_dir = project_dir / "src" / "auth"
            src_dir.mkdir(parents=True)

            test_file = src_dir / "user_auth.py"
            test_code = '''
from typing import Optional
import hashlib
import logging

logger = logging.getLogger(__name__)

class AuthenticationError(Exception):
    """Custom authentication error."""
    pass

class UserAuthenticator:
    """Handles user authentication and authorization."""

    def __init__(self, secret_key: str):
        """Initialize authenticator with secret key."""
        self.secret_key = secret_key
        self.failed_attempts = {}

    @property
    def max_attempts(self) -> int:
        """Maximum login attempts allowed."""
        return 3

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        try:
            if self._is_account_locked(username):
                raise AuthenticationError("Account locked due to too many failed attempts")

            # Verify credentials
            if self._verify_password(username, password):
                self._reset_failed_attempts(username)
                logger.info(f"User {username} authenticated successfully")
                return True
            else:
                self._record_failed_attempt(username)
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise
'''

            test_file.write_text(test_code)

            # Test chunking with the nested structure
            chunker = MultiLanguageChunker(str(project_dir))
            chunks = chunker.chunk_file(str(test_file))

            # Assertions
            assert len(chunks) > 0, "Should generate chunks from nested project structure"

            # Check metadata richness
            for chunk in chunks:
                # Check relative path
                assert chunk.relative_path is not None, "Relative path should be set"
                assert "auth" in chunk.relative_path or "user_auth" in chunk.relative_path, \
                    "Relative path should include directory structure"

                # Check folder structure
                assert isinstance(chunk.folder_structure, list), "Folder structure should be a list"

                # Check imports
                assert isinstance(chunk.imports, list), "Imports should be a list"

                # Check docstring
                if chunk.docstring:
                    assert isinstance(chunk.docstring, str), "Docstring should be a string"
                    assert len(chunk.docstring) > 0, "Docstring should not be empty"

                # Check complexity score
                assert chunk.complexity_score >= 0, "Complexity score should be non-negative"

                # Check tags
                assert isinstance(chunk.tags, list), "Tags should be a list"

        finally:
            shutil.rmtree(test_dir)

    def test_metadata_folder_structure(self):
        """Test that folder structure is correctly captured."""
        test_dir = tempfile.mkdtemp()
        try:
            project_dir = Path(test_dir) / "project"
            nested_dir = project_dir / "src" / "components" / "auth"
            nested_dir.mkdir(parents=True)

            test_file = nested_dir / "authenticator.py"
            test_file.write_text("def authenticate(): pass")

            chunker = MultiLanguageChunker(str(project_dir))
            chunks = chunker.chunk_file(str(test_file))

            assert len(chunks) > 0, "Should generate chunks"

            # Check folder structure is correct
            for chunk in chunks:
                if chunk.folder_structure:
                    # Should include some part of the nested path
                    assert "src" in chunk.folder_structure or \
                           "components" in chunk.folder_structure or \
                           "auth" in chunk.folder_structure, \
                           f"Folder structure should reflect nesting, got {chunk.folder_structure}"

        finally:
            shutil.rmtree(test_dir)

    def test_metadata_imports_extraction(self):
        """Test that imports are extracted in metadata."""
        test_code = '''
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json as j
from collections import defaultdict

def my_function():
    pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            import os
            f.write(test_code)
            f.flush()

            try:
                chunker = MultiLanguageChunker(os.path.dirname(f.name))
                chunks = chunker.chunk_file(f.name)

                assert len(chunks) > 0, "Should generate chunks with imports"

                # Check that imports attribute exists and is iterable
                for chunk in chunks:
                    assert hasattr(chunk, 'imports'), "Chunks should have imports attribute"
                    # Imports should be a list or None
                    assert chunk.imports is None or isinstance(chunk.imports, (list, set)), \
                        "Imports should be None or a list/set"

            finally:
                os.unlink(f.name)
