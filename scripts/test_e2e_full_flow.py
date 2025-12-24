#!/usr/bin/env python3
"""
End-to-End Verification Script for MCP Code Search Server (Logic Only).
Includes mocks for all external dependencies including mcp SDK.
"""

import os
import sys
import shutil
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import types

# --- MOCKING INFRASTRUCTURE ---

# Mock FAISS
mock_faiss = MagicMock()
mock_index = MagicMock()
mock_index.ntotal = 0
def add_with_ids(vectors, ids):
    mock_index.ntotal += len(vectors)
mock_index.add_with_ids = MagicMock(side_effect=add_with_ids)
mock_index.search.return_value = ([[1.0]], [[0]]) 
mock_faiss.IndexFlatIP.return_value = mock_index
mock_faiss.read_index.return_value = mock_index
sys.modules["faiss"] = mock_faiss

# Mock torch
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.backends"] = MagicMock()

# Mock sentence_transformers
mock_st = MagicMock()
def mock_encode(texts, **kwargs): return [[0.1] * 768 for _ in range(len(texts))]
mock_st.encode.side_effect = mock_encode
mock_st.encode_document.side_effect = mock_encode
mock_st.encode_query.side_effect = mock_encode
mock_st.get_sentence_embedding_dimension.return_value = 768
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock(return_value=mock_st)

# Mock sqlitedict shared state
class MockSqliteDict(dict):
    _shared_state = {}
    
    def __init__(self, filename=None, tablename='unnamed', autocommit=False, flag='c', journal_mode="DELETE", encode=None, decode=None, **kwargs):
        self.filename = str(filename) if filename else "memory"
        if self.filename not in self._shared_state:
             self._shared_state[self.filename] = {}
        # Initialize dict with shared state
        super().__init__(self._shared_state[self.filename])
        
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._shared_state[self.filename][key] = value
        
    def __delitem__(self, key):
        super().__delitem__(key)
        if key in self._shared_state[self.filename]:
            del self._shared_state[self.filename][key]
            
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._shared_state[self.filename].update(self)

    def commit(self): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

sys.modules["sqlitedict"] = MagicMock()
sys.modules["sqlitedict"].SqliteDict = MockSqliteDict

# Mock MCP SDK
mock_mcp = MagicMock()
mock_fastmcp = MagicMock()
mock_mcp.server.fastmcp = mock_fastmcp
sys.modules["mcp"] = mock_mcp
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = mock_fastmcp

# --- END MOCKS ---

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("E2E_Test")

# Imports after path setup
try:
    from mcp_server.code_search_server import CodeSearchServer
    from embeddings.embedding_models_register import AVAILIABLE_MODELS
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

def create_test_project(root_dir: Path):
    """Create a multi-language dummy project."""
    src = root_dir / "src"
    src.mkdir()
    (src / "math_ops.py").write_text('''
def add(a, b):
    """Adds two numbers."""
    return a + b
''')
    (src / "main.go").write_text('''
package main
import "fmt"
func main() {
    fmt.Println("Hello from Go")
}
''')
    logger.info(f"Created test project at {root_dir}")

def run_verification():
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir) / "project"
        project_path.mkdir()
        create_test_project(project_path)
        
        server = CodeSearchServer()
        server.storage_root = Path(temp_dir) / "claude_code_search_test"
        server.storage_root.mkdir()
        
        logger.info("--- Step 1: Indexing Directory ---")
        result = server.index_directory(
            directory_path=str(project_path),
            project_name="e2e_test_project",
            incremental=False
        )
        
        if not result.get("success"):
            logger.error(f"Indexing failed: {result}")
            sys.exit(1)
            
        logger.info(f"Indexing success: {result}")
        assert result["chunks_added"] > 0, "No chunks indexed!"
        
        logger.info("--- Step 2: Semantic Search ---")
        hits = server.search_code(query="add numbers", project_path=str(project_path))
        if hits:
             logger.info(f"✅ Search execution successful (returned {len(hits)} hits)")
        else:
             logger.warning(f"Search returned no hits")

        hits = server.search_code(
            query="helper function", 
            project_path=str(project_path),
            file_pattern="*.go"
        )
        logger.info("✅ Filtered search execution successful")

        stats = server.get_stats(project_path=str(project_path))
        logger.info(f"Stats: {stats}")
        assert stats["total_chunks"] > 0
        
        logger.info("--- Step 3: Incremental Indexing Check ---")
        (project_path / "src" / "math_ops.py").write_text('''
def add(a, b):
    return a + b
def new_func():
    return "new"
''')
        result_inc = server.index_directory(
             directory_path=str(project_path),
             project_name="e2e_test_project",
             incremental=True
        )
        logger.info(f"Incremental Result: {result_inc}")
        assert result_inc["success"]
        logger.info("✅ Incremental indexing execution passed")
        
    logger.info("\n🎉 E2E VERIFICATION (LOGIC) PASSED SUCCESSFULLY 🎉")

if __name__ == "__main__":
    run_verification()
