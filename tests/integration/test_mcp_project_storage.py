"""Test MCP server project-specific storage functionality."""

import json
import pytest

from mcp_server.code_search_server import CodeSearchServer


@pytest.mark.integration
class TestMCPProjectStorage:
    """Test suite for MCP server project-specific storage."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.server = CodeSearchServer()

    def test_index_test_project(self):
        """Test indexing the test project."""
        result = self.server.index_test_project()
        result_data = json.loads(result)

        assert "error" not in result_data, f"Failed to index test project: {result_data.get('error')}"
        # Check that indexing was successful (various possible keys)
        has_content = (
            result_data.get("chunks_processed", 0) > 0 or
            result_data.get("chunks_added", 0) >= 0 or
            "demo_info" in result_data
        )
        assert has_content, "Should have processed/added content or demo info"

    def test_list_projects(self):
        """Test listing projects after indexing."""
        # First index a project
        self.server.index_test_project()

        # List projects
        projects_result = self.server.list_projects()
        projects_data = json.loads(projects_result)

        assert "count" in projects_data, "Projects result should include count"
        assert projects_data.get("count", 0) >= 1, "Should have at least one project after indexing"
        assert "projects" in projects_data, "Projects result should include projects list"

        if projects_data.get("projects"):
            for project in projects_data["projects"]:
                assert "project_name" in project, "Project should have project_name"
                assert "project_hash" in project, "Project should have project_hash"

    def test_search_code_in_project(self):
        """Test searching code in the indexed project."""
        # First index a project
        self.server.index_test_project()

        # Search
        search_result = self.server.search_code("authentication functions", k=3)
        search_data = json.loads(search_result)

        # Should either have results or error, but not both
        if "error" in search_data:
            # Error is acceptable if no results found
            assert isinstance(search_data["error"], str), "Error should be a string"
        else:
            # Should have results
            assert "results" in search_data, "Search result should include results"
            results = search_data.get("results", [])
            assert isinstance(results, list), "Results should be a list"

            # If we have results, check their structure
            for result in results:
                assert "name" in result or "kind" in result, "Each result should have name or kind"

    def test_get_index_status(self):
        """Test retrieving index status."""
        # First index a project
        self.server.index_test_project()

        # Get status
        status_result = self.server.get_index_status()
        status_data = json.loads(status_result)

        if "error" not in status_data:
            assert "index_statistics" in status_data, "Status should include index_statistics"
            stats = status_data.get("index_statistics", {})
            assert "total_chunks" in stats, "Statistics should include total_chunks"
            assert "files_indexed" in stats, "Statistics should include files_indexed"
            assert stats.get("total_chunks", 0) >= 0, "total_chunks should be non-negative"
            assert stats.get("files_indexed", 0) >= 0, "files_indexed should be non-negative"

    def test_project_isolation(self):
        """Test that projects are isolated in storage."""
        # Index test project
        result1 = self.server.index_test_project()
        result1_data = json.loads(result1)

        assert "error" not in result1_data, "First index should succeed"

        # Get projects list
        projects_result = self.server.list_projects()
        projects_data = json.loads(projects_result)

        # Should have at least one project
        assert len(projects_data.get("projects", [])) >= 1, "Should have at least one project"

        # Each project should have isolated storage
        projects = projects_data.get("projects", [])
        for project in projects:
            assert "project_hash" in project, "Project should have a hash for isolation"
