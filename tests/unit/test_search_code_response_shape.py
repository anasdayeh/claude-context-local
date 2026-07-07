from mcp_server.code_search_server import CodeSearchServer
from search.searcher import SearchResult


def test_search_code_switch_error_returns_dict():
    server = CodeSearchServer()
    result = server.search_code("anything", project_path="/nonexistent/path")
    assert isinstance(result, dict)
    assert "error" in result


def test_search_result_payload_includes_useful_snippet_preview_and_context():
    result = SearchResult(
        chunk_id="chunk-1",
        similarity_score=0.9,
        content_preview="",
        file_path="/repo/mcp_server/mcp_tools.py",
        relative_path="mcp_server/mcp_tools.py",
        folder_structure=["mcp_server"],
        chunk_type="function",
        name="search_code",
        parent_name=None,
        start_line=10,
        end_line=25,
        docstring="Search code and return MCP result payloads.",
        tags=["api"],
        context_info={
            "file_context": {"total_chunks_in_file": 3, "folder_path": "mcp_server"},
            "file_neighbors": [
                {
                    "chunk_id": "chunk-0",
                    "lines": "1-9",
                    "chunk_type": "function",
                    "name": "index_directory",
                    "preview": "def index_directory():\n    return {'success': True}",
                }
            ],
        },
        content=(
            "def search_code(query: str):\n"
            "    results = run_search(query)\n"
            "    return {'results': results, 'meta': {'tool': 'search_code'}}\n"
        ),
        query="search_code results meta",
    )

    payload = result.to_search_tool_dict()

    assert "return {'results': results" in payload["snippet"]
    assert payload["content_preview"].startswith("def search_code")
    assert payload["context"]["file_context"]["total_chunks_in_file"] == 3
    assert payload["context"]["file_neighbors"][0]["name"] == "index_directory"
