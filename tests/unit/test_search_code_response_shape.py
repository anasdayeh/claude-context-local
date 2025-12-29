from mcp_server.code_search_server import CodeSearchServer


def test_search_code_switch_error_returns_dict():
    server = CodeSearchServer()
    result = server.search_code("anything", project_path="/nonexistent/path")
    assert isinstance(result, dict)
    assert "error" in result
