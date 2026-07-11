from chunking.code_chunk import CodeChunk
from scripts.dump_chunks import serialize_chunk


def _chunk():
    return CodeChunk(
        content="def long_function():\n" + "    value = 1\n" * 500,
        chunk_type="function",
        start_line=1,
        end_line=501,
        file_path="/repo/example.py",
        relative_path="example.py",
        folder_structure=[],
        name="long_function",
        docstring="A long function used for testing.",
    )


def test_production_dump_uses_same_bounded_text_as_code_embedder():
    row = serialize_chunk(_chunk(), input_mode="production", max_chars=2048)

    assert len(row["text"]) <= 2048
    assert row["text"].startswith("Name: long_function\nType: function")
    assert row["input_policy"] == "production"
    assert row["source_chars"] > row["embedded_chars"]


def test_raw_dump_is_explicit_and_preserves_source_content():
    chunk = _chunk()
    row = serialize_chunk(chunk, input_mode="raw", max_chars=2048)

    assert row["text"] == chunk.content
    assert row["input_policy"] == "raw"


def test_production_formatter_never_exceeds_small_requested_limit():
    chunk = _chunk()
    chunk.name = "n" * 200

    row = serialize_chunk(chunk, input_mode="production", max_chars=32)

    assert len(row["text"]) <= 32
