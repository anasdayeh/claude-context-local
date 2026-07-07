"""The index manifest records which embedding model built it, and refuses to be
opened by an incompatible one (different vector space) with a clear error instead
of a raw FAISS dimension crash.
"""
import pytest

from search.shard_manifest import ShardManifest, IndexModelMismatchError


def _m(**kw):
    base = dict(version=1, project_path="/x", embedding_dimension=768,
                index_type="flat", shard_count=0)
    base.update(kw)
    return ShardManifest(**base)


def test_embedding_model_roundtrips(tmp_path):
    m = _m(embedding_model="google/embeddinggemma-300m")
    p = tmp_path / "manifest.json"
    m.save(p)
    assert ShardManifest.load(p).embedding_model == "google/embeddinggemma-300m"


def test_legacy_manifest_loads_with_blank_model(tmp_path):
    # Pre-existing manifests have no embedding_model field — must default to "".
    p = tmp_path / "manifest.json"
    p.write_text('{"version":1,"project_path":"/x","embedding_dimension":768,'
                 '"index_type":"flat","shard_count":0,"shards":[]}')
    assert ShardManifest.load(p).embedding_model == ""


def test_assert_compatible_ok_on_match():
    _m(embedding_model="google/embeddinggemma-300m").assert_compatible(
        "google/embeddinggemma-300m", 768)  # must not raise


def test_assert_compatible_raises_on_model_mismatch():
    with pytest.raises(IndexModelMismatchError) as e:
        _m(embedding_model="google/embeddinggemma-300m").assert_compatible(
            "Qwen/Qwen3-Embedding-4B", 0)
    assert "embeddinggemma" in str(e.value) and "Qwen" in str(e.value)


def test_assert_compatible_raises_on_dim_mismatch_even_if_model_blank():
    with pytest.raises(IndexModelMismatchError):
        _m(embedding_dimension=768, embedding_model="").assert_compatible(
            "Qwen/Qwen3-Embedding-4B", 2560)


def test_assert_compatible_skips_unknown_values():
    # Nothing knowable to compare → no raise (legacy/empty).
    _m(embedding_dimension=0, embedding_model="").assert_compatible("anything", 0)
