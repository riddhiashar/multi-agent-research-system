import time
from tools import cache


def test_roundtrip_and_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_PATH", tmp_path / "c.json")
    monkeypatch.setenv("SEARCH_CACHE", "true")
    assert cache.get("search", "Query A") is None
    cache.put("search", "Query A", value={"answer": "x"})
    assert cache.get("search", "  query a ") == {"answer": "x"}      # key is normalised
    assert cache.get("search", "query a", ttl=-1) is None            # expired


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_PATH", tmp_path / "c.json")
    monkeypatch.delenv("SEARCH_CACHE", raising=False)
    cache.put("search", "q", value=1)
    assert cache.get("search", "q") is None and not (tmp_path / "c.json").exists()


def test_corrupt_file_is_ignored(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    monkeypatch.setattr(cache, "CACHE_PATH", p)
    monkeypatch.setenv("SEARCH_CACHE", "true")
    assert cache.get("search", "q") is None
    cache.put("search", "q", value=2)
    assert cache.get("search", "q") == 2
