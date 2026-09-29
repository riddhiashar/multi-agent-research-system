import agents.web_search_agent as wsa


def fake_ok(q):
    return {"answer": "some answer", "sources": [{"title": "T", "url": "http://a"}], "provider": "gemini"}


def fake_boom(q):
    raise RuntimeError("both providers down")


def test_search_one_ok(monkeypatch):
    monkeypatch.setattr(wsa, "research_with_fallback", fake_ok)
    r = wsa.search_one("q")
    assert r["status"] == "ok" and r["provider"] == "gemini"


def test_search_one_never_raises(monkeypatch):
    monkeypatch.setattr(wsa, "research_with_fallback", fake_boom)
    r = wsa.search_one("q")
    assert r["status"] == "failed" and "RuntimeError" in r["error"]


def test_search_one_empty_when_no_sources(monkeypatch):
    monkeypatch.setattr(wsa, "research_with_fallback",
                        lambda q: {"answer": "x", "sources": [], "provider": "tavily"})
    assert wsa.search_one("q")["status"] == "empty"


def test_node_dedupes_and_caps(monkeypatch):
    monkeypatch.setattr(wsa, "research_with_fallback", fake_ok)
    plan = ["a", "A", "b", "c", "d", "e", "f", "g"]
    out = wsa.web_search_node({"plan": plan})
    assert len(out["search_results"]) == wsa.MAX_QUERIES


def test_node_empty_plan(monkeypatch):
    out = wsa.web_search_node({"plan": []})
    assert out["errors"] and "search_results" not in out


def test_node_partial_failure(monkeypatch):
    def flaky(q):
        if q == "bad":
            raise RuntimeError("x")
        return fake_ok(q)
    monkeypatch.setattr(wsa, "research_with_fallback", flaky)
    out = wsa.web_search_node({"plan": ["good", "bad"]})
    assert len(out["search_results"]) == 1 and len(out["errors"]) == 1
