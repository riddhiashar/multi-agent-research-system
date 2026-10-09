import agents.web_search_agent as wsa


def test_rate_limit_detection():
    assert wsa._is_rate_limit_error(Exception("429 Too Many Requests"))
    assert wsa._is_rate_limit_error(Exception("RESOURCE_EXHAUSTED: quota"))
    assert wsa._is_rate_limit_error(Exception("rate limit reached"))
    assert not wsa._is_rate_limit_error(Exception("connection reset"))


def test_search_one_retries_on_rate_limit_then_succeeds(monkeypatch):
    calls = {"count": 0}
    def flaky(q):
        calls["count"] += 1
        if calls["count"] < 2:
            raise Exception("429 rate limit reached")
        return {"answer": "ok", "sources": [{"url": "http://a"}], "provider": "tavily"}
    monkeypatch.setattr(wsa, "research_with_fallback", flaky)
    monkeypatch.setattr(wsa.time, "sleep", lambda s: None)  # skip real waiting in tests

    result = wsa.search_one("q", max_retries=2)
    assert result["status"] == "ok"
    assert calls["count"] == 2


def test_search_one_does_not_retry_non_rate_limit_errors(monkeypatch):
    calls = {"count": 0}
    def boom(q):
        calls["count"] += 1
        raise Exception("connection reset")
    monkeypatch.setattr(wsa, "research_with_fallback", boom)

    result = wsa.search_one("q", max_retries=2)
    assert result["status"] == "failed"
    assert calls["count"] == 1  # no retry for non-rate-limit errors


def test_search_single_node_success(monkeypatch):
    monkeypatch.setattr(wsa, "research_with_fallback",
                        lambda q: {"answer": "x", "sources": [{"url": "http://a"}], "provider": "tavily"})
    out = wsa.search_single_node({"query": "test question"})
    assert len(out["search_results"]) == 1
    assert out["search_results"][0]["sub_question"] == "test question"
    assert out["model_usage_log"][0]["agent"] == "web_search_agent"


def test_search_single_node_failure_goes_to_errors(monkeypatch):
    def boom(q):
        raise Exception("connection reset")
    monkeypatch.setattr(wsa, "research_with_fallback", boom)
    out = wsa.search_single_node({"query": "test question"})
    assert "errors" in out
    assert "search_results" not in out
