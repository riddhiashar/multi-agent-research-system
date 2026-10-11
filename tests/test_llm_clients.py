import tools.llm_clients as llc


class FakeResponse:
    content = "ok"


def test_is_rate_limit_error_detects_variants():
    assert llc.is_rate_limit_error(Exception("429 Too Many Requests"))
    assert llc.is_rate_limit_error(Exception("RESOURCE_EXHAUSTED: quota"))
    assert llc.is_rate_limit_error(Exception("rate limit reached"))
    assert not llc.is_rate_limit_error(Exception("connection reset"))
    assert not llc.is_rate_limit_error(Exception("invalid API key"))


def test_call_llm_retries_on_rate_limit_then_succeeds(monkeypatch):
    calls = {"count": 0}

    class FlakyLLM:
        def invoke(self, prompt):
            calls["count"] += 1
            if calls["count"] < 2:
                raise Exception("429 rate limit")
            return FakeResponse()

    monkeypatch.setattr(llc, "reasoning_llm", FlakyLLM())
    monkeypatch.setattr(llc.time, "sleep", lambda s: None)

    result = llc.call_llm("prompt", max_retries=2)
    assert result == "ok"
    assert calls["count"] == 2


def test_call_llm_does_not_retry_non_rate_limit_errors(monkeypatch):
    calls = {"count": 0}

    class BrokenLLM:
        def invoke(self, prompt):
            calls["count"] += 1
            raise ValueError("real bug")

    monkeypatch.setattr(llc, "reasoning_llm", BrokenLLM())

    try:
        llc.call_llm("prompt", max_retries=2)
        assert False, "should have raised"
    except ValueError:
        pass
    assert calls["count"] == 1


def test_call_llm_exhausts_retries_and_raises(monkeypatch):
    calls = {"count": 0}

    class AlwaysLimitedLLM:
        def invoke(self, prompt):
            calls["count"] += 1
            raise Exception("429 rate limit")

    monkeypatch.setattr(llc, "reasoning_llm", AlwaysLimitedLLM())
    monkeypatch.setattr(llc.time, "sleep", lambda s: None)

    try:
        llc.call_llm("prompt", max_retries=2)
        assert False, "should have raised after exhausting retries"
    except Exception as e:
        assert "429" in str(e)
    assert calls["count"] == 3
