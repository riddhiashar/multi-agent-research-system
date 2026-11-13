import uuid
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from tools.metrics import RunMetrics, peak_tokens_in_window, evidence_stats


def _resp(out, inp=10, reasoning=0):
    msg = AIMessage(content="x", usage_metadata={
        "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
        "output_token_details": {"reasoning": reasoning}})
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_peak_window_math():
    assert peak_tokens_in_window([(0, 400), (30, 400), (59, 100), (200, 900)], 60) == 900
    assert peak_tokens_in_window([(0, 400), (30, 400), (59, 100), (100, 900)], 60) == 1000  # 59s + 100s fit in one window
    assert peak_tokens_in_window([(0, 500), (61, 500)], 60) == 500
    assert peak_tokens_in_window([], 60) == 0


def test_llm_call_attributed_to_node_with_reasoning():
    m, rid = RunMetrics(), uuid.uuid4()
    m.on_chat_model_start({}, [], run_id=rid, metadata={"langgraph_node": "summarization"})
    m.on_llm_end(_resp(300, reasoning=120), run_id=rid)
    n = m.summary()["nodes"]["summarization"]
    assert n["output"] == 300 and n["reasoning"] == 120 and n["calls"] == 1


def test_parallel_calls_keep_separate_attribution():
    m = RunMetrics()
    a, b = uuid.uuid4(), uuid.uuid4()
    m.on_chat_model_start({}, [], run_id=a, metadata={"langgraph_node": "web_search_single"})
    m.on_chat_model_start({}, [], run_id=b, metadata={"langgraph_node": "web_search_single"})
    m.on_llm_end(_resp(50), run_id=a)
    m.on_llm_end(_resp(70), run_id=b)
    assert m.summary()["nodes"]["web_search_single"]["output"] == 120


def test_rate_limit_error_counted():
    m, rid = RunMetrics(), uuid.uuid4()
    m.on_chat_model_start({}, [], run_id=rid, metadata={"langgraph_node": "fact_checking"})
    m.on_llm_error(Exception("Error code: 429 - rate_limit_exceeded"), run_id=rid)
    assert m.summary()["rate_limit_errors"] == 1


def test_node_latency_only_counts_the_node_chain():
    m, rid, inner = RunMetrics(), uuid.uuid4(), uuid.uuid4()
    md = {"langgraph_node": "supervisor"}
    m.on_chain_start({}, {}, run_id=rid, metadata=md, name="supervisor")
    m.on_chain_start({}, {}, run_id=inner, metadata=md, name="RunnableSequence")
    m.on_chain_end({}, run_id=inner)
    m.on_chain_end({}, run_id=rid)
    assert m.summary()["nodes"]["supervisor"]["node_runs"] == 1


def test_evidence_stats_distribution_and_merge_count():
    state = {"extracted_facts": [1, 2, 3, 4],
             "fact_check_results": [
                 {"sources": [1, 2], "independent_sources": 1, "status": "UNVERIFIED"},
                 {"sources": [1], "independent_sources": 1, "status": "UNVERIFIED"},
                 {"sources": [1, 2, 3], "independent_sources": 2, "status": "CONFIRMED", "merged_count": 2}]}
    ev = evidence_stats(state)
    assert ev["cited_per_claim"] == {1: 1, 2: 1, 3: 1}
    assert ev["independent_per_claim"] == {1: 2, 2: 1}
    assert ev["claims_that_were_merged"] == 1 and ev["extracted_claims"] == 4
