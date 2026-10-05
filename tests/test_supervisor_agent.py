import agents.supervisor_agent as sup


def test_plan_generation_fallback_on_failure(monkeypatch):
    def boom(p):
        raise RuntimeError("down")
    monkeypatch.setattr(sup, "call_llm", boom)
    plan = sup.generate_plan("my query")
    assert plan == ["my query"]


def test_routes_to_planning_when_no_plan(monkeypatch):
    monkeypatch.setattr(sup, "generate_plan", lambda q: ["sub-q 1", "sub-q 2"])
    state = {"query": "q", "iteration_count": 0, "plan": [], "model_usage_log": []}
    out = sup.supervisor_node(state)
    assert out["next_agent"] == "web_search"
    assert out["plan"] == ["sub-q 1", "sub-q 2"]


def test_routes_through_full_pipeline_in_order():
    pipeline_agents = [a for a, _ in sup.PIPELINE]
    state = {"query": "q", "iteration_count": 0, "plan": ["x"], "model_usage_log": []}

    for i, (agent_name, node_name) in enumerate(sup.PIPELINE):
        out = sup.supervisor_node(state)
        assert out["next_agent"] == node_name, f"expected {node_name} at step {i}, got {out['next_agent']}"
        state["model_usage_log"].append({"agent": agent_name})
        state["iteration_count"] = out["iteration_count"]

    final = sup.supervisor_node(state)
    assert final["next_agent"] == "done"


def test_loop_guard_triggers():
    state = {"query": "q", "iteration_count": sup.MAX_ITERATIONS, "plan": ["x"], "model_usage_log": []}
    out = sup.supervisor_node(state)
    assert out["next_agent"] == "failed"


def test_route_from_supervisor_reads_next_agent():
    state = {"next_agent": "fact_checking"}
    assert sup.route_from_supervisor(state) == "fact_checking"
