"""Run-level observability: tokens, latency and errors per node, plus evidence stats."""
import json
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit")


def peak_tokens_in_window(events, window=60.0):
    """events: [(timestamp, output_tokens)]. Max token sum inside any `window`-second span."""
    events = sorted(events)
    best = cur = lo = 0
    for hi in range(len(events)):
        cur += events[hi][1]
        while events[hi][0] - events[lo][0] > window:
            cur -= events[lo][1]
            lo += 1
        best = max(best, cur)
    return best


class RunMetrics(BaseCallbackHandler):
    """Pass as config={"callbacks": [metrics]} to graph.invoke."""

    def __init__(self):
        self._lock = threading.Lock()
        self._llm_runs = {}    # run_id -> (node, start)
        self._node_runs = {}   # run_id -> (node, start)
        self.llm_calls = []
        self.node_times = defaultdict(list)
        self.node_errors = Counter()
        self.output_events = []   # (timestamp, output_tokens)
        self.error_log = []       # (node, kind, message) for every failed LLM call
        self.started = time.time()

    @staticmethod
    def _node(metadata):
        return (metadata or {}).get("langgraph_node", "unknown")

    # ---- LLM calls ----
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        with self._lock:
            self._llm_runs[run_id] = (self._node(metadata), time.time())

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        with self._lock:
            self._llm_runs[run_id] = (self._node(metadata), time.time())

    def on_llm_end(self, response, *, run_id, **kwargs):
        now = time.time()
        with self._lock:
            node, start = self._llm_runs.pop(run_id, ("unknown", now))
        inp = out = reasoning = 0
        try:
            msg = response.generations[0][0].message
            um = getattr(msg, "usage_metadata", None) or {}
            inp = um.get("input_tokens", 0)
            out = um.get("output_tokens", 0)
            reasoning = (um.get("output_token_details") or {}).get("reasoning", 0)
        except (IndexError, AttributeError):
            pass
        if not out:
            tu = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            inp = inp or tu.get("prompt_tokens", 0)
            out = tu.get("completion_tokens", 0)
        with self._lock:
            self.llm_calls.append({"node": node, "input": inp, "output": out,
                                   "reasoning": reasoning, "seconds": now - start,
                                   "error": None})
            self.output_events.append((now, out))

    def on_llm_error(self, error, *, run_id, **kwargs):
        now = time.time()
        with self._lock:
            node, start = self._llm_runs.pop(run_id, ("unknown", now))
        msg = str(error).lower()
        kind = "rate_limit" if any(m in msg for m in RATE_LIMIT_MARKERS) else "other"
        self.error_log.append((node, kind, str(error)[:300]))
        with self._lock:
            self.llm_calls.append({"node": node, "input": 0, "output": 0,
                                   "reasoning": 0, "seconds": now - start,
                                   "error": kind})

    # ---- node latency (a node is a chain whose name equals its langgraph_node) ----
    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node and kwargs.get("name") == node:
            with self._lock:
                self._node_runs[run_id] = (node, time.time())

    def _finish_node(self, run_id, failed=False):
        with self._lock:
            entry = self._node_runs.pop(run_id, None)
            if entry:
                node, start = entry
                self.node_times[node].append(time.time() - start)
                if failed:
                    self.node_errors[node] += 1

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._finish_node(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._finish_node(run_id, failed=True)

    # ---- summary ----
    def summary(self):
        nodes = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "reasoning": 0,
                                     "rate_limited": 0, "other_errors": 0,
                                     "llm_seconds": 0.0, "max_call_output": 0})
        for c in self.llm_calls:
            n = nodes[c["node"]]
            n["calls"] += 1
            n["input"] += c["input"]
            n["output"] += c["output"]
            n["reasoning"] += c["reasoning"]
            n["llm_seconds"] += c["seconds"]
            n["max_call_output"] = max(n["max_call_output"], c["output"])
            if c["error"] == "rate_limit":
                n["rate_limited"] += 1
            elif c["error"]:
                n["other_errors"] += 1
        for node, times in self.node_times.items():
            nodes[node]["node_runs"] = len(times)
            nodes[node]["node_seconds"] = sum(times)
            nodes[node]["slowest_run"] = max(times)
            nodes[node]["node_errors"] = self.node_errors[node]
        return {
            "total_seconds": time.time() - self.started,
            "total_output_tokens": sum(c["output"] for c in self.llm_calls),
            "total_reasoning_tokens": sum(c["reasoning"] for c in self.llm_calls),
            "peak_output_tokens_60s": peak_tokens_in_window(self.output_events),
            "rate_limit_errors": sum(1 for c in self.llm_calls if c["error"] == "rate_limit"),
            "nodes": {k: dict(v) for k, v in nodes.items()},
        }


def evidence_stats(state):
    """Evidence picture from the final state: claims before/after merging, cited vs
    independent sources per claim, and search-provider mix."""
    cited, independent, status = Counter(), Counter(), Counter()
    results = state.get("fact_check_results", []) or []
    for f in results:
        cited[len(f.get("sources") or f.get("source_ids") or [])] += 1
        independent[f.get("independent_sources", "n/a")] += 1
        status[f.get("status") or f.get("verdict") or "n/a"] += 1
    extracted = state.get("extracted_facts", []) or []
    merged_groups = [f for f in results if f.get("merged_count", 1) > 1]
    providers = Counter(r.get("provider", "n/a") for r in (state.get("search_results", []) or [])
                        if isinstance(r, dict))
    return {"extracted_claims": len(extracted),
            "claims_after_merge_expand": len(results),
            "claims_that_were_merged": len(merged_groups),
            "cited_per_claim": dict(sorted(cited.items())),
            "independent_per_claim": dict(independent),
            "status": dict(status),
            "search_providers": dict(providers),
            "claims_verified_by_search": sum(1 for f in results if f.get("verified_by_search"))}


def save_run(metrics, state, label="run"):
    Path("data/runs").mkdir(parents=True, exist_ok=True)
    path = Path("data/runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{label}.json"
    path.write_text(json.dumps({"metrics": metrics.summary(),
                                "evidence": evidence_stats(state)}, indent=2, default=str))
    return path


def print_run_report(metrics, state):
    s = metrics.summary()
    print(f"Total: {s['total_seconds']:.0f}s | output tokens: {s['total_output_tokens']} "
          f"(reasoning: {s['total_reasoning_tokens']}) | 429s: {s['rate_limit_errors']}")
    print(f"Peak output tokens in any 60s window: {s['peak_output_tokens_60s']} (Groq limit: 1000)\n")
    print(f"{'node':<22}{'calls':>6}{'out':>7}{'reason':>8}{'max/call':>9}{'429':>5}{'node_s':>8}")
    for name, n in s["nodes"].items():
        print(f"{name:<22}{n['calls']:>6}{n['output']:>7}{n['reasoning']:>8}"
              f"{n['max_call_output']:>9}{n['rate_limited']:>5}{n.get('node_seconds', 0):>8.1f}")
    ev = evidence_stats(state)
    print("\nEvidence")
    print(f"  claims extracted: {ev['extracted_claims']} -> after merge/expand: "
          f"{ev['claims_after_merge_expand']} ({ev['claims_that_were_merged']} rows from merged groups)")
    print("  cited sources per claim:     ", ev["cited_per_claim"])
    print("  independent domains per claim:", ev["independent_per_claim"])
    print("  status:                       ", ev["status"])
    print("  search providers:             ", ev["search_providers"])
    print("  claims given extra sources by verification search:", ev["claims_verified_by_search"])
    for node, kind, msg in metrics.error_log:
        print(f"  LLM error in {node} ({kind}): {msg}")
