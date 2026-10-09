# Parallel Execution Decision (Day 15, implemented Day 17)

## Benchmark results
Sequential vs. parallel ThreadPoolExecutor on real search_one calls showed a
measurable speedup (see notebook output) — the gain scales with the number of
sub-questions per plan, which is typically 2-5 for this system.

## Chosen approach for Day 17
Use ThreadPoolExecutor inside web_search_node (not asyncio, not LangGraph's
Send API) because:
1. Existing agent functions are synchronous — ThreadPoolExecutor parallelizes
   them with zero rewrites, whereas asyncio would require rewriting every
   agent as async, a much bigger and riskier change.
2. The work is I/O-bound (waiting on Gemini/Tavily HTTP calls), which is
   exactly what threads handle well in Python despite the GIL — the GIL
   releases during I/O wait.
3. LangGraph's Send API (fan-out to multiple node instances) is the more
   "idiomatic" LangGraph approach, and is worth learning, but adds real
   complexity to the graph's conditional routing. Given Supervisor already
   does deterministic routing (Day 13), introducing Send-based fan-out risks
   destabilizing something that currently works reliably. ThreadPoolExecutor
   inside the existing node is lower-risk for the same benefit.

## What will change in web_search_node (Day 17)
- Replace `[search_one(q) for q in queries]` with the `search_many_parallel`
  pattern benchmarked today.
- No change needed to search_one, data_extraction_agent, or any downstream
  agent — this is purely an internal change to how web_search_node gathers
  results, invisible to the rest of the pipeline.

## Verified today
- Parallel execution produces identical coverage to sequential (same queries,
  no lost/duplicate work) — see Cell 4.
- Thread-safety audit of shared clients (Tavily, Gemini, logger) — no unsafe
  shared mutable state identified.


## Follow-up: failure is intermittent, not consistent
Re-ran the same 4 queries under parallel execution a third time: all 4 succeeded
(0 failures). Combined with the earlier two runs (1 failure each), this confirms
the failure mode is probabilistic — consistent with hitting a rate limit right at
its threshold under simultaneous load, not a structural bug in the parallel
execution code itself. This is good news for Day 17: retry-with-backoff (already
present in search_one's design pattern from Day 8, just needs to trigger reliably
on rate-limit-specific errors per the Day 18 plan) should be sufficient to absorb
this, rather than requiring a fundamentally different concurrency approach.


## Day 17 follow-up: rate-limit pressure shifted downstream
Real end-to-end run after implementing Send-based parallel web search: search
stage itself was flawless (4/4 ok, 0 errors, 0 retries needed). However,
summarization (3/4 failed) and report generation (narrative failed, used
fallback_template) BOTH hit Groq 429 errors in the same run.

Root cause: parallel search finishes all sub-questions almost simultaneously,
which means extraction and fact-checking for all of them complete close
together too, concentrating summarization/report LLM calls into a tighter
time window than the old sequential pipeline had. The retry logic added in
Day 17 only covers search_one (via web_search_agent) — summarization_agent
and report_generator_agent still call call_llm directly with zero retry.

All fail-safes worked correctly: failed summaries still returned real content
with fact counts, the report still completed via fallback_template. Nothing
crashed. But this confirms the gap needs fixing for the WHOLE pipeline, not
just search.

Action for Day 18: move retry logic from web_search_agent into the shared
call_llm wrapper in tools/llm_clients.py, so every agent gets rate-limit
retry for free rather than reimplementing it per-agent.
