# Week 2 Review (Day 14)

## Built
All five specialist agents, plus the Supervisor tying them together into a fully
autonomous graph:
- Web Search (Gemini grounded search, Tavily fallback)
- Data Extraction (atomic fact extraction, JSON-structured)
- Fact-Checking (batched, contradiction detection, fail-safe to UNVERIFIED)
- Summarization (per-sub-topic, explicitly surfaces disputes rather than smoothing them)
- Report Generator (LLM narrative + deterministic, un-fail-able confidence appendix)
- Supervisor (deterministic rule-based routing for the fixed pipeline order, one LLM
  call only for initial query decomposition, loop guard at MAX_ITERATIONS)

## Verified working end-to-end
A single bare query, with zero manual intervention, now produces a complete,
dispute-aware report. Confirmed via a real run (solid-state EV battery safety query):
6 iterations, 4 sub-questions, 30 facts extracted, report generated, LangSmith trace
confirmed the full supervisor-routed loop visually.

## Real production finding
During the live end-to-end run, Groq rate-limited a fact-checking batch
(429 on output tokens/minute). The fail-safe fired correctly: that batch defaulted
to UNVERIFIED rather than crashing or silently marking facts CONFIRMED. This is the
exact behavior the fail-safe was designed for â€” validated under real conditions,
not just mocked tests.

## Known gap, scheduled not yet fixed
No retry/backoff specifically for rate-limit errors (as opposed to generic failures)
in the fact-checking batch calls. Scheduled for Day 18 (error handling pass). Given
free-tier rate limits, this will likely recur, especially once Day 17 adds parallel
execution and increases concurrent API calls.

## Design pattern established and reused across all 5 agents
Every node logs to model_usage_log on EVERY path (success, empty input, and error),
not just the happy path. This is what makes the Supervisor's deterministic routing
possible â€” it can reliably tell "ran" from "hasn't run yet" by reading the log.

## Test coverage
41 tests passing: unit tests per agent (mocking call_llm or whole node functions)
plus 3 integration tests verifying the graph's wiring, routing order, and loop-guard
termination independent of any agent's internal logic.

## Week 3 plan
Parallel execution of sub-question chains (Day 17), error handling/retry pass
including the rate-limit gap above (Day 18), human-in-the-loop interrupt before
report generation (Day 19), checkpointing/memory across sessions (Day 20).
