# State Management Deep-Dive (Day 16)

## Verified
1. `Annotated[list, add]` correctly merges writes from parallel branches run in
   the same LangGraph superstep — confirmed with 3 synthetic branches, all 3
   writes present in the final state, none lost or overwritten.
2. A plain `list` field (no reducer) causes LangGraph to raise an error on
   concurrent writes rather than silently dropping one — this is the SAFE
   failure mode, not a bug. It forces the developer to declare a reducer
   rather than getting silent data loss.
3. LangGraph's `Send` API is the idiomatic fan-out primitive — dispatches N
   parallel branches, each with its own isolated state slice (not the full
   parent state), results merge back via the reducer. Verified with both
   synthetic and REAL search_one calls (3 real parallel searches, all 3
   results correctly merged).
4. Merge ORDER is not guaranteed across parallel branches. Verified this does
   NOT break anything downstream: data_extraction_agent and all subsequent
   agents read fields (sub_question, answer, sources) from each result dict
   directly, never by list position/index. This was an accidental but
   important design correctness from Day 9 onward.

## Decision for Day 17
Use `Send` (LangGraph's native fan-out), not Day 15's ThreadPoolExecutor
approach, for the actual web_search_node implementation. Reasoning:
- Send is the idiomatic LangGraph mechanism, integrates with the existing
  reducer-based state schema with zero extra glue code.
- ThreadPoolExecutor (Day 15) proved the SPEEDUP is real and worth pursuing,
  but Send achieves the same parallelism while staying inside LangGraph's
  state/checkpointing/tracing model, rather than doing ad-hoc threading
  inside a single node that LangGraph can't see into.
- Revising Day 15's conclusion: Send over ThreadPoolExecutor for the final
  implementation, now that this deep-dive confirmed Send's mechanics are
  correct, safe, and already proven against real API calls.

## Remaining risk carried into Day 17
Send-based fan-out will still make N simultaneous API calls, same as Day 15's
ThreadPoolExecutor did — the intermittent rate-limit failures found on Day 15
are not specific to ThreadPoolExecutor and will still need Day 18's retry work
regardless of which mechanism is used.
