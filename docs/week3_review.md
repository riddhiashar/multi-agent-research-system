# Week 3 Checkpoint (Day 21)

## Tested
3 end-to-end queries through the full autonomous graph: a 3-way comparison
(battery recycling), a narrow technical question (lithium plating causes),
and a deliberately vague/opinion-adjacent query (is battery tech improving
fast enough).

## Results
| Query | Time | Iterations | Errors | Report |
|---|---|---|---|---|
| Recycling comparison | 92.4s | 6 | 0 | 5003 chars (3 summaries, not 4 — see issue below) |
| Lithium plating | 162.1s | 6 | 0 | 7110 chars (4 summaries, mostly correctly hedged UNVERIFIED) |
| Vague/opinion query | 175.1s | 6 | 0 | 6609 chars (4 summaries, correctly mostly UNVERIFIED) |

Timing climbed steadily across the session (92s -> 162s -> 175s), consistent
with accumulating pressure against Groq's rolling 60s output-token budget
across consecutive runs in one sitting.

## Issue found: sub-topics silently vanish when extraction finds zero facts
Query 1's plan had 4 sub-questions. One ("environmental impacts... carbon
footprint") returned facts_found=0 at extraction. Because summarization_agent
groups by sub_question keys PRESENT in fact_check_results, and that
sub-question never produced any fact_check_results entries, it never
appeared in summarization at all -- not even as a "no information found"
summary. The Day 11 empty-facts fallback only protects against an empty
LIST being passed in, not a sub-topic being entirely ABSENT. Net effect:
the final report silently covers 3 of 4 planned angles with no indication
a 4th was ever attempted.

This is a real gap, not a crash -- nothing errored, which is exactly why
it went unnoticed until this checkpoint. Candidate fix (not done today):
summarization_node should iterate over the ORIGINAL plan, not just the
keys present in fact_check_results, and explicitly emit a "no information
found" summary for any planned sub-question with zero results.

## Good signals
- Query 3 (vague/opinion query): Supervisor correctly decomposed a subjective
  question into concrete, researchable sub-questions rather than inviting
  ungrounded opinion. Resulting confidence breakdown was mostly UNVERIFIED --
  an honest reflection of the question's inherent uncertainty, not false
  confidence.
- Query 3 also hit a real mid-run RateLimitError that was fully absorbed by
  the fail-safe -- errors: [] in the final state despite the failure,
  confirming Day 18's fix holds under genuinely new, unplanned load.

## Week 3 recap
- Day 15: benchmarked parallel vs sequential search (2-2.8x speedup)
- Day 16: verified LangGraph reducer/Send mechanics against real API calls
- Day 17: shipped Send-based parallel web search + rate-limit retry
- Day 18: centralized retry logic in call_llm, fixed downstream rate-limit gap
- Day 19: DuckDuckGo fallback tier + concurrency semaphore + DEMO_MODE
  (per mentor feedback), human-in-the-loop interrupt_before
- Day 20: SqliteSaver persistent checkpointing, verified across simulated
  process restarts and independent threads
- Day 21: 3 real end-to-end stress-test queries; found and documented the
  silent-sub-topic-drop gap above

## Carried into Week 4
- The silent-drop gap above is a good candidate for Day 22's evaluation
  harness: "does the report account for every planned sub-question" is a
  concrete, testable quality criterion, not just a one-off bug to patch.
- Groq's 1000 output-token/min limit remains the binding constraint under
  real multi-query sessions; fail-safes handle it correctly but it's worth
  keeping in mind for demo pacing (per mentor's earlier feedback).
