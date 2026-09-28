# Multi-Agent Research & Intelligence System

A LangGraph-based multi-agent system with a Supervisor/Planner Agent coordinating
specialized agents (Web Search, Data Extraction, Fact-Checking, Summarization,
Report Generation) to research a topic and produce a cited, structured report.

## Architecture
See `docs/architecture.md` for the full diagram and design decisions.

- **Supervisor/Planner, Fact-Checker, Summarizer, Report Generator** — Groq (qwen/qwen3.8-27b)
- **Web Search Agent** — Gemini (grounded search) with automatic Tavily fallback

## Setup
1. `python -m venv venv && source venv/bin/activate` (or `venv\\Scripts\\activate` on Windows)
2. `pip install -r requirements.txt`
3. Create a `.env` file with `GROQ_API_KEY`, `GEMINI_API_KEY`, `TAVILY_API_KEY`, and optionally `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`

## Project structure
- `agents/` — individual agent logic
- `tools/` — reusable tools and LLM client setup
- `graph/` — LangGraph orchestration
- `state/` — shared state schema
- `tests/` — unit and integration tests
- `docs/` — architecture notes and diagrams


## Design decisions
- **Production architecture:** Supervisor pattern (see `docs/architecture.md`).
- **Hierarchical and Network patterns** are documented there as alternatives, with the
  conditions under which we would switch. They are not implemented.
