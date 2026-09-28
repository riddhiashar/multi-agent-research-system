import os
os.makedirs("docs", exist_ok=True)
F = "`" * 3

doc = f"""# System Architecture

## 1. Production architecture: Supervisor pattern

A single Supervisor/Planner agent owns all routing decisions. Specialist agents
never call each other directly. Each one does its job and returns control to the
Supervisor, which decides what runs next.

{F}mermaid
graph TD
    A[User Query] --> B[Supervisor/Planner Agent - Groq]
    B -->|needs research| C[Web Search Agent - Gemini, Tavily fallback]
    B -->|needs extraction| D[Data Extraction Agent - Groq]
    B -->|needs verification| E[Fact-Checking Agent - Groq]
    B -->|needs summary| F[Summarization Agent - Groq]
    B -->|ready to finalize| G[Report Generator Agent - Groq]
    C --> B
    D --> B
    E --> B
    F --> B
    G --> H[Final Report]
{F}

### Why Supervisor
- Fixed, known set of 5 specialists, so extra hierarchy adds complexity without benefit.
- All routing decisions live in one node, which makes debugging simple (one LangSmith trace).
- The research flow is naturally mostly linear (plan, research, verify, summarize, report).
- Guaranteed termination is easier to enforce (max-iteration cap in one place).

### Agent and model assignment
| Agent | Model / backend | Reason |
|---|---|---|
| Supervisor/Planner | Groq (qwen/qwen3.8-27b) | Fast reasoning, reliable tool calling |
| Web Search | Gemini grounded search, Tavily fallback | Native citations; fallback survives quota/outage |
| Data Extraction | Groq | Structured output from provided text |
| Fact-Checking | Groq | Cross-referencing claims across sources |
| Summarization | Groq | Condensing verified facts |
| Report Generator | Groq | Assembling final cited report |

## 2. Alternative pattern: Hierarchical (documented, not implemented)

A top supervisor delegates to team leads, and each lead manages its own specialists.

{F}mermaid
graph TD
    U[User Query] --> T[Top Supervisor]
    T --> R[Research Team Lead]
    T --> W[Writing Team Lead]
    R --> S1[Web Search Agent]
    R --> S2[Data Extraction Agent]
    R --> S3[Fact-Checking Agent]
    W --> S4[Summarization Agent]
    W --> S5[Report Generator Agent]
{F}

- **Pros:** scales to many agents; each lead only knows its own team.
- **Cons:** more LLM calls per decision, deeper call chains, harder tracing.
- **Switch to this if:** the agent count grows past roughly 8-10, or the Fact-Checker
  needs its own sub-agents (e.g. cross-referencing vs. source-credibility scoring).
  A small hierarchical pocket inside the Supervisor design is the likely first step.

## 3. Alternative pattern: Network (documented, not implemented)

No central authority. Any agent can hand off to any other.

{F}mermaid
graph LR
    A1[Web Search] <--> A2[Fact-Checker]
    A2 <--> A3[Summarizer]
    A1 <--> A4[Data Extraction]
    A4 <--> A5[Report Generator]
    A3 <--> A5
{F}

- **Pros:** no bottleneck; very flexible; agents can loop back freely.
- **Cons:** hard to debug ("who called whom?"), risk of infinite loops, no guaranteed termination.
- **Switch to this if:** the workflow becomes genuinely open-ended and non-linear, which
  it is not for this project. Kept as a learning reference only.

## 4. Known risks and mitigations
| Risk | Mitigation |
|---|---|
| Gemini free-tier 503/429/deprecations | Automatic Tavily fallback (built) |
| Groq model deprecation or tool-calling bugs | Central model config in tools/llm_clients.py; list live models before pinning |
| Supervisor loops forever | Max-iteration counter in state (added Day 7) |
| Fact-checking with a single source | Require 2+ independent sources per claim (Day 23) |
"""

with open("docs/architecture.md", "w", encoding="utf-8") as f:
    f.write(doc)
print("docs/architecture.md rewritten")