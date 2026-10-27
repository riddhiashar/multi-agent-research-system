import os
import time
import threading
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tavily import TavilyClient
from ddgs import DDGS

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

assert GROQ_KEY, "Missing GROQ_API_KEY"
assert GEMINI_KEY, "Missing GEMINI_API_KEY"
assert TAVILY_KEY, "Missing TAVILY_API_KEY"

reasoning_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
gemini_client = genai.Client(api_key=GEMINI_KEY)
tavily_client = TavilyClient(api_key=TAVILY_KEY)

# Concurrency cap (Day 19, per mentor feedback): regardless of how many
# sub-questions the Supervisor dispatches via Send, only this many search
# calls actually run at once. DEMO_MODE forces fully sequential (1) for
# maximum reliability during a live demo; otherwise capped at 2 as advised.
MAX_CONCURRENT_SEARCHES = 1 if DEMO_MODE else 2
_search_semaphore = threading.Semaphore(MAX_CONCURRENT_SEARCHES)


def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg


def call_llm(prompt: str, max_retries: int = 2) -> str:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return reasoning_llm.invoke(prompt).content
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e) and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_error


def _search_duckduckgo(query: str) -> dict:
    """Third fallback tier. Free, keyless, but unofficial (scrapes DuckDuckGo's
    HTML) — more prone to blocking than a real API, so it's a last resort,
    not a primary choice."""
    results = list(DDGS().text(query, max_results=3))
    sources = [
        {"title": r.get("title", ""), "url": r.get("href") or r.get("url", "")}
        for r in results
    ]
    answer = " ".join(r.get("body", "")[:300] for r in results)
    return {"answer": answer, "sources": sources, "provider": "duckduckgo"}


def research_with_fallback(query: str) -> dict:
    """Gemini -> Tavily -> DuckDuckGo. Concurrency-limited via a semaphore so
    no more than MAX_CONCURRENT_SEARCHES calls hit any provider at once,
    regardless of how many parallel graph branches call this simultaneously."""
    with _search_semaphore:
        try:
            resp = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=query,
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
            )
            sources = []
            if resp.candidates[0].grounding_metadata:
                sources = [
                    {"title": c.web.title, "url": c.web.uri}
                    for c in resp.candidates[0].grounding_metadata.grounding_chunks
                ]
            return {"answer": resp.text, "sources": sources, "provider": "gemini"}
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            print(f"⚠️ Gemini unavailable ({type(e).__name__}), falling back to Tavily...")

        try:
            results = tavily_client.search(query=query, max_results=3)
            sources = [{"title": r["title"], "url": r["url"]} for r in results["results"]]
            answer = " ".join(r.get("content", "")[:300] for r in results["results"])
            return {"answer": answer, "sources": sources, "provider": "tavily"}
        except Exception as e:
            print(f" Tavily unavailable ({type(e).__name__}), falling back to DuckDuckGo...")

        try:
            return _search_duckduckgo(query)
        except Exception as e:
            print(f"DuckDuckGo also unavailable ({type(e).__name__})")
            return {"answer": "", "sources": [], "provider": "none"}
