import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tavily import TavilyClient

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

assert GROQ_KEY, "Missing GROQ_API_KEY"
assert GEMINI_KEY, "Missing GEMINI_API_KEY"
assert TAVILY_KEY, "Missing TAVILY_API_KEY"

reasoning_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
gemini_client = genai.Client(api_key=GEMINI_KEY)
tavily_client = TavilyClient(api_key=TAVILY_KEY)


def is_rate_limit_error(e: Exception) -> bool:
    """Shared across every agent now (moved here from web_search_agent on
    Day 18) — detects rate-limit errors by message content since exception
    TYPES differ between Groq/Gemini/Tavily SDKs, but the signal (429,
    'rate limit', 'RESOURCE_EXHAUSTED') is consistent enough to check as text."""
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg


def call_llm(prompt: str, max_retries: int = 2) -> str:
    """Shared LLM call wrapper, used by every agent. As of Day 18, retries
    automatically on rate-limit errors with exponential backoff — previously
    this retry logic only existed in web_search_agent, which meant Day 17's
    real run showed summarization and report generation failing on 429s with
    no recovery. Now every agent gets this for free by importing call_llm."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return reasoning_llm.invoke(prompt).content
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e) and attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            raise
    raise last_error


def research_with_fallback(query: str) -> dict:
    """Tries Gemini (grounded search + citations) first. Falls back to Tavily
    on Gemini-specific failures (its own API/client/server errors), not on
    ANY exception — narrowed on Day 18 per the gap flagged back on Day 5, so
    a real bug in this function's own code can't get silently misattributed
    to 'Gemini is down' and masked by the Tavily fallback."""
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
        print(f"Gemini unavailable ({type(e).__name__}), falling back to Tavily...")
        results = tavily_client.search(query=query, max_results=3)
        sources = [{"title": r["title"], "url": r["url"]} for r in results["results"]]
        answer = " ".join(r.get("content", "")[:300] for r in results["results"])
        return {"answer": answer, "sources": sources, "provider": "tavily"}
