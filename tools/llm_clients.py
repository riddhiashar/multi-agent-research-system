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

from tools import cache
from tools.diversity import (select_diverse, add_new_domains, domain_count,
                             MAX_SOURCES_PER_QUERY, MIN_DIVERSE_DOMAINS)
from tools.evidence import source_domain

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

MAX_CONCURRENT_SEARCHES = 1 if DEMO_MODE else 2
_search_semaphore = threading.Semaphore(MAX_CONCURRENT_SEARCHES)

# Day 24: fetch a wider candidate pool, but KEEP only distinct domains.
CANDIDATE_POOL = 8


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


def _mark_snippets(snippets) -> str:
    """Tags each snippet [S1], [S2]... so extraction can say which source states a claim (Day 23)."""
    return " ".join(f"[S{i}] {text[:300]}" for i, text in enumerate(snippets, start=1))


def _package(results, provider: str) -> dict:
    """results: [{"title","url","content"}] -> the dict the agents expect."""
    sources = [{"id": i, "title": r.get("title", ""), "url": r.get("url", "")}
               for i, r in enumerate(results, start=1)]
    answer = _mark_snippets([r.get("content", "") for r in results])
    return {"answer": answer, "sources": sources, "provider": provider}


def _ddg_candidates(query: str, n: int = CANDIDATE_POOL) -> list:
    raw = list(DDGS().text(query, max_results=n))
    return [{"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""),
             "content": r.get("body", "")} for r in raw]


def _search_duckduckgo(query: str) -> dict:
    picked = select_diverse(_ddg_candidates(query), max_total=MAX_SOURCES_PER_QUERY)
    return _package(picked, "duckduckgo")


def _research_uncached(query: str) -> dict:
    """Gemini -> Tavily -> DuckDuckGo, concurrency-limited by a semaphore."""
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
                    {"id": i, "title": c.web.title, "url": c.web.uri}
                    for i, c in enumerate(resp.candidates[0].grounding_metadata.grounding_chunks, start=1)
                ]
            # Gemini's answer is one synthesized text, so it has no per-snippet tags.
            return {"answer": resp.text, "sources": sources, "provider": "gemini"}
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            print(f"⚠️ Gemini unavailable ({type(e).__name__}), falling back to Tavily...")

        try:
            raw = tavily_client.search(query=query, max_results=CANDIDATE_POOL)["results"]
            candidates = [{"title": r.get("title", ""), "url": r.get("url", ""),
                           "content": r.get("content", "")} for r in raw]
            picked = select_diverse(candidates, max_total=MAX_SOURCES_PER_QUERY)
            provider = "tavily"
            # Too few distinct domains? Top up from DuckDuckGo, but only with domains we don't have.
            if domain_count(picked) < MIN_DIVERSE_DOMAINS:
                try:
                    topped = add_new_domains(picked, _ddg_candidates(query), MAX_SOURCES_PER_QUERY)
                    if len(topped) > len(picked):
                        picked, provider = topped, "tavily+duckduckgo"
                except Exception as e:
                    print(f"⚠️ DuckDuckGo top-up skipped ({type(e).__name__})")
            return _package(picked, provider)
        except Exception as e:
            print(f"⚠️ Tavily unavailable ({type(e).__name__}), falling back to DuckDuckGo...")

        try:
            return _search_duckduckgo(query)
        except Exception as e:
            print(f"⚠️ DuckDuckGo also unavailable ({type(e).__name__})")
            return {"answer": "", "sources": [], "provider": "none"}


def research_with_fallback(query: str) -> dict:
    """Cached wrapper (Day 25). Caching is opt-in via SEARCH_CACHE=true; failures are never cached."""
    cached = cache.get("research", query)
    if cached is not None:
        return {**cached, "cached": True}
    result = _research_uncached(query)
    if result.get("provider") != "none" and result.get("answer"):
        cache.put("research", query, value=result)
    return result


def search_candidates(query: str, exclude_domains=(), max_total: int = MAX_SOURCES_PER_QUERY) -> list:
    """Raw, full-text search results for verifying one claim (Day 25): Tavily, then DuckDuckGo.
    Results from `exclude_domains` are dropped; one result per remaining domain is kept."""
    exclude = {d.lower() for d in exclude_domains}
    cached = cache.get("candidates", query, ",".join(sorted(exclude)))
    if cached is not None:
        return cached

    def usable(results):
        keep = [r for r in results
                if source_domain({"url": r.get("url", ""), "title": r.get("title", "")}) not in exclude]
        return select_diverse(keep, max_total=max_total) if keep else []

    found = []
    with _search_semaphore:
        try:
            kwargs = {"query": query, "max_results": CANDIDATE_POOL}
            if exclude:
                kwargs["exclude_domains"] = sorted(exclude)
            raw = tavily_client.search(**kwargs)["results"]
            found = usable([{"title": r.get("title", ""), "url": r.get("url", ""),
                             "content": r.get("content", "")} for r in raw])
        except Exception as e:
            print(f" Verification search via Tavily failed ({type(e).__name__}), trying DuckDuckGo...")
        if not found:
            try:
                found = usable(_ddg_candidates(query))
            except Exception as e:
                print(f"Verification search via DuckDuckGo failed ({type(e).__name__})")
    if found:
        cache.put("candidates", query, ",".join(sorted(exclude)), value=found)
    return found
