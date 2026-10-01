import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from google import genai
from google.genai import types
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


def call_llm(prompt: str) -> str:
    """Shared wrapper around the reasoning LLM call. Every agent should import and
    use THIS function rather than calling reasoning_llm.invoke directly — it keeps
    the LangChain/Pydantic object out of agent code entirely, so tests can mock a
    plain function instead of monkeypatching a Pydantic model (which Pydantic v2
    blocks for undeclared attributes, as hit on Day 9)."""
    return reasoning_llm.invoke(prompt).content


def research_with_fallback(query: str) -> dict:
    """Tries Gemini (grounded search + citations) first. Falls back to Tavily
    automatically on any Gemini failure, so the pipeline never fully blocks
    on one provider's outage."""
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
    except Exception as e:
        print(f" Gemini unavailable ({type(e).__name__}), falling back to Tavily...")
        results = tavily_client.search(query=query, max_results=3)
        sources = [{"title": r["title"], "url": r["url"]} for r in results["results"]]
        answer = " ".join(r.get("content", "")[:300] for r in results["results"])
        return {"answer": answer, "sources": sources, "provider": "tavily"}
