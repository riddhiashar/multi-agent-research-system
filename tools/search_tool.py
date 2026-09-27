import time
from langchain_core.tools import tool
from tools.llm_clients import tavily_client


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic. Use this when you need
    up-to-date facts, news, or information that might not be in your training data."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = tavily_client.search(query=query, max_results=3)
            formatted = "\n\n".join(
                f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:300]}"
                for r in results["results"]
            )
            return formatted
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f" Search failed ({e.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"Search failed after {max_retries} attempts: {e}"
