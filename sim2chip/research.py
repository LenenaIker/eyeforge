"""Web research for the interview (Tavily): facts Nemotron does not know or should not guess, such as the real
size of an object or a part, its English or technical name, or what it looks like. Stdlib only.

Configuration (environment or the repo's .env): TAVILY_API_KEY. Without it the interview has no research action.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

URL = "https://api.tavily.com/search"
USD_PER_SEARCH = 0.008  # one credit per basic search, pay as you go (the usage panel prices it)
MAX_RESULTS = 5


class ResearchError(RuntimeError):
    pass


def search(query: str, key: str, timeout: float = 30.0) -> dict:
    """Tavily's short answer and its sources: {"answer", "sources": [{"title", "url", "content"}]}."""
    body = {"query": query[:400], "search_depth": "basic", "max_results": MAX_RESULTS, "include_answer": True,
            "chunks_per_source": 2}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ResearchError(f"Tavily: HTTP {e.code} {e.read().decode(errors='replace')[:200]}") from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise ResearchError(f"Tavily: {e}") from None
    return {"answer": data.get("answer") or "",
            "sources": [{"title": s.get("title", ""), "url": s.get("url", ""), "content": (s.get("content") or "")[:500]}
                        for s in data.get("results") or []]}


def searcher():
    """query -> results, or None when there is no Tavily key."""
    from .config import env
    key = env("TAVILY_API_KEY")
    return (lambda query: search(query, key)) if key else None
