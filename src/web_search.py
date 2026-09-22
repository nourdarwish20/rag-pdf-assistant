"""Simple, fast web search used only when the PDF cannot answer.

The PDF is always the primary source. This module is a fallback,
never a replacement.
"""

from typing import List, Dict

from src.config import (
    EXA_API_KEY,
    EXA_NUM_RESULTS,
    EXA_SEARCH_TYPE,
)


def search_web(query: str) -> List[Dict[str, str]]:
    """Return a few short web snippets for one query.

    Returns an empty list when Exa is not configured or the call
    fails, so the caller can keep the "not in the document" answer
    instead of showing an error.
    """

    if not EXA_API_KEY:
        return []

    try:
        from exa_py import Exa

        client = Exa(api_key=EXA_API_KEY)

        response = client.search(
            query,
            type=EXA_SEARCH_TYPE,           # "fast" - never a deep search
            num_results=EXA_NUM_RESULTS,
            contents={"highlights": True},  # short, relevance-sized snippets
        )

    except Exception:
        return []

    results = []

    for item in getattr(response, "results", []):

        highlights = getattr(item, "highlights", None) or []

        snippet = " ".join(highlights).strip()

        if not snippet:
            continue

        results.append({
            "title": getattr(item, "title", "") or "Untitled",
            "url": getattr(item, "url", "") or "",
            "snippet": snippet,
        })

    return results
