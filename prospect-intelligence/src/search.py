"""
src/search.py — Tavily web search integration.

Responsibilities:
- Call Tavily to discover relevant public web pages for a search query.
- Return structured SearchResult objects.
- Deduplicate results by canonical domain.
- Handle API failures, rate limits and empty results with bounded retries.
- Never fabricate results or URLs.
- Completely independent of Streamlit or any UI framework.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tavily import TavilyClient

from src.schemas import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_domain(url: str) -> str:
    """Extract and normalise the domain from a URL string."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().strip()
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split(":")[0]  # strip port
        return domain
    except Exception:
        return url.lower().strip()


def _is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return result.scheme in {"http", "https"} and bool(result.netloc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

def _is_transient(exc: BaseException) -> bool:
    """True for rate-limit or server-side errors that are worth retrying."""
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "429", "503", "timeout", "connection"))


# ---------------------------------------------------------------------------
# SearchClient
# ---------------------------------------------------------------------------

class SearchClient:
    """
    Wraps the Tavily API.  One instance is reused across multiple queries.

    Args:
        api_key:    Tavily API key (loaded from config, never hardcoded).
        timeout:    Per-request timeout in seconds.
        max_retries: Maximum retry attempts for transient failures.
    """

    def __init__(self, api_key: str, timeout: int = 30, max_retries: int = 3) -> None:
        self._client = TavilyClient(api_key=api_key)
        self._timeout = timeout
        self._max_retries = max_retries

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "basic",
    ) -> list[SearchResult]:
        """
        Run a Tavily search and return a deduplicated list of SearchResult objects.

        Args:
            query:        Natural language search query.
            max_results:  Maximum number of results to return (1–20).
            search_depth: "basic" (faster, cheaper) or "advanced".

        Returns:
            List of SearchResult objects, one per unique domain.
            May be empty if Tavily returns nothing or all results are invalid.

        Raises:
            RuntimeError: On permanent API failure after all retries exhausted.
        """
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        max_results = max(1, min(20, max_results))

        logger.info("Tavily search: %r (max=%d, depth=%s)", query, max_results, search_depth)

        raw_results = self._call_tavily(query, max_results, search_depth)
        return self._parse_and_deduplicate(raw_results)

    def _call_tavily(
        self,
        query: str,
        max_results: int,
        search_depth: str,
    ) -> list[dict]:
        """
        Call Tavily with retry/backoff.  Returns the raw result list.
        """
        attempts = 0
        last_exc: Exception | None = None

        while attempts < self._max_retries:
            try:
                response = self._client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=False,
                    include_raw_content=False,
                )
                results = response.get("results", [])
                logger.info("Tavily returned %d raw results.", len(results))
                return results

            except Exception as exc:
                attempts += 1
                last_exc = exc

                if not _is_transient(exc):
                    logger.error("Permanent Tavily error after %d attempt(s): %s", attempts, exc)
                    raise RuntimeError(f"Tavily search failed (permanent): {exc}") from exc

                if attempts < self._max_retries:
                    wait = 2 ** attempts  # 2, 4, 8 seconds
                    logger.warning(
                        "Tavily transient error (attempt %d/%d), retrying in %ds: %s",
                        attempts, self._max_retries, wait, exc,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Tavily search failed after {self._max_retries} attempts: {last_exc}"
        )

    def _parse_and_deduplicate(self, raw_results: list[dict]) -> list[SearchResult]:
        """
        Parse raw Tavily result dicts into SearchResult objects.
        Deduplicate: keep the first (highest-ranked) result per domain.
        Skip results with invalid or missing URLs.
        """
        seen_domains: set[str] = set()
        results: list[SearchResult] = []
        retrieved_at = datetime.now(timezone.utc)

        for item in raw_results:
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("content") or item.get("snippet") or "").strip()

            if not url or not _is_valid_url(url):
                logger.debug("Skipping invalid URL: %r", url)
                continue

            if not title:
                title = urlparse(url).netloc or url

            if not snippet:
                snippet = title  # fallback — at minimum repeat the title

            domain = _normalise_domain(url)
            if domain in seen_domains:
                logger.debug("Duplicate domain, skipping: %s", domain)
                continue

            seen_domains.add(domain)

            try:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_domain=domain,
                        retrieved_at=retrieved_at,
                    )
                )
            except Exception as exc:
                logger.warning("Could not parse search result for %r: %s", url, exc)

        logger.info("Returning %d deduplicated results.", len(results))
        return results


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_search(
    query: str,
    max_results: int,
    api_key: str,
    timeout: int = 30,
    max_retries: int = 3,
) -> list[SearchResult]:
    """
    Convenience wrapper — creates a SearchClient and runs one search.
    Returns an empty list (never raises) so the caller can handle gracefully.
    """
    try:
        client = SearchClient(api_key=api_key, timeout=timeout, max_retries=max_retries)
        return client.search(query=query, max_results=max_results)
    except Exception as exc:
        logger.error("run_search failed: %s", exc)
        return []
