#!/usr/bin/env python3
"""Free Web Search MCP Server v4.0.0.

Zero-cost web search and content extraction via MCP protocol.
Uses DuckDuckGo Lite + Mojeek + Qwant for search (parallel, first-wins),
DDG Instant Answer API for factual queries, Wikipedia REST API for encyclopedic
summaries, and Jina AI Reader / trafilatura for content extraction.
No API keys required.
"""

import asyncio
import logging
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Coroutine
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("free-web-search")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DDG_TIME_RANGE_MAP = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}

# Language code -> DDG kl parameter mapping
LANGUAGE_KL_MAP = {
    "en": "us-en",
    "zh": "cn-zh",
    "de": "de-de",
    "fr": "fr-fr",
    "es": "es-es",
    "pt": "pt-pt",
    "ja": "jp-jp",
    "ko": "kr-kr",
    "ru": "ru-ru",
    "it": "it-it",
    "nl": "nl-nl",
    "pl": "pl-pl",
    "ar": "ar-ar",
    "tr": "tr-tr",
    "id": "id-id",
    "th": "th-th",
    "vi": "vn-vn",
    "sv": "se-sv",
    "da": "dk-da",
    "fi": "fi-fi",
    "nb": "no-no",
    "cs": "cz-cs",
    "el": "gr-el",
    "he": "il-he",
    "hu": "hu-hu",
    "ro": "ro-ro",
    "uk": "uk-uk",
    "bg": "bg-bg",
}

JINA_READER_URL = "https://r.jina.ai"

READ_ONLY_HINT = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)

# Tracking params to strip from URLs
_TRACKING_PARAMS = frozenset([
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid",
    "ref", "ref_src", "ref_url",
    "si", "mc_cid", "mc_eid",
    "_ga", "_gl", "_hsenc", "_hsmi", "__hstc", "__hsfp",
    "vero_id", "_openstat", "yclid", "wickedid",
    "twclid", "ttclid", "li_fat_id", "gbraid", "wbraid",
])

# Domains to skip in all search results
_SKIP_DOMAINS = frozenset([
    "mojeek.com", "blocksurvey.io", "facebook.com", "twitter.com",
    "linkedin.com", "instagram.com", "youtube.com/results",
    "duckduckgo.com", "qwant.com",
])

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""  # "duckduckgo", "mojeek", or "qwant"


@dataclass
class ExtractedContent:
    content: str
    title: str = ""
    url: str = ""
    date: str = ""
    language: str = ""
    description: str = ""
    author: str = ""
    site_name: str = ""
    extraction_method: str = ""
    links: list[str] | None = None  # Optional: extracted link URLs


# ---------------------------------------------------------------------------
# Persistent HTTP client (module-level, reused across calls)
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


def _get_shared_client() -> httpx.AsyncClient:
    """Return the module-level persistent HTTP client, creating it if needed."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(15.0, connect=8.0),
            http2=True,
        )
    return _http_client


async def close_shared_client():
    """Close the persistent client (call on shutdown)."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# Retry helper (BUG FIX v4: uses callable factory, not a coroutine object)
# ---------------------------------------------------------------------------

async def _retry_async(
    fn: Callable[..., Coroutine],
    retries: int = 2,
    base_delay: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an async function with exponential-backoff retry.

    Accepts a callable that returns a coroutine (not the coroutine itself),
    so each retry creates a fresh coroutine.

    Retries on: TimeoutException, ConnectError, and HTTPStatusError for 5xx/429.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            return await fn(*args, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed (%s: %s)", attempt + 1, 1 + retries, type(exc).__name__, exc)
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code not in (429, 500, 502, 503, 504):
                raise
            logger.warning("Attempt %d/%d failed (HTTP %d)", attempt + 1, 1 + retries, exc.response.status_code)
        except asyncio.CancelledError:
            raise  # Don't retry cancelled tasks
        if attempt < retries:
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore


# ---------------------------------------------------------------------------
# URL / text normalization utilities
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip tracking params, normalize scheme, remove trailing slash."""
    parsed = urlparse(url)
    # Strip tracking params
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    clean_qs = {k: v for k, v in qs.items() if k.lower() not in _TRACKING_PARAMS}
    clean_query = urllib.parse.urlencode(clean_qs, doseq=True)
    # Normalize scheme
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    # Remove trailing slash (but keep root /)
    path = parsed.path.rstrip("/") or "/"
    netloc = parsed.netloc.lower()
    # Remove fragment
    return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, clean_query, ""))


def _extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www."""
    parsed = urlparse(url)
    return parsed.netloc.lower().lstrip("www.")


def _domain_matches(url: str, domain: str) -> bool:
    """Check if a URL's domain matches (including subdomains).

    Examples:
        _domain_matches("https://docs.github.com/foo", "github.com") -> True
        _domain_matches("https://github.com/foo", "github.com") -> True
        _domain_matches("https://notgithub.com/foo", "github.com") -> False
    """
    url_domain = urlparse(url).netloc.lower().lstrip("www.")
    target = domain.lower().lstrip("www.")
    return url_domain == target or url_domain.endswith("." + target)


def clean_title(title: str) -> str:
    """Collapse whitespace, remove breadcrumb patterns."""
    # Remove breadcrumb: "Home > Category > Title" -> "Title"
    title = re.sub(r'^[\w\s.&-]+(?:\s*>\s*[\w\s.&-]+)+\s*>\s*', '', title)
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def cap_snippet(snippet: str, max_len: int = 200) -> str:
    """Truncate snippet to max_len at word boundary with ellipsis."""
    if not snippet:
        return ""
    if len(snippet) <= max_len:
        return snippet
    truncated = snippet[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len * 0.6:
        truncated = truncated[:last_space]
    return truncated + "..."


def _dedup_domain(results: list[SearchResult], max_per_domain: int = 2) -> list[SearchResult]:
    """Keep at most max_per_domain results per domain."""
    domain_count: Counter = Counter()
    out: list[SearchResult] = []
    for r in results:
        domain = _extract_domain(r.url)
        if domain_count[domain] >= max_per_domain:
            continue
        domain_count[domain] += 1
        out.append(r)
    return out


def _smart_truncate(text: str, max_length: int) -> str:
    """Truncate at paragraph boundary, then sentence boundary, then word boundary."""
    if len(text) <= max_length:
        return text

    # Try paragraph boundary
    cut = text.rfind('\n\n', 0, max_length)
    if cut > max_length * 0.5:
        return text[:cut] + "\n\n[Content truncated]"

    # Try sentence boundary
    for sep in ('. ', '.\n', '! ', '? '):
        cut = text.rfind(sep, 0, max_length)
        if cut > max_length * 0.5:
            return text[:cut + 1] + "\n\n[Content truncated]"

    # Word boundary
    cut = text.rfind(' ', 0, max_length)
    if cut > max_length * 0.5:
        return text[:cut] + "...\n\n[Content truncated]"

    return text[:max_length] + "...\n\n[Content truncated]"


# ---------------------------------------------------------------------------
# Shared HTTP headers for search backends
# ---------------------------------------------------------------------------

def _search_headers() -> dict[str, str]:
    """Browser-like headers for search requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
    }


# ---------------------------------------------------------------------------
# DuckDuckGo Lite — search backend
# ---------------------------------------------------------------------------

def _parse_ddg_redirect(href: str) -> str:
    """Extract actual URL from a DDG redirect link."""
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        return urllib.parse.unquote(match.group(1))
    return href


async def search_ddg_lite(
    query: str,
    max_results: int = 10,
    time_range: str | None = None,
    language: str | None = None,
) -> list[SearchResult]:
    """Search via DuckDuckGo Lite HTML endpoint."""
    kl = LANGUAGE_KL_MAP.get(language, "us-en") if language else "us-en"
    params: dict[str, str] = {"q": query, "kl": kl}
    if time_range and time_range in DDG_TIME_RANGE_MAP:
        params["df"] = DDG_TIME_RANGE_MAP[time_range]

    async with httpx.AsyncClient(
        headers=_search_headers(),
        follow_redirects=True,
        timeout=httpx.Timeout(15.0, connect=8.0),
        http1=True,
    ) as ddg:
        resp = await _retry_async(ddg.get, retries=1, url="https://lite.duckduckgo.com/lite/", params=params)
        resp.raise_for_status()

        # DDG returns 202 on rate-limit; treat as empty
        if resp.status_code == 202 or len(resp.text) < 500:
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for link in soup.select("a.result-link"):
            href = link.get("href", "")
            url = _parse_ddg_redirect(href)
            if not url.startswith("http") or url in seen_urls:
                continue
            if any(d in url for d in _SKIP_DOMAINS):
                continue
            seen_urls.add(url)
            title = clean_title(link.get_text(strip=True))
            snippet_el = link.find_next("td", class_="result-snippet")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo"))
            if len(results) >= max_results:
                break

        return results


# ---------------------------------------------------------------------------
# Mojeek — search backend (free, unlimited, independent)
# ---------------------------------------------------------------------------

async def search_mojeek(
    query: str,
    max_results: int = 10,
    language: str | None = None,
) -> list[SearchResult]:
    """Search via Mojeek (independent search engine, free, unlimited)."""
    params: dict[str, str] = {"q": query}
    if language and language in LANGUAGE_KL_MAP:
        lang_code = language.split("-")[0] if "-" in language else language
        params["lang"] = lang_code

    async with httpx.AsyncClient(
        headers=_search_headers(),
        follow_redirects=True,
        timeout=httpx.Timeout(15.0, connect=8.0),
        http1=True,
    ) as client:
        resp = await _retry_async(client.get, retries=1, url="https://www.mojeek.com/search", params=params)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        # Mojeek wraps results in divs with class "result" or specific structure
        for result_div in soup.select("div.result, div[class*='result'], article"):
            a = result_div.select_one("a[href^='http']")
            if not a:
                continue
            href = a.get("href", "")
            if any(d in href for d in _SKIP_DOMAINS) or href in seen_urls:
                continue
            title = clean_title(a.get_text(strip=True))
            if len(title) < 15:
                continue
            seen_urls.add(href)
            # Snippet from abbr, p, or span inside the result div
            snippet = ""
            for el in result_div.select("p, span, abbr, .s, .desc"):
                t = el.get_text(strip=True)
                if len(t) > 30 and t != title:
                    snippet = t
                    break
            results.append(SearchResult(title=title, url=href, snippet=snippet, source="mojeek"))
            if len(results) >= max_results:
                break

        # Fallback: if no results found with specific selectors, try broader approach
        if not results:
            for item in soup.select("li, div"):
                a = item.select_one("a[href^='http']")
                if not a:
                    continue
                href = a.get("href", "")
                if any(d in href for d in _SKIP_DOMAINS) or href in seen_urls:
                    continue
                title = clean_title(a.get_text(strip=True))
                if len(title) < 20:
                    continue
                seen_urls.add(href)
                snippet = ""
                for p in item.select("p, span"):
                    t = p.get_text(strip=True)
                    if len(t) > 30 and t != title:
                        snippet = t
                        break
                results.append(SearchResult(title=title, url=href, snippet=snippet, source="mojeek"))
                if len(results) >= max_results:
                    break

        return results


# ---------------------------------------------------------------------------
# Qwant — search backend (free, EU-based, privacy-focused)
# ---------------------------------------------------------------------------

async def search_qwant(
    query: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """Search via Qwant (EU privacy search engine, free, no API key)."""
    params: dict[str, str] = {"q": query, "t": "web"}

    async with httpx.AsyncClient(
        headers=_search_headers(),
        follow_redirects=True,
        timeout=httpx.Timeout(15.0, connect=8.0),
        http1=True,
    ) as client:
        resp = await _retry_async(client.get, retries=1, url="https://www.qwant.com/", params=params)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        # Qwant results are in specific containers
        for item in soup.select("[class*='result'], [class*='web-result'], .tile"):
            a = item.select_one("a[href^='http']")
            if not a:
                continue
            href = a.get("href", "")
            # Skip Qwant internal links
            if "qwant.com" in href or href in seen_urls:
                continue
            if any(d in href for d in _SKIP_DOMAINS):
                continue
            title = clean_title(a.get_text(strip=True))
            if len(title) < 15:
                continue
            seen_urls.add(href)
            # Snippet
            snippet = ""
            for el in item.select("p, span, .desc, .url__desc"):
                t = el.get_text(strip=True)
                if len(t) > 30 and t != title:
                    snippet = t
                    break
            results.append(SearchResult(title=title, url=href, snippet=snippet, source="qwant"))
            if len(results) >= max_results:
                break

        return results


# ---------------------------------------------------------------------------
# Parallel search — race DDG + Mojeek + Qwant
# ---------------------------------------------------------------------------

async def _parallel_search(
    query: str,
    max_results: int = 10,
    time_range: str | None = None,
    language: str | None = None,
) -> list[SearchResult]:
    """Race DDG Lite, Mojeek, and Qwant in parallel; return first that produces results."""
    ddg_task = asyncio.create_task(search_ddg_lite(query, max_results, time_range, language))
    mojeek_task = asyncio.create_task(search_mojeek(query, max_results, language))
    qwant_task = asyncio.create_task(search_qwant(query, max_results))

    pending = {ddg_task, mojeek_task, qwant_task}

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            try:
                results = task.result()
                if results:
                    # Cancel remaining tasks
                    for p in pending:
                        p.cancel()
                    return results
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Search backend failed: %s", exc)

    return []


# ---------------------------------------------------------------------------
# DDG autocomplete — related searches
# ---------------------------------------------------------------------------

async def get_related_searches(
    query: str,
    max_suggestions: int = 10,
) -> list[str]:
    """Get related search queries from DDG autocomplete."""
    async with httpx.AsyncClient(
        headers=_search_headers(),
        timeout=httpx.Timeout(8.0, connect=5.0),
        http1=True,
    ) as ddg:
        resp = await _retry_async(
            ddg.get, retries=1,
            url="https://duckduckgo.com/ac/",
            params={"q": query, "type": "list"},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            suggestions = data[1]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            suggestions = [item.get("phrase", "") for item in data]
        else:
            suggestions = data if isinstance(data, list) else []
        return [s.strip() for s in suggestions if isinstance(s, str) and s.strip()][:max_suggestions]


# ---------------------------------------------------------------------------
# DDG Instant Answer API
# ---------------------------------------------------------------------------

async def get_instant_answer(query: str) -> dict[str, Any]:
    """Fetch structured answer from DDG Instant Answer API."""
    async with httpx.AsyncClient(
        headers=_search_headers(),
        timeout=httpx.Timeout(8.0, connect=5.0),
        http1=True,
    ) as client:
        resp = await _retry_async(
            client.get, retries=1,
            url="https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "0"},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Wikipedia REST API (multi-language)
# ---------------------------------------------------------------------------

async def get_wiki_summary(title: str, lang: str = "en") -> dict[str, Any]:
    """Fetch Wikipedia article summary via REST API. Supports all Wikipedia languages."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        http1=True,
    ) as client:
        resp = await _retry_async(client.get, retries=1, url=url)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Jina AI Reader — content extraction (primary, JSON mode)
# ---------------------------------------------------------------------------

async def fetch_with_jina(
    client: httpx.AsyncClient,
    url: str,
    return_format: str = "markdown",
    with_links: bool = False,
) -> ExtractedContent:
    """Fetch content via Jina AI Reader JSON mode for rich metadata.

    Args:
        return_format: 'markdown', 'text', or 'html'
        with_links: if True, include links summary in response
    """
    headers: dict[str, str] = {
        "Accept": "application/json",
        "X-Return-Format": return_format,
        "X-No-Cache": "true",
    }
    if with_links:
        headers["X-With-Links-Summary"] = "true"

    resp = await client.get(
        f"{JINA_READER_URL}/{url}",
        headers=headers,
        timeout=25.0,
    )
    resp.raise_for_status()
    data = resp.json()

    # Extract links if present
    links = None
    links_data = data.get("links", {})
    if isinstance(links_data, dict) and with_links:
        links = list(links_data.keys())[:50]

    return ExtractedContent(
        content=data.get("content", ""),
        title=data.get("title", ""),
        url=data.get("url", url),
        date=data.get("publishedTime", ""),
        language=data.get("metadata", {}).get("lang", "") if isinstance(data.get("metadata"), dict) else "",
        description=data.get("description", ""),
        site_name=data.get("metadata", {}).get("siteName", "") if isinstance(data.get("metadata"), dict) else "",
        extraction_method=f"jina_json ({return_format})",
        links=links,
    )


# ---------------------------------------------------------------------------
# Trafilatura — content extraction (local fallback, rich metadata)
# BUG FIX v4: no double-fetch, uses already-fetched HTML
# ---------------------------------------------------------------------------

async def fetch_with_trafilatura(client: httpx.AsyncClient, url: str) -> ExtractedContent:
    """Fetch and extract content using trafilatura bare_extraction for metadata."""
    resp = await client.get(url, timeout=15.0)
    resp.raise_for_status()

    # Use the already-fetched HTML instead of fetching again
    doc = trafilatura.bare_extraction(resp.text)

    if doc and doc.as_text():
        return ExtractedContent(
            content=doc.as_text(),
            title=doc.title or "",
            url=url,
            date=doc.date or "",
            language=doc.language or "",
            description=doc.description or "",
            author=doc.author or "",
            site_name=doc.sitename or "",
            extraction_method="trafilatura",
        )

    # Final fallback: BeautifulSoup text extraction with expanded tag stripping
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "form", "iframe", "noscript", "svg", "table", "button"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return ExtractedContent(
        content=text,
        title=soup.title.string if soup.title and soup.title.string else "",
        url=url,
        extraction_method="beautifulsoup",
    )


# ---------------------------------------------------------------------------
# Content fetcher with fallback chain
# ---------------------------------------------------------------------------

async def fetch_content(
    url: str,
    max_length: int = 15000,
    return_format: str = "markdown",
    with_links: bool = False,
) -> ExtractedContent:
    """Fetch readable content from a URL with Jina JSON -> trafilatura fallback."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL scheme: {url}. Must start with http:// or https://")

    client = _get_shared_client()

    try:
        ec = await fetch_with_jina(client, url, return_format, with_links)
        if ec.content and len(ec.content.strip()) > 100:
            ec.content = _smart_truncate(ec.content, max_length)
            return ec
    except Exception as e:
        logger.warning("Jina fetch failed for %s: %s", url, e)

    try:
        ec = await fetch_with_trafilatura(client, url)
        if ec.content and len(ec.content.strip()) > 50:
            ec.content = _smart_truncate(ec.content, max_length)
            return ec
    except Exception as e:
        logger.warning("Trafilatura fetch failed for %s: %s", url, e)

    raise RuntimeError(f"Failed to fetch content from {url}. All extraction methods failed.")


# ---------------------------------------------------------------------------
# Result post-processing
# ---------------------------------------------------------------------------

def post_process_results(results: list[SearchResult]) -> list[SearchResult]:
    """Apply all quality filters to search results."""
    processed: list[SearchResult] = []
    for r in results:
        r.url = normalize_url(r.url)
        r.title = clean_title(r.title)
        r.snippet = cap_snippet(r.snippet)
        # Filter empty snippets
        if not r.snippet.strip():
            continue
        processed.append(r)
    return _dedup_domain(processed)


def _apply_domain_filter(
    results: list[SearchResult],
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
) -> list[SearchResult]:
    """Filter results by domain allowlist/blocklist.

    BUG FIX v4: uses proper subdomain matching via _domain_matches().
    """
    if include_domains:
        results = [r for r in results
                   if any(_domain_matches(r.url, d) for d in include_domains)]
    if exclude_domains:
        results = [r for r in results
                   if not any(_domain_matches(r.url, d) for d in exclude_domains)]
    return results


# ---------------------------------------------------------------------------
# Format helpers (Markdown, LLM-optimized)
# BUG FIX v4: format_fetch_content metadata uses proper separator
# ---------------------------------------------------------------------------

def format_search_results(results: list[SearchResult], query: str, tool_name: str = "web_search") -> str:
    """Format search results into Markdown for LLM consumption."""
    if not results:
        return f"## {tool_name}: {query}\n\nNo results found. Try rephrasing or simplifying the query."

    source = results[0].source
    source_labels = {"duckduckgo": "DuckDuckGo", "mojeek": "Mojeek", "qwant": "Qwant"}
    source_label = source_labels.get(source, source)

    lines = [f"## {tool_name}: {query}\n"]
    lines.append(f"*Source: {source_label}*\n")

    for i, r in enumerate(results, 1):
        domain = _extract_domain(r.url)
        lines.append(f"**[{i}] [{r.title}]({r.url})**")
        lines.append(f"*{domain}*")
        if r.snippet:
            lines.append(f"> {r.snippet}")
        lines.append("")

    return "\n".join(lines)


def format_fetch_content(ec: ExtractedContent) -> str:
    """Format extracted content with metadata header."""
    parts = [f"## fetch_url: {ec.url}\n"]

    # Build metadata line with proper separators
    meta_parts: list[str] = []
    if ec.title:
        meta_parts.append(ec.title)
    if ec.author:
        meta_parts.append(f"by {ec.author}")
    if ec.site_name and ec.site_name != ec.title:
        meta_parts.append(ec.site_name)
    if ec.date:
        meta_parts.append(ec.date)
    if ec.language:
        meta_parts.append(ec.language.upper())
    if ec.extraction_method:
        meta_parts.append(f"via {ec.extraction_method}")

    if meta_parts:
        parts.append("*" + " | ".join(meta_parts) + "*\n")

    if ec.description:
        parts.append(f"> {ec.description}\n")

    parts.append(ec.content)

    # Append links summary if available
    if ec.links:
        parts.append("\n---\n**Links found on page:**")
        for link_url in ec.links[:20]:
            parts.append(f"- {link_url}")

    return "\n".join(parts)


def format_instant_answer(data: dict[str, Any], query: str) -> str:
    """Format DDG Instant Answer into structured Markdown."""
    parts = [f"## instant_answer: {query}\n"]
    has_content = False

    abstract = data.get("Abstract", "")
    if abstract:
        parts.append(f"**Summary:** {abstract}\n")
        has_content = True

    answer = data.get("Answer", "")
    if answer:
        parts.append(f"**Answer:** {answer}\n")
        has_content = True

    definition = data.get("Definition", "")
    if definition:
        parts.append(f"**Definition:** {definition}\n")
        has_content = True

    # Infobox
    infobox = data.get("Infobox", {})
    if infobox and isinstance(infobox, dict):
        content = infobox.get("content", [])
        if content and isinstance(content, list):
            parts.append("**Key Facts:**")
            for item in content[:15]:
                if isinstance(item, dict):
                    label = item.get("label", "")
                    value = item.get("value", "")
                    if label and value:
                        if isinstance(value, list):
                            value_str = ", ".join(
                                v.get("label", str(v)) if isinstance(v, dict) else str(v)
                                for v in value
                            )
                        else:
                            value_str = str(value)
                        parts.append(f"- **{label}:** {value_str}")
                        has_content = True
            parts.append("")

    # Related Topics
    related = data.get("RelatedTopics", [])
    if related and isinstance(related, list) and not has_content:
        # If we don't have a direct answer, show related topics as suggestions
        parts.append("**Related topics:**")
        for topic in related[:5]:
            if isinstance(topic, dict):
                text = topic.get("Text", "")
                if text:
                    parts.append(f"- {text}")
                    has_content = True
            elif isinstance(topic, str):
                parts.append(f"- {topic}")
                has_content = True
        parts.append("")

    # Source attribution
    source = data.get("AbstractSource", "")
    url = data.get("AbstractURL", "")
    if source:
        parts.append(f"*Source: {source}*")
        has_content = True
    if url:
        parts.append(f"*{url}*")

    if not has_content:
        parts.append(
            "No structured answer found for this query. "
            "Try:\n"
            "- Rephrasing as a factual question (e.g. \"What is X?\", \"Who invented Y?\")\n"
            "- Using `web_search` or `deep_search` for broader topics\n"
            "- Using `wiki_summary` for encyclopedic topics"
        )

    return "\n".join(parts)


def format_wiki_summary(data: dict[str, Any], title: str, lang: str = "en") -> str:
    """Format Wikipedia summary into structured Markdown."""
    parts = [f"## wiki_summary: {title}\n"]

    page_title = data.get("title", title)
    if page_title != title:
        parts.append(f"**Article:** {page_title}\n")

    description = data.get("description", "")
    if description:
        parts.append(f"*{description}*\n")

    extract = data.get("extract", "")
    if extract:
        parts.append(extract)

    thumbnail = data.get("thumbnail", {})
    if thumbnail and isinstance(thumbnail, dict) and thumbnail.get("source"):
        parts.append(f"\n[Thumbnail]({thumbnail['source']})")

    ts = data.get("timestamp", "")
    if ts:
        parts.append(f"\n*Last updated: {ts}*")

    page_url_obj = data.get("content_urls", {})
    if isinstance(page_url_obj, dict):
        desktop = page_url_obj.get("desktop", {})
        if isinstance(desktop, dict) and desktop.get("page"):
            parts.append(f"*{desktop['page']}*")

    if lang != "en":
        parts.insert(1, f"*Language: {lang}*\n")

    if not extract and not description:
        parts.append(
            "No Wikipedia article found. Try:\n"
            f"- Using `web_search` to find the topic\n"
            f"- Checking the article title spelling\n"
            f"- Using `wiki_summary(title=\"...\", lang=\"en\")` for English Wikipedia"
        )

    return "\n".join(parts)


def format_auto_answer(
    query: str,
    instant_data: dict[str, Any],
    wiki_data: dict[str, Any],
    search_results: list[SearchResult],
    wiki_title: str,
) -> str:
    """Format combined auto_answer output."""
    parts = [f"## auto_answer: {query}\n"]
    has_content = False

    # Best answer from DDG
    abstract = instant_data.get("Abstract", "")
    answer = instant_data.get("Answer", "")
    definition = instant_data.get("Definition", "")

    if answer:
        parts.append(f"**Answer:** {answer}\n")
        has_content = True
    elif abstract:
        parts.append(f"**Summary:** {abstract}\n")
        has_content = True
    elif definition:
        parts.append(f"**Definition:** {definition}\n")
        has_content = True

    # Wikipedia extract (usually high quality for factual queries)
    wiki_extract = wiki_data.get("extract", "")
    if wiki_extract and not has_content:
        parts.append(f"**From Wikipedia — {wiki_data.get('title', query)}:**\n")
        parts.append(wiki_extract)
        has_content = True
    elif wiki_extract and has_content:
        # Add Wikipedia as supplementary
        parts.append(f"**Additional context (Wikipedia):**\n")
        # Use a shorter version
        truncated = _smart_truncate(wiki_extract, 2000)
        parts.append(truncated)

    # Key facts from DDG infobox
    infobox = instant_data.get("Infobox", {})
    if infobox and isinstance(infobox, dict):
        content = infobox.get("content", [])
        if content and isinstance(content, list):
            parts.append("\n**Key Facts:**")
            for item in content[:10]:
                if isinstance(item, dict):
                    label = item.get("label", "")
                    value = item.get("value", "")
                    if label and value:
                        if isinstance(value, list):
                            value_str = ", ".join(
                                v.get("label", str(v)) if isinstance(v, dict) else str(v)
                                for v in value
                            )
                        else:
                            value_str = str(value)
                        parts.append(f"- **{label}:** {value_str}")
                        has_content = True
            parts.append("")

    # Source attribution
    source = instant_data.get("AbstractSource", "")
    if source:
        parts.append(f"*Answer source: {source}*")

    # Web results as references
    if search_results:
        parts.append("\n**References:**")
        for i, r in enumerate(search_results[:5], 1):
            parts.append(f"- [{r.title}]({r.url})")

    if not has_content:
        parts.append(
            "No direct answer found. The query may be too complex or too specific. Try:\n"
            "- Using `web_search` for general web results\n"
            "- Using `deep_search` for detailed research\n"
            "- Rephrasing as a simpler factual question"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = Server("free-web-search", version="4.0.0")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web for information. Returns ranked results with titles, URLs, and snippets. "
                "Uses DuckDuckGo + Mojeek + Qwant in parallel (first-wins for speed). "
                "Supports time-range filtering, language selection, and domain filtering.\n\n"
                "Supports `site:` operator (e.g. `query=\"react hooks site:github.com\"`)\n\n"
                "Examples:\n"
                '- web_search(query="Python async best practices")\n'
                '- web_search(query="React 19 release", time_range="month")\n'
                '- web_search(query="climate change 2025", max_results=15)\n'
                '- web_search(query="React hooks", include_domains=["github.com", "stackoverflow.com"])\n'
                '- web_search(query="AI news", exclude_domains=["facebook.com"], language="en")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-20, default 10)",
                        "default": 10,
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Filter by recency: 'day', 'week', 'month', 'year', or null for all time",
                        "enum": ["day", "week", "month", "year", None],
                        "default": None,
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language code (e.g. 'en', 'zh', 'de'). Default: auto-detect",
                        "default": None,
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only return results from these domains (e.g. ['github.com', 'docs.python.org']). Matches subdomains.",
                        "default": None,
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude results from these domains (e.g. ['facebook.com', 'twitter.com']). Matches subdomains.",
                        "default": None,
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="news_search",
            description=(
                "Search for recent news and current events. Defaults to last week. "
                "Uses DuckDuckGo + Mojeek + Qwant in parallel.\n\n"
                "Examples:\n"
                '- news_search(query="AI breakthroughs")\n'
                '- news_search(query="space exploration", time_range="week")\n'
                '- news_search(query="tech industry", max_results=15, language="zh")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "News search query string"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-20, default 10)",
                        "default": 10,
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Filter by recency: 'day', 'week', 'month', 'year', or null for all time",
                        "enum": ["day", "week", "month", "year", None],
                        "default": "week",
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language code (e.g. 'en', 'zh', 'de'). Default: auto-detect",
                        "default": None,
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only return results from these domains",
                        "default": None,
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude results from these domains",
                        "default": None,
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="fetch_url",
            description=(
                "Fetch and extract readable content from a URL. Returns Markdown with metadata header "
                "(Title | Author | Site | Date | Language | Method). Handles JavaScript-rendered pages.\n\n"
                "Examples:\n"
                '- fetch_url(url="https://docs.python.org/3/library/asyncio.html")\n'
                '- fetch_url(url="https://en.wikipedia.org/wiki/Quantum_computing")\n'
                '- fetch_url(url="https://example.com/article", max_length=8000)\n'
                '- fetch_url(url="https://example.com", return_format="text")\n'
                '- fetch_url(url="https://example.com/blog", with_links=true)'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch and extract content from"},
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum content length in characters (default 15000, max 50000)",
                        "default": 15000,
                    },
                    "return_format": {
                        "type": "string",
                        "description": "Output format: 'markdown' (default), 'text' (plain text), or 'html'",
                        "enum": ["markdown", "text", "html"],
                        "default": "markdown",
                    },
                    "with_links": {
                        "type": "boolean",
                        "description": "If true, include a list of links found on the page at the end of the output",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="deep_search",
            description=(
                "One-shot research: search the web, then fetch full content from the top results in parallel. "
                "Returns combined Markdown with numbered references. Supports time-range and language.\n\n"
                "Examples:\n"
                '- deep_search(query="LLM memory requirements")\n'
                '- deep_search(query="Rust async patterns", max_results=5)\n'
                '- deep_search(query="quantum computing 2025", time_range="year", language="en")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to fetch content from (1-5, default 3)",
                        "default": 3,
                    },
                    "max_content_length": {
                        "type": "integer",
                        "description": "Max content length per result in characters (default 4000, max 10000)",
                        "default": 4000,
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Filter by recency: 'day', 'week', 'month', 'year', or null for all time",
                        "enum": ["day", "week", "month", "year", None],
                        "default": None,
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language code (e.g. 'en', 'zh', 'de')",
                        "default": None,
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="instant_answer",
            description=(
                "Get a structured, factual answer for definitional or factual queries. "
                "Returns summary, answer, key facts (infobox), and definition. "
                "Best for questions like 'What is X?', 'Capital of Y?', 'Who invented Z?'\n\n"
                "Examples:\n"
                '- instant_answer(query="What is Python?")\n'
                '- instant_answer(query="Capital of France")\n'
                '- instant_answer(query="Speed of light")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Factual or definitional question",
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="wiki_summary",
            description=(
                "Get a Wikipedia article summary. Returns title, description, extract text, "
                "thumbnail, and timestamp. Supports all Wikipedia languages.\n\n"
                "Examples:\n"
                '- wiki_summary(title="Artificial intelligence")\n'
                '- wiki_summary(title="Machine learning")\n'
                '- wiki_summary(title="Quantum computing", lang="de")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Wikipedia article title (case-insensitive, spaces allowed)",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Wikipedia language code (default 'en'). Examples: 'de', 'fr', 'es', 'zh', 'ja', 'ru'",
                        "default": "en",
                    },
                },
                "required": ["title"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="auto_answer",
            description=(
                "Comprehensive answer combining instant answer, Wikipedia, and web search. "
                "Best for complex factual questions that benefit from multiple sources. "
                "Returns a synthesized answer with key facts and references.\n\n"
                "Examples:\n"
                '- auto_answer(query="What is quantum entanglement?")\n'
                '- auto_answer(query="History of the Roman Empire")\n'
                '- auto_answer(query="How does CRISPR work?")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question to answer (can be factual, conceptual, or encyclopedic)",
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="related_searches",
            description=(
                "Get related or expanded search queries for a given topic. Useful for broadening "
                "or refining a search. Returns a list of suggested queries.\n\n"
                "Examples:\n"
                '- related_searches(query="machine learning")\n'
                '- related_searches(query="climate change solutions")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Base query to get related searches for"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of related queries (1-15, default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
    ]


def error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]):
    try:
        if name == "web_search":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")

            max_results = max(1, min(20, arguments.get("max_results", 10)))
            time_range = arguments.get("time_range")
            language = arguments.get("language")
            include_domains = arguments.get("include_domains")
            exclude_domains = arguments.get("exclude_domains")

            try:
                results = await _parallel_search(query, max_results, time_range, language)
                results = post_process_results(results)
                results = _apply_domain_filter(results, include_domains, exclude_domains)
                if not results:
                    return error_result(
                        f"No results found for: {query}. "
                        "Try rephrasing, simplifying the query, or removing domain filters."
                    )
                return [TextContent(type="text", text=format_search_results(results, query, "web_search"))]
            except Exception as e:
                return error_result(f"Search error: {e}")

        elif name == "news_search":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")

            max_results = max(1, min(20, arguments.get("max_results", 10)))
            time_range = arguments.get("time_range", "week")
            language = arguments.get("language")
            include_domains = arguments.get("include_domains")
            exclude_domains = arguments.get("exclude_domains")

            try:
                results = await _parallel_search(query, max_results, time_range, language)
                results = post_process_results(results)
                results = _apply_domain_filter(results, include_domains, exclude_domains)
                if not results:
                    return error_result(
                        f"No results found for: {query}. "
                        "Try rephrasing, simplifying the query, or removing domain filters."
                    )
                return [TextContent(type="text", text=format_search_results(results, query, "news_search"))]
            except Exception as e:
                return error_result(f"News search error: {e}")

        elif name == "fetch_url":
            url = arguments.get("url", "")
            if not url:
                return error_result("Error: 'url' is required")

            max_length = min(arguments.get("max_length", 15000), 50000)
            return_format = arguments.get("return_format", "markdown") or "markdown"
            with_links = arguments.get("with_links", False)

            try:
                ec = await fetch_content(url, max_length, return_format, with_links)
                return [TextContent(type="text", text=format_fetch_content(ec))]
            except Exception as e:
                return error_result(f"Fetch error: {e}")

        elif name == "deep_search":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")

            num_results = max(1, min(5, arguments.get("max_results", 3)))
            max_content_length = min(arguments.get("max_content_length", 4000), 10000)
            time_range = arguments.get("time_range")
            language = arguments.get("language")

            try:
                # Step 1: Search
                results = await _parallel_search(query, max(5, num_results * 2), time_range, language)
                results = post_process_results(results)
                if not results:
                    return error_result(f"No results found for: {query}")

                top_results = results[:num_results]

                # Step 2: Fetch content from top results in parallel
                async def _fetch_one(r: SearchResult) -> tuple[SearchResult, ExtractedContent]:
                    ec = await fetch_content(r.url, max_content_length)
                    return (r, ec)

                tasks = [_fetch_one(r) for r in top_results]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)

                # Step 3: Combine results
                parts = [f"## deep_search: {query}\n"]
                for idx, item in enumerate(gathered, 1):
                    if isinstance(item, Exception):
                        logger.warning("deep_search fetch failed for result %d: %s", idx, item)
                        r = top_results[idx - 1]
                        parts.append(f"### [{idx}] {r.title}")
                        parts.append(f"*{_extract_domain(r.url)}*\n")
                        if r.snippet:
                            parts.append(f"> {r.snippet}\n")
                        continue

                    r, ec = item
                    parts.append(f"### [{idx}] [{r.title}]({r.url})")
                    meta = []
                    if ec.author:
                        meta.append(ec.author)
                    if ec.date:
                        meta.append(ec.date)
                    if meta:
                        parts.append(f"*{' | '.join(meta)}*\n")
                    parts.append(ec.content)
                    parts.append("")

                return [TextContent(type="text", text="\n".join(parts))]
            except Exception as e:
                return error_result(f"Deep search error: {e}")

        elif name == "instant_answer":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")

            try:
                data = await get_instant_answer(query)
                return [TextContent(type="text", text=format_instant_answer(data, query))]
            except Exception as e:
                return error_result(f"Instant answer error: {e}")

        elif name == "wiki_summary":
            title = arguments.get("title", "")
            if not title:
                return error_result("Error: 'title' is required")

            lang = arguments.get("lang", "en") or "en"

            try:
                data = await get_wiki_summary(title, lang)
                return [TextContent(type="text", text=format_wiki_summary(data, title, lang))]
            except Exception as e:
                return error_result(f"Wikipedia summary error: {e}")

        elif name == "auto_answer":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")

            try:
                # Fire all three sources in parallel
                instant_task = asyncio.create_task(get_instant_answer(query))
                wiki_task = asyncio.create_task(get_wiki_summary(query))
                search_task = asyncio.create_task(
                    _parallel_search(query, 5, language="en")
                )

                instant_data = await instant_task
                wiki_data = await wiki_task
                search_results = await search_task

                # Post-process search results
                search_results = post_process_results(search_results)

                return [TextContent(type="text", text=format_auto_answer(
                    query, instant_data, wiki_data, search_results, query
                ))]
            except Exception as e:
                return error_result(f"Auto answer error: {e}")

        elif name == "related_searches":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")

            max_results = min(arguments.get("max_results", 10), 15)

            try:
                suggestions = await get_related_searches(query, max_results)
                if not suggestions:
                    return [TextContent(type="text", text=f"## related_searches: {query}\n\nNo related searches found.")]
                lines = [f"## related_searches: {query}\n"]
                for i, s in enumerate(suggestions, 1):
                    lines.append(f"{i}. {s}")
                return [TextContent(type="text", text="\n".join(lines))]
            except Exception as e:
                return error_result(f"Related searches error: {e}")

        else:
            return error_result(f"Unknown tool: {name}")

    except Exception as e:
        logger.error("Unhandled error in %s: %s", name, e)
        return error_result(f"Internal error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    await close_shared_client()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
