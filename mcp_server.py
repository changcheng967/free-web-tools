#!/usr/bin/env python3
"""Free Web Search MCP Server v5.0.0.

Zero-cost web search and content extraction via MCP protocol.
Uses DuckDuckGo Lite + Mojeek + Bing + Startpage for search (parallel, first-wins),
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

BING_FRESHNESS_MAP = {
    "day": "ez5_80505_",
    "week": "ez5_80506_",
    "month": "ez5_80507_",
    "year": "ez5_80508_",
}

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
    "duckduckgo.com",
])

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""  # "duckduckgo", "mojeek", "bing", or "startpage"


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


def _is_valid_url(url: str) -> bool:
    """Reject URLs with invalid characters (Unicode garbage, missing scheme, etc.)."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False
        # Reject non-ASCII in domain/path — catches '›' (U+203A) and similar garbage
        for c in parsed.netloc + parsed.path:
            if ord(c) > 127:
                return False
        return True
    except Exception:
        return False


def _sanitize_url(url: str) -> str:
    """Clean common URL corruption from search engine HTML parsing.

    Handles: '›' (U+203A) used as visual separator, stray whitespace, etc.
    """
    # Replace Unicode angle brackets used as visual path separators
    url = url.replace("\u203a", "/").replace("\u00bb", "/").replace("›", "/")
    # Collapse multiple slashes
    url = re.sub(r'/+', '/', url)
    return url.strip()


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
# Query parsing utilities
# ---------------------------------------------------------------------------

def _parse_site_operator(query: str) -> tuple[str, list[str]]:
    """Extract site:domain.com operators from query, return (cleaned_query, [domains])."""
    domains: list[str] = []
    # Match site: followed by a domain (letters, digits, dots, hyphens)
    clean = re.sub(r'\bsite:([a-zA-Z0-9._-]+\.[a-zA-Z]{2,})\b', lambda m: (domains.append(m.group(1)) or ''), query)
    clean = clean.strip()
    # If query is now empty after removing site: operators, use a generic search
    if not clean and domains:
        clean = domains[0]
    return clean, domains


def _extract_wiki_query(query: str) -> str:
    """Strip question words from query to extract a Wikipedia article title.

    "What is quantum entanglement?" -> "quantum entanglement"
    "Who invented the telephone?" -> "telephone"
    """
    q = query.strip().rstrip("?!.!")
    # Remove common question prefixes (case-insensitive)
    q = re.sub(
        r'^(what\s+(?:is|are|was|were)\s+|who\s+(?:is|are|was|were|invented|discovered|created)\s+|'
        r'when\s+(?:was|were|is|did)\s+|where\s+(?:is|are|was|were)\s+|'
        r'how\s+(?:does|do|did|is|are|was|were)\s+|why\s+(?:do|does|did|is|are|was|were)\s+)',
        '', q, flags=re.IGNORECASE,
    )
    # Remove common filler words
    q = re.sub(r'\b(?:the|a|an|of|in|for|to|about)\b', '', q, flags=re.IGNORECASE)
    # Collapse whitespace and strip
    q = re.sub(r'\s+', ' ', q).strip()
    return q


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
    """Search via Mojeek (independent search engine, free, unlimited).

    Mojeek HTML structure: all results are in a single container. Each result has
    two <a> tags with the same href — first is the URL display (with › separators),
    second is the actual title. Snippets are in following <p> tags.
    """
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

        # Mojeek result structure: pairs of <a> tags with same href.
        # First <a> has URL display text (contains ›), second has title.
        # Snippet follows in a <p> tag.
        all_links = soup.find_all("a", href=True)

        for a in all_links:
            href = a.get("href", "")
            text = a.get_text(strip=True)

            # Skip non-external links
            if not href.startswith("http"):
                continue
            # Skip internal Mojeek links
            if "mojeek.com" in href:
                continue
            # Skip URL display links — they contain › or start with http
            # We want the TITLE links which come second with the same href
            is_url_display = text.startswith("http") or "›" in text or "\u203a" in text
            if is_url_display:
                continue

            # This is a title link
            if not _is_valid_url(href) or href in seen_urls:
                continue
            if any(d in href for d in _SKIP_DOMAINS):
                continue

            title = clean_title(text)
            if len(title) < 10:
                continue

            seen_urls.add(href)

            # Find snippet: look at next sibling elements for a <p> with content.
            # Mojeek has two <p> tags per result: first is URL display (skip),
            # second is the actual snippet.
            snippet = ""
            sibling = a.next_element
            p_count = 0
            while sibling and not isinstance(sibling, str):
                if hasattr(sibling, 'name') and sibling.name == 'p':
                    p_text = sibling.get_text(strip=True)
                    p_count += 1
                    # Skip first <p> (URL display) and "See more results" links
                    is_url_display = (p_text.startswith("http") or "›" in p_text
                                      or p_text.startswith("See more"))
                    if not is_url_display and len(p_text) > 20:
                        snippet = p_text
                        break
                    # Only check the first 3 <p> tags
                    if p_count >= 3:
                        break
                sibling = sibling.next_element

            results.append(SearchResult(title=title, url=href, snippet=snippet, source="mojeek"))
            if len(results) >= max_results:
                break

        return results


# ---------------------------------------------------------------------------
# Startpage — search backend (Google proxy, works in China, English results)
# ---------------------------------------------------------------------------

_STARTPAGE_SKIP = _SKIP_DOMAINS | frozenset(["startpage.com"])


async def search_startpage(
    query: str,
    max_results: int = 10,
    time_range: str | None = None,
) -> list[SearchResult]:
    """Search via Startpage (Google proxy, returns English results, works in China)."""
    params: dict[str, str] = {"query": query}
    if time_range in ("day", "week", "month", "year"):
        params["with_date"] = time_range

    async with httpx.AsyncClient(
        headers=_search_headers(),
        follow_redirects=True,
        timeout=httpx.Timeout(15.0, connect=8.0),
        http1=True,
    ) as client:
        resp = await _retry_async(client.get, retries=1, url="https://www.startpage.com/search", params=params)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for div in soup.select("div.result"):
            # Title and URL from a.result-title (direct URL, no redirect)
            a = div.select_one("a.result-title")
            if not a:
                continue

            href = a.get("href", "")
            if not _is_valid_url(href) or href in seen_urls:
                continue
            if any(d in href for d in _STARTPAGE_SKIP):
                continue

            title = clean_title(a.get_text(strip=True))
            if len(title) < 5:
                continue

            seen_urls.add(href)

            # Snippet from p.description
            snippet = ""
            p = div.select_one("p.description")
            if p:
                snippet = p.get_text(strip=True)

            results.append(SearchResult(title=title, url=href, snippet=snippet, source="startpage"))
            if len(results) >= max_results:
                break

        return results


# ---------------------------------------------------------------------------
# Bing — search backend (reliable, works globally including China)
# ---------------------------------------------------------------------------

_BING_SKIP_DOMAINS = _SKIP_DOMAINS | frozenset(["bing.com", "microsoft.com", "live.com"])


async def search_bing(
    query: str,
    max_results: int = 10,
    language: str | None = None,
    time_range: str | None = None,
) -> list[SearchResult]:
    """Search via Bing (uses <cite> for actual URLs, bypasses redirect tracking)."""
    params: dict[str, str] = {"q": query}
    # Force English market to avoid geo-targeted results (e.g., Chinese results in China)
    if language and language != "en":
        params["setlang"] = language
    else:
        params["mkt"] = "en-US"
    if time_range and time_range in BING_FRESHNESS_MAP:
        params["filters"] = f'ex1:"{BING_FRESHNESS_MAP[time_range]}"'

    async with httpx.AsyncClient(
        headers=_search_headers(),
        follow_redirects=True,
        timeout=httpx.Timeout(15.0, connect=8.0),
        http1=True,
    ) as client:
        resp = await _retry_async(client.get, retries=1, url="https://www.bing.com/search", params=params)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a:
                continue

            # Extract actual URL from <cite> element (Bing hrefs are redirect URLs)
            cite = li.select_one("cite")
            url = ""
            if cite:
                cite_text = cite.get_text(strip=True).replace("›", "/").replace("\u203a", "/")
                # cite might be like "https://www.python.org" or "www.python.org › ... › ..."
                # Take only the first part before any whitespace/special chars
                cite_text = re.split(r'[\s\u203a›]+', cite_text)[0]
                if cite_text.startswith("//"):
                    cite_text = "https:" + cite_text
                elif cite_text.startswith("www."):
                    cite_text = "https://" + cite_text
                url = cite_text

            if not _is_valid_url(url) or url in seen_urls:
                continue
            if any(d in url for d in _BING_SKIP_DOMAINS):
                continue
            seen_urls.add(url)

            title = clean_title(a.get_text(strip=True))
            if len(title) < 5:
                continue

            # Snippet from paragraph
            snippet = ""
            p = li.select_one("p, .b_caption p")
            if p:
                snippet = p.get_text(strip=True)

            results.append(SearchResult(title=title, url=url, snippet=snippet, source="bing"))
            if len(results) >= max_results:
                break

        return results


# ---------------------------------------------------------------------------
# Parallel search — race DDG + Mojeek + Bing + Startpage
# ---------------------------------------------------------------------------

async def _parallel_search(
    query: str,
    max_results: int = 10,
    time_range: str | None = None,
    language: str | None = None,
) -> list[SearchResult]:
    """Race DDG Lite, Mojeek, Bing, and Startpage in parallel; return first that produces results."""
    ddg_task = asyncio.create_task(search_ddg_lite(query, max_results, time_range, language))
    mojeek_task = asyncio.create_task(search_mojeek(query, max_results, language))
    bing_task = asyncio.create_task(search_bing(query, max_results, language, time_range))
    startpage_task = asyncio.create_task(search_startpage(query, max_results, time_range))

    pending = {ddg_task, mojeek_task, bing_task, startpage_task}

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
    """Get related search queries from DDG autocomplete, with Bing Suggest fallback."""

    async def _ddg_suggest() -> list[str]:
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

    async def _bing_suggest() -> list[str]:
        async with httpx.AsyncClient(
            headers=_search_headers(),
            timeout=httpx.Timeout(8.0, connect=5.0),
            http1=True,
        ) as client:
            resp = await _retry_async(
                client.get, retries=1,
                url="https://www.bing.com/osjson.aspx",
                params={"query": query},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
                return [s.strip() for s in data[1] if isinstance(s, str) and s.strip()][:max_suggestions]
            return []

    try:
        results = await _ddg_suggest()
        if results:
            return results
    except Exception as exc:
        logger.warning("DDG autocomplete failed, trying Bing fallback: %s", exc)

    try:
        results = await _bing_suggest()
        if results:
            return results
    except Exception as exc:
        logger.warning("Bing suggest also failed: %s", exc)

    return []


# ---------------------------------------------------------------------------
# DDG Instant Answer API
# ---------------------------------------------------------------------------

async def get_instant_answer(query: str) -> dict[str, Any]:
    """Fetch structured answer from DDG Instant Answer API, with Wikipedia fallback."""
    try:
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
    except Exception as exc:
        logger.warning("DDG Instant Answer failed, trying Wikipedia fallback: %s", exc)

    # Wikipedia fallback: extract topic from query and look up Wikipedia
    wiki_query = _extract_wiki_query(query)
    if wiki_query:
        try:
            wiki_data = await get_wiki_summary(wiki_query)
            if wiki_data and wiki_data.get("extract"):
                # Format Wikipedia data to match DDG Instant Answer structure
                return {
                    "Abstract": wiki_data.get("extract", ""),
                    "AbstractSource": "Wikipedia",
                    "AbstractURL": wiki_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "Heading": wiki_data.get("title", query),
                }
        except Exception as exc:
            logger.warning("Wikipedia fallback also failed: %s", exc)

    # Return empty DDG-style response so format_instant_answer handles gracefully
    return {}


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

    if doc and doc.text:
        return ExtractedContent(
            content=doc.text,
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
# Arxiv-specific content extraction
# ---------------------------------------------------------------------------

async def _fetch_arxiv(client: httpx.AsyncClient, url: str) -> ExtractedContent | None:
    """Extract content from arxiv papers via direct HTML parsing of /abs/ page."""
    # Convert /html/ to /abs/ for reliable parsing
    abs_url = re.sub(r'arxiv\.org/html/', 'arxiv.org/abs/', url)

    try:
        resp = await client.get(abs_url, timeout=15.0)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Extract metadata
        title_el = soup.select_one("h1.title, .title")
        abstract_el = soup.select_one("blockquote.abstract, .abstract")
        author_els = soup.select("div.authors a")

        title = ""
        if title_el:
            title = title_el.get_text(strip=True).replace("Title:", "").strip()

        abstract = ""
        if abstract_el:
            abstract = abstract_el.get_text(strip=True).replace("Abstract:", "").strip()

        authors = ", ".join(a.get_text(strip=True) for a in author_els[:10])

        if not title and not abstract:
            return None

        content_parts = []
        if title:
            content_parts.append(f"# {title}\n")
        if authors:
            content_parts.append(f"**Authors:** {authors}\n")
        content_parts.append(f"**URL:** {abs_url}\n")
        content_parts.append(f"**PDF:** {abs_url.replace('/abs/', '/pdf/')}\n")
        if abstract:
            content_parts.append(f"\n## Abstract\n\n{abstract}")

        return ExtractedContent(
            content="\n".join(content_parts),
            title=title,
            url=url,
            author=authors,
            extraction_method="arxiv_direct",
        )
    except Exception as e:
        logger.warning("Arxiv direct fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# GitHub REST API helper
# ---------------------------------------------------------------------------

_GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "free-web-tools-mcp/5.0.0",
}


async def _github_api_get(path: str) -> dict | list | None:
    """GET from GitHub REST API. Returns JSON on success, None on failure."""
    client = _get_shared_client()
    try:
        url = f"https://api.github.com{path}"
        resp = await client.get(url, headers=_GITHUB_API_HEADERS, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("GitHub API %s returned %d", path, resp.status_code)
    except Exception as exc:
        logger.warning("GitHub API request failed for %s: %s", path, exc)
    return None


# ---------------------------------------------------------------------------
# GitHub — repo info
# ---------------------------------------------------------------------------

async def github_repo_info(
    owner: str,
    repo: str,
    include_readme: bool = True,
) -> str:
    """Fetch repo metadata + optional README from GitHub."""
    client = _get_shared_client()

    # Fetch repo metadata via API
    repo_data = await _github_api_get(f"/repos/{owner}/{repo}")
    if not repo_data:
        raise RuntimeError(f"Repository not found: {owner}/{repo}")

    default_branch = repo_data.get("default_branch", "main")

    # Fetch README
    readme = ""
    if include_readme:
        for name in ["README.md", "README.rst", "README.txt", "README", "readme.md"]:
            for br in [default_branch, "main", "master"]:
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/{br}/{name}"
                try:
                    resp = await client.get(url, timeout=10.0, follow_redirects=True)
                    if resp.status_code == 200 and len(resp.text) > 10:
                        readme = resp.text
                        break
                except Exception:
                    continue
            if readme:
                break

    return _format_github_repo(repo_data, readme)


# ---------------------------------------------------------------------------
# GitHub — file content
# ---------------------------------------------------------------------------

_BINARY_EXTENSIONS = frozenset([
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".so", ".dll", ".dylib", ".exe", ".bin", ".dat",
    ".pyc", ".pyd", ".o", ".a", ".lib",
])


async def github_file_content(
    owner: str,
    repo: str,
    path: str,
    branch: str | None = None,
    max_length: int = 15000,
) -> str:
    """Fetch a specific file from a GitHub repository via raw.githubusercontent.com."""
    # Check for binary files
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in _BINARY_EXTENSIONS:
        raise RuntimeError(f"Binary file not supported: {path}")

    client = _get_shared_client()

    # Determine branch
    if not branch:
        repo_data = await _github_api_get(f"/repos/{owner}/{repo}")
        branch = repo_data.get("default_branch", "main") if repo_data else "main"

    # Fetch from raw.githubusercontent.com (no rate limit)
    encoded_path = urllib.parse.quote(path, safe="/")
    for br in [branch, "main", "master"]:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{br}/{encoded_path}"
        try:
            resp = await client.get(url, timeout=15.0, follow_redirects=True)
            if resp.status_code == 200:
                content = resp.text
                actual_branch = br
                break
        except Exception:
            continue
    else:
        raise RuntimeError(f"File not found: {owner}/{repo}/{path} (tried branches: {branch}, main, master)")

    # Truncate if needed
    if len(content) > max_length:
        content = _smart_truncate(content, max_length)

    return _format_github_file(owner, repo, path, content, actual_branch)


# ---------------------------------------------------------------------------
# GitHub — search repos
# ---------------------------------------------------------------------------

async def github_search_repos(
    query: str,
    language: str | None = None,
    sort: str = "stars",
    max_results: int = 10,
    min_stars: int | None = None,
) -> str:
    """Search GitHub for repositories."""
    # Build query
    parts = [query]
    if language:
        parts.append(f"language:{language}")
    if min_stars:
        parts.append(f"stars:>={min_stars}")
    q = " ".join(parts)

    encoded_q = urllib.parse.quote(q, safe="")
    path = f"/search/repositories?q={encoded_q}&sort={sort}&order=desc&per_page={max_results}"
    data = await _github_api_get(path)

    if not data or not data.get("items"):
        return f"## github_search_repos: {query}\n\nNo repositories found."

    return _format_github_search_repos(data["items"][:max_results], query, data.get("total_count", 0), language)


# ---------------------------------------------------------------------------
# GitHub — issues / PRs
# ---------------------------------------------------------------------------

async def github_issues(
    owner: str,
    repo: str,
    issue_number: int | None = None,
    state: str = "open",
    issue_type: str = "issue",
    search: str = "",
    sort: str = "updated",
    max_results: int = 10,
) -> str:
    """List or fetch GitHub issues / PRs."""
    if issue_number is not None:
        return await _github_single_issue(owner, repo, issue_number)

    # List / search issues
    if search:
        q = f"repo:{owner}/{repo} is:{issue_type} is:{state} {search}"
        encoded_q = urllib.parse.quote(q, safe="")
        path = f"/search/issues?q={encoded_q}&sort={sort}&order=desc&per_page={max_results}"
        data = await _github_api_get(path)
    else:
        path = f"/repos/{owner}/{repo}/issues?state={state}&sort={sort}&direction=desc&per_page={max_results}"
        data = await _github_api_get(path)

    if not data:
        return f"## github_issues: {owner}/{repo}\n\nNo issues found."

    items = data.get("items", data) if isinstance(data, dict) else data
    if not items:
        return f"## github_issues: {owner}/{repo}\n\nNo issues found."

    # Filter by type if needed (GitHub API returns both issues and PRs)
    if issue_type == "pr":
        items = [i for i in items if i.get("pull_request")]
    elif issue_type == "issue":
        items = [i for i in items if not i.get("pull_request")]

    return _format_github_issues_list(items[:max_results], owner, repo, state, issue_type)


async def _github_single_issue(owner: str, repo: str, number: int) -> str:
    """Fetch a single issue or PR with details and comments."""
    issue = await _github_api_get(f"/repos/{owner}/{repo}/issues/{number}")
    if not issue:
        raise RuntimeError(f"Issue/PR not found: {owner}/{repo}#{number}")

    is_pr = "pull_request" in issue
    pr_data = None
    if is_pr:
        pr_data = await _github_api_get(f"/repos/{owner}/{repo}/pulls/{number}")

    # Fetch comments (up to 10)
    comments_text = ""
    comment_count = issue.get("comments", 0)
    if comment_count > 0:
        comments = await _github_api_get(f"/repos/{owner}/{repo}/issues/{number}/comments")
        if isinstance(comments, list):
            parts = []
            for c in comments[:10]:
                user = c.get("user", {}).get("login", "unknown")
                body = (c.get("body") or "")[:500]
                if body:
                    parts.append(f"**@{user}:**\n{body}")
            comments_text = "\n\n---\n\n".join(parts)

    return _format_github_issue_detail(issue, pr_data, comments_text)


# ---------------------------------------------------------------------------
# Code search — grep.app
# ---------------------------------------------------------------------------

async def code_search(
    query: str,
    language: str | None = None,
    repo: str | None = None,
    max_results: int = 10,
) -> str:
    """Search code across GitHub via grep.app API (free, no key)."""
    client = _get_shared_client()
    params: dict[str, str] = {"q": query}
    if language:
        params["filter[lang][0]"] = language
    if repo:
        params["filter[repo][0]"] = repo

    try:
        resp = await client.get(
            "https://grep.app/api/search",
            params=params,
            timeout=12.0,
            headers={"User-Agent": "free-web-tools-mcp/5.0.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Code search failed: {exc}")

    hits = data.get("hits", {})
    items = hits.get("hits", [])
    total = hits.get("total", 0)

    if not items:
        return f"## code_search: {query}\n\nNo code results found."

    return _format_code_search(items[:max_results], query, total, language)


# ---------------------------------------------------------------------------
# Package info — PyPI / npm / crates.io
# ---------------------------------------------------------------------------

async def package_info(
    name: str,
    registry: str = "pypi",
    version: str | None = None,
) -> str:
    """Fetch package metadata from PyPI, npm, or crates.io."""
    client = _get_shared_client()

    if registry == "pypi":
        return await _fetch_pypi(client, name, version)
    elif registry == "npm":
        return await _fetch_npm(client, name, version)
    elif registry == "crates":
        return await _fetch_crates(client, name)
    else:
        raise RuntimeError(f"Unknown registry: {registry}. Use 'pypi', 'npm', or 'crates'.")


async def _fetch_pypi(client: httpx.AsyncClient, name: str, version: str | None = None) -> str:
    """Fetch package info from PyPI JSON API."""
    if version:
        url = f"https://pypi.org/pypi/{name}/{version}/json"
    else:
        url = f"https://pypi.org/pypi/{name}/json"
    try:
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        if resp.status_code == 404:
            raise RuntimeError(f"Package '{name}' not found on PyPI")
        resp.raise_for_status()
        data = resp.json()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"PyPI lookup failed for '{name}': {exc}")

    info = data.get("info", {})
    return _format_pypi(info, name)


async def _fetch_npm(client: httpx.AsyncClient, name: str, version: str | None = None) -> str:
    """Fetch package info from npm registry."""
    try:
        resp = await client.get(
            f"https://registry.npmjs.org/{name}",
            timeout=10.0,
            follow_redirects=True,
        )
        if resp.status_code == 404:
            raise RuntimeError(f"Package '{name}' not found on npm")
        resp.raise_for_status()
        data = resp.json()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"npm lookup failed for '{name}': {exc}")

    return _format_npm(data, name, version)


async def _fetch_crates(client: httpx.AsyncClient, name: str) -> str:
    """Fetch crate info from crates.io API."""
    try:
        resp = await client.get(
            f"https://crates.io/api/v1/crates/{name}",
            timeout=10.0,
            headers={"User-Agent": "free-web-tools-mcp/5.0.0"},
        )
        if resp.status_code == 404:
            raise RuntimeError(f"Crate '{name}' not found on crates.io")
        resp.raise_for_status()
        data = resp.json()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"crates.io lookup failed for '{name}': {exc}")

    crate = data.get("crate", {})
    return _format_crates(crate, name)


# ---------------------------------------------------------------------------
# Format helpers for new tools
# ---------------------------------------------------------------------------

def _human_count(n: int) -> str:
    """Format large numbers: 69200 -> '69.2k'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _format_github_repo(repo_data: dict, readme: str) -> str:
    """Format GitHub repo info as Markdown."""
    full_name = repo_data.get("full_name", "")
    parts = [f"## github_repo_info: {full_name}\n"]

    desc = repo_data.get("description", "")
    if desc:
        parts.append(f"**{desc}**\n")

    # Metadata line
    meta = []
    if repo_data.get("language"):
        meta.append(repo_data["language"])
    if repo_data.get("stargazers_count") is not None:
        meta.append(f"★ {_human_count(repo_data['stargazers_count'])}")
    if repo_data.get("forks_count") is not None:
        meta.append(f"⑂ {_human_count(repo_data['forks_count'])}")
    if repo_data.get("open_issues_count") is not None:
        meta.append(f"Issues: {repo_data['open_issues_count']:,}")
    if repo_data.get("license", {}).get("spdx_id") and repo_data["license"]["spdx_id"] != "NOASSERTION":
        meta.append(f"License: {repo_data['license']['spdx_id']}")
    if repo_data.get("default_branch"):
        meta.append(f"Branch: {repo_data['default_branch']}")
    if repo_data.get("pushed_at"):
        meta.append(f"Last push: {repo_data['pushed_at'][:10]}")

    if meta:
        parts.append("*" + " | ".join(meta) + "*\n")

    # Topics
    topics = repo_data.get("topics", [])
    if topics:
        parts.append(f"**Topics:** {', '.join(topics)}\n")

    # Links
    if repo_data.get("html_url"):
        parts.append(f"**URL:** {repo_data['html_url']}")
    if repo_data.get("homepage") and repo_data["homepage"].startswith("http"):
        parts.append(f"**Homepage:** {repo_data['homepage']}")

    # README
    if readme:
        truncated = _smart_truncate(readme, 10000)
        parts.append(f"\n### README\n")
        parts.append(truncated)
    else:
        parts.append("\n*No README found.*")

    return "\n".join(parts)


def _format_github_file(owner: str, repo: str, path: str, content: str, branch: str) -> str:
    """Format GitHub file content as Markdown."""
    parts = [f"## github_file_content: {owner}/{repo}/{path}\n"]
    url = f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"
    parts.append(f"*Branch: {branch} | URL: {url}*\n")
    parts.append(content)
    return "\n".join(parts)


def _format_github_search_repos(items: list[dict], query: str, total: int, language: str | None) -> str:
    """Format GitHub repo search results."""
    parts = [f"## github_search_repos: {query}\n"]
    lang_suffix = f" ({language})" if language else ""
    parts.append(f"*{_human_count(total)} results{lang_suffix}*\n")

    for i, item in enumerate(items, 1):
        full_name = item.get("full_name", "")
        desc = item.get("description", "")
        stars = item.get("stargazers_count", 0)
        lang = item.get("language", "")
        url = item.get("html_url", "")

        parts.append(f"**[{i}] [{full_name}]({url})**")
        meta = []
        if lang:
            meta.append(lang)
        meta.append(f"★ {_human_count(stars)}")
        license_name = item.get("license", {}).get("spdx_id", "")
        if license_name and license_name != "NOASSERTION":
            meta.append(license_name)
        if item.get("archived"):
            meta.append("ARCHIVED")
        parts.append(f"*{' | '.join(meta)}*")
        if desc:
            parts.append(f"> {desc[:200]}")
        parts.append("")

    return "\n".join(parts)


def _format_github_issues_list(items: list[dict], owner: str, repo: str, state: str, issue_type: str) -> str:
    """Format GitHub issues/PRs list."""
    parts = [f"## github_issues: {owner}/{repo} ({state} {issue_type}s)\n"]

    for i, item in enumerate(items, 1):
        number = item.get("number", "")
        title = item.get("title", "")
        html_url = item.get("html_url", "")
        is_pr = "pull_request" in item
        type_label = "PR" if is_pr else "Issue"
        user = item.get("user", {}).get("login", "")
        labels = ", ".join(l["name"] for l in item.get("labels", []) if l.get("name"))
        comments = item.get("comments", 0)
        created = item.get("created_at", "")[:10]

        parts.append(f"**[#{number}] [{title}]({html_url})** ({type_label})")
        meta = [f"by @{user}" if user else "", created]
        if labels:
            meta.append(f"Labels: {labels}")
        meta.append(f"Comments: {comments}")
        parts.append(f"*{' | '.join(m for m in meta if m)}*")

        body = (item.get("body") or "")[:150]
        if body:
            parts.append(f"> {body}...")
        parts.append("")

    return "\n".join(parts)


def _format_github_issue_detail(issue: dict, pr_data: dict | None, comments: str) -> str:
    """Format a single GitHub issue/PR with full details."""
    number = issue.get("number", "")
    title = issue.get("title", "")
    is_pr = "pull_request" in issue
    type_label = "PR" if is_pr else "Issue"

    parts = [f"## github_issues: #{number} {title}\n"]

    # Metadata
    meta = []
    state = issue.get("state", "")
    meta.append(f"State: {state}")
    if issue.get("user", {}).get("login"):
        meta.append(f"Author: @{issue['user']['login']}")
    meta.append(f"Created: {issue.get('created_at', '')[:10]}")
    labels = ", ".join(l["name"] for l in issue.get("labels", []) if l.get("name"))
    if labels:
        meta.append(f"Labels: {labels}")
    if issue.get("assignees"):
        assignees = ", ".join("@" + a["login"] for a in issue["assignees"] if a.get("login"))
        if assignees:
            meta.append(f"Assignees: {assignees}")
    if issue.get("milestone", {}).get("title"):
        meta.append(f"Milestone: {issue['milestone']['title']}")

    if pr_data:
        if pr_data.get("merged"):
            meta.append("Merged: Yes")
        if pr_data.get("additions") is not None:
            meta.append(f"+{pr_data['additions']}/-{pr_data['deletions']} ({pr_data.get('changed_files', '?')} files)")

    parts.append(f"**{type_label}** " + "* | ".join(meta) + "*\n")

    # Body
    body = issue.get("body", "")
    if body:
        parts.append(body[:5000])
        if len(body) > 5000:
            parts.append("\n*[Body truncated]*")

    # URL
    if issue.get("html_url"):
        parts.append(f"\n*{issue['html_url']}*")

    # Comments
    if comments:
        parts.append(f"\n### Comments\n")
        parts.append(comments)

    return "\n".join(parts)


def _format_code_search(items: list[dict], query: str, total: int, language: str | None) -> str:
    """Format grep.app code search results."""
    lang_suffix = f" ({language})" if language else ""
    parts = [f"## code_search: {query}{lang_suffix}\n"]
    parts.append(f"*{_human_count(total)} results across GitHub*\n")

    for i, item in enumerate(items, 1):
        repo_info = item.get("repo", {})
        repo_name = repo_info.get("raw", "") if isinstance(repo_info, dict) else str(repo_info)
        path = item.get("path", {}).get("raw", "") if isinstance(item.get("path"), dict) else item.get("path", "")
        branch = item.get("branch", "main")

        # Content snippet
        content = item.get("content", "")
        if isinstance(content, dict):
            content = content.get("raw", "")
        snippet = content[:600] if content else ""

        url = f"https://github.com/{repo_name}/blob/{branch}/{path}"

        parts.append(f"**[{i}] [{repo_name}]({url})** — `{path}`")
        if snippet:
            parts.append(f"```\n{snippet}\n```")
        parts.append("")

    return "\n".join(parts)


def _format_pypi(info: dict, name: str) -> str:
    """Format PyPI package info."""
    parts = [f"## package_info: {name} (PyPI)\n"]

    # Core info
    if info.get("summary"):
        parts.append(f"**{info['summary']}**\n")

    meta = []
    if info.get("version"):
        meta.append(f"Latest: {info['version']}")
    if info.get("license"):
        meta.append(f"License: {info['license']}")
    if info.get("author"):
        meta.append(f"Author: {info['author']}")
    requires_python = info.get("requires_python", "")
    if requires_python:
        meta.append(f"Python: {requires_python}")
    if meta:
        parts.append("*" + " | ".join(meta) + "*\n")

    # Links
    links = []
    if info.get("home_page"):
        links.append(f"Homepage: {info['home_page']}")
    if info.get("project_url"):
        # project_url can be a string or list
        urls = info["project_url"] if isinstance(info["project_url"], list) else [info["project_url"]]
        for u in urls[:3]:
            links.append(str(u))
    if info.get("package_url"):
        links.append(f"PyPI: {info['package_url']}")
    if links:
        parts.append("\n".join(f"- {l}" for l in links) + "\n")

    # Dependencies
    deps = info.get("requires_dist") or []
    if deps:
        parts.append(f"**Dependencies ({len(deps)}):**")
        for d in deps[:20]:
            parts.append(f"- {d}")
        if len(deps) > 20:
            parts.append(f"- ... and {len(deps) - 20} more")
        parts.append("")

    return "\n".join(parts)


def _format_npm(data: dict, name: str, version: str | None = None) -> str:
    """Format npm package info."""
    parts = [f"## package_info: {name} (npm)\n"]

    latest = data.get("dist-tags", {}).get("latest", "")
    ver_data = data.get("versions", {}).get(latest, {})

    desc = data.get("description") or ver_data.get("description", "")
    if desc:
        parts.append(f"**{desc}**\n")

    meta = []
    if latest:
        meta.append(f"Latest: {latest}")
    if data.get("license"):
        meta.append(f"License: {data['license']}")
    if meta:
        parts.append("*" + " | ".join(meta) + "*\n")

    if data.get("homepage"):
        parts.append(f"- Homepage: {data['homepage']}")
    if data.get("repository", {}).get("url"):
        repo_url = data["repository"]["url"]
        repo_url = repo_url.removeprefix("git+").removesuffix(".git")
        parts.append(f"- Repository: {repo_url}")
    if data.get("bugs", {}).get("url"):
        parts.append(f"- Issues: {data['bugs']['url']}")
    parts.append("")

    # Dependencies
    deps = ver_data.get("dependencies", {})
    if deps:
        parts.append(f"**Dependencies ({len(deps)}):**")
        for dep, ver in list(deps.items())[:20]:
            parts.append(f"- {dep}: {ver}")
        if len(deps) > 20:
            parts.append(f"- ... and {len(deps) - 20} more")
        parts.append("")

    return "\n".join(parts)


def _format_crates(crate: dict, name: str) -> str:
    """Format crates.io package info."""
    parts = [f"## package_info: {name} (crates.io)\n"]

    if crate.get("description"):
        parts.append(f"**{crate['description']}**\n")

    meta = []
    if crate.get("max_version"):
        meta.append(f"Latest: {crate['max_version']}")
    if crate.get("downloads"):
        meta.append(f"Downloads: {_human_count(crate['downloads'])}")
    if crate.get("license"):
        meta.append(f"License: {crate['license']}")
    if crate.get("updated_at"):
        meta.append(f"Updated: {crate['updated_at'][:10]}")
    if meta:
        parts.append("*" + " | ".join(meta) + "*\n")

    if crate.get("homepage"):
        parts.append(f"- Homepage: {crate['homepage']}")
    if crate.get("repository"):
        parts.append(f"- Repository: {crate['repository']}")
    if crate.get("documentation"):
        parts.append(f"- Docs: {crate['documentation']}")
    parts.append("")

    # Categories / keywords
    keywords = crate.get("keywords", [])
    if keywords:
        parts.append(f"**Keywords:** {', '.join(keywords)}")
    categories = crate.get("categories", [])
    if categories:
        parts.append(f"**Categories:** {', '.join(categories)}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Content fetcher with fallback chain
# ---------------------------------------------------------------------------

async def fetch_content(
    url: str,
    max_length: int = 15000,
    return_format: str = "markdown",
    with_links: bool = False,
) -> ExtractedContent:
    """Fetch readable content from a URL with arxiv-aware -> Jina JSON -> trafilatura fallback."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL scheme: {url}. Must start with http:// or https://")

    client = _get_shared_client()

    # Arxiv-specific handling: Jina returns empty for arxiv HTML papers
    if "arxiv.org" in url:
        ec = await _fetch_arxiv(client, url)
        if ec and ec.content and len(ec.content.strip()) > 50:
            ec.content = _smart_truncate(ec.content, max_length)
            return ec

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
        if not _is_valid_url(r.url):
            continue
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
    source_labels = {"duckduckgo": "DuckDuckGo", "mojeek": "Mojeek", "bing": "Bing", "startpage": "Startpage"}
    source_label = source_labels.get(source, source)

    lines = [f"## {tool_name}: {query} ({len(results)} results)\n"]
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

server = Server("free-web-search", version="5.0.0")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web for information. Returns ranked results with titles, URLs, and snippets. "
                "Uses DuckDuckGo + Mojeek + Bing + Startpage in parallel (first-wins for speed). "
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
                "Uses DuckDuckGo + Mojeek + Bing + Startpage in parallel.\n\n"
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
        Tool(
            name="github_repo_info",
            description=(
                "Get repository info from GitHub: README, stars, language, license, topics. "
                "Returns structured metadata + README content.\n\n"
                "Examples:\n"
                '- github_repo_info(owner="anthropics", repo="claude-code")\n'
                '- github_repo_info(owner="pallets", repo="flask", include_readme=false)'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub owner/org (e.g. 'pallets')"},
                    "repo": {"type": "string", "description": "Repository name (e.g. 'flask')"},
                    "include_readme": {
                        "type": "boolean",
                        "description": "Include README content (default: true)",
                        "default": True,
                    },
                },
                "required": ["owner", "repo"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="github_file_content",
            description=(
                "Fetch a specific file's content from a GitHub repository. Returns raw code/text. "
                "Supports any text file: source code, config, markdown, etc.\n\n"
                "Examples:\n"
                '- github_file_content(owner="python", repo="cpython", path="Lib/os.py")\n'
                '- github_file_content(owner="pallets", repo="flask", path="pyproject.toml", branch="main")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub owner/org"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "path": {"type": "string", "description": "File path within repo (e.g. 'src/main.py')"},
                    "branch": {
                        "type": "string",
                        "description": "Git branch/tag (default: repo's default branch)",
                        "default": None,
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Max content length in characters (default 15000, max 50000)",
                        "default": 15000,
                    },
                },
                "required": ["owner", "repo", "path"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="github_search_repos",
            description=(
                "Search GitHub for repositories. Supports GitHub search syntax: language, stars, topics.\n\n"
                "Examples:\n"
                '- github_search_repos(query="python web framework")\n'
                '- github_search_repos(query="mcp server", language="typescript", min_stars=100)\n'
                '- github_search_repos(query="http client", language="rust", sort="stars")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query. Supports GitHub syntax."},
                    "language": {
                        "type": "string",
                        "description": "Filter by language (e.g. 'python', 'rust', 'typescript')",
                        "default": None,
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["stars", "forks", "updated"],
                        "description": "Sort by (default: stars)",
                        "default": "stars",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (1-30, default 10)",
                        "default": 10,
                    },
                    "min_stars": {
                        "type": "integer",
                        "description": "Minimum star count filter",
                        "default": None,
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="github_issues",
            description=(
                "List or fetch GitHub issues and pull requests. Get details, search within issues, filter by state.\n\n"
                "Examples:\n"
                '- github_issues(owner="pallets", repo="flask")\n'
                '- github_issues(owner="pallets", repo="flask", issue_number=5917)\n'
                '- github_issues(owner="microsoft", repo="vscode", issue_type="pr", state="closed", search="fix terminal")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub owner/org"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "issue_number": {
                        "type": "integer",
                        "description": "Fetch a specific issue/PR by number (omit to list)",
                        "default": None,
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Issue state filter (default: open)",
                        "default": "open",
                    },
                    "issue_type": {
                        "type": "string",
                        "enum": ["issue", "pr", "all"],
                        "description": "Filter by type (default: issue)",
                        "default": "issue",
                    },
                    "search": {
                        "type": "string",
                        "description": "Search within issue titles/bodies",
                        "default": "",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["updated", "created", "comments"],
                        "description": "Sort by (default: updated)",
                        "default": "updated",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results for listing (1-30, default 10)",
                        "default": 10,
                    },
                },
                "required": ["owner", "repo"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="code_search",
            description=(
                "Search code across GitHub repositories. Finds usage patterns, examples, and implementations. "
                "Powered by grep.app (free, no API key).\n\n"
                "Examples:\n"
                '- code_search(query="asyncio.run")\n'
                '- code_search(query="FastAPI decorator", language="python")\n'
                '- code_search(query="useEffect cleanup", language="typescript")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Code search query"},
                    "language": {
                        "type": "string",
                        "description": "Filter by language (e.g. 'python', 'rust', 'typescript')",
                        "default": None,
                    },
                    "repo": {
                        "type": "string",
                        "description": "Restrict to a specific repo (e.g. 'python/cpython')",
                        "default": None,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (1-20, default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            annotations=READ_ONLY_HINT,
        ),
        Tool(
            name="package_info",
            description=(
                "Look up package metadata from PyPI, npm, or crates.io. "
                "Returns version, dependencies, license, links.\n\n"
                "Examples:\n"
                '- package_info(name="flask", registry="pypi")\n'
                '- package_info(name="react", registry="npm")\n'
                '- package_info(name="tokio", registry="crates")'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Package name"},
                    "registry": {
                        "type": "string",
                        "enum": ["pypi", "npm", "crates"],
                        "description": "Package registry (default: pypi)",
                        "default": "pypi",
                    },
                    "version": {
                        "type": "string",
                        "description": "Specific version (default: latest)",
                        "default": None,
                    },
                },
                "required": ["name"],
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
                # Parse site: operators from query
                clean_query, site_domains = _parse_site_operator(query)
                # Merge site: domains with explicit include_domains
                if site_domains:
                    if include_domains:
                        include_domains = list(set(include_domains) | set(site_domains))
                    else:
                        include_domains = site_domains

                results = await _parallel_search(clean_query, max_results, time_range, language)
                results = post_process_results(results)
                results = _apply_domain_filter(results, include_domains, exclude_domains)
                if not results:
                    return error_result(
                        f"No results found for: {clean_query}. "
                        "Try rephrasing, simplifying the query, or removing domain filters."
                    )
                return [TextContent(type="text", text=format_search_results(results, clean_query, "web_search"))]
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
                # Parse site: operators from query
                clean_query, site_domains = _parse_site_operator(query)
                # Merge site: domains with explicit include_domains
                if site_domains:
                    if include_domains:
                        include_domains = list(set(include_domains) | set(site_domains))
                    else:
                        include_domains = site_domains

                results = await _parallel_search(clean_query, max_results, time_range, language)
                results = post_process_results(results)
                results = _apply_domain_filter(results, include_domains, exclude_domains)
                if not results:
                    return error_result(
                        f"No results found for: {clean_query}. "
                        "Try rephrasing, simplifying the query, or removing domain filters."
                    )
                return [TextContent(type="text", text=format_search_results(results, clean_query, "news_search"))]
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
                wiki_query = _extract_wiki_query(query)
                instant_task = asyncio.create_task(get_instant_answer(query))
                wiki_task = asyncio.create_task(get_wiki_summary(wiki_query))
                search_task = asyncio.create_task(
                    _parallel_search(query, 5, language="en")
                )

                # Use return_exceptions=True so DDG failure doesn't kill the whole call
                gathered = await asyncio.gather(
                    instant_task, wiki_task, search_task, return_exceptions=True
                )

                instant_data = {}
                if isinstance(gathered[0], dict):
                    instant_data = gathered[0]
                elif isinstance(gathered[0], Exception):
                    logger.warning("auto_answer: instant_answer failed: %s", gathered[0])

                wiki_data = {}
                if isinstance(gathered[1], dict):
                    wiki_data = gathered[1]
                elif isinstance(gathered[1], Exception):
                    logger.warning("auto_answer: wiki_summary failed: %s", gathered[1])

                search_results = []
                if isinstance(gathered[2], list):
                    search_results = gathered[2]
                elif isinstance(gathered[2], Exception):
                    logger.warning("auto_answer: search failed: %s", gathered[2])

                # Post-process search results
                search_results = post_process_results(search_results)

                return [TextContent(type="text", text=format_auto_answer(
                    query, instant_data, wiki_data, search_results, wiki_query
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

        elif name == "github_repo_info":
            owner = arguments.get("owner", "")
            repo = arguments.get("repo", "")
            if not owner or not repo:
                return error_result("Error: 'owner' and 'repo' are required")
            include_readme = arguments.get("include_readme", True)
            try:
                result = await github_repo_info(owner, repo, include_readme)
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return error_result(f"GitHub repo info error: {e}")

        elif name == "github_file_content":
            owner = arguments.get("owner", "")
            repo = arguments.get("repo", "")
            path = arguments.get("path", "")
            if not owner or not repo or not path:
                return error_result("Error: 'owner', 'repo', and 'path' are required")
            branch = arguments.get("branch")
            max_length = min(arguments.get("max_length", 15000), 50000)
            try:
                result = await github_file_content(owner, repo, path, branch, max_length)
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return error_result(f"GitHub file content error: {e}")

        elif name == "github_search_repos":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")
            language = arguments.get("language")
            sort = arguments.get("sort", "stars")
            max_results = max(1, min(30, arguments.get("max_results", 10)))
            min_stars = arguments.get("min_stars")
            try:
                result = await github_search_repos(query, language, sort, max_results, min_stars)
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return error_result(f"GitHub search error: {e}")

        elif name == "github_issues":
            owner = arguments.get("owner", "")
            repo = arguments.get("repo", "")
            if not owner or not repo:
                return error_result("Error: 'owner' and 'repo' are required")
            issue_number = arguments.get("issue_number")
            state = arguments.get("state", "open")
            issue_type = arguments.get("issue_type", "issue")
            search = arguments.get("search", "")
            sort = arguments.get("sort", "updated")
            max_results = max(1, min(30, arguments.get("max_results", 10)))
            try:
                result = await github_issues(owner, repo, issue_number, state, issue_type, search, sort, max_results)
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return error_result(f"GitHub issues error: {e}")

        elif name == "code_search":
            query = arguments.get("query", "")
            if not query:
                return error_result("Error: 'query' is required")
            language = arguments.get("language")
            repo = arguments.get("repo")
            max_results = max(1, min(20, arguments.get("max_results", 10)))
            try:
                result = await code_search(query, language, repo, max_results)
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return error_result(f"Code search error: {e}")

        elif name == "package_info":
            pkg_name = arguments.get("name", "")
            if not pkg_name:
                return error_result("Error: 'name' is required")
            registry = arguments.get("registry", "pypi")
            version = arguments.get("version")
            try:
                result = await package_info(pkg_name, registry, version)
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return error_result(f"Package info error: {e}")

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
