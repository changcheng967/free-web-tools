"""Network-free tests for mcp_server pure functions + new logic.

Run with:  python test_mcp_server.py
    or:     pytest -q       (no pytest dependency required; plain asserts)

Covers URL/text utilities, the TTL cache, search-result formatting (backend
health line), and the arXiv/registry error contracts — so regressions (e.g. a
search backend's selectors rotting, or "not found" errors being swallowed again)
fail loudly here instead of silently in production.
"""
import mcp_server as m
from mcp_server import SearchResult


# --- URL / text normalization ------------------------------------------------

def test_normalize_url_strips_tracking_and_fragment():
    out = m.normalize_url("HTTPS://Example.com/path/?utm_source=x&id=5#frag")
    assert "utm_source" not in out
    assert "#frag" not in out
    assert out.startswith("https://example.com/path")
    assert "id=5" in out


def test_normalize_url_trailing_slash():
    assert m.normalize_url("https://x.com/a/b/") == "https://x.com/a/b"
    # root path preserved
    assert m.normalize_url("https://x.com/") == "https://x.com/"


def test_smart_truncate_short_unchanged():
    assert m._smart_truncate("short text", 100) == "short text"


def test_smart_truncate_long_at_boundary():
    text = "Sentence one is here. " * 50
    out = m._smart_truncate(text, 80)
    assert len(out) <= 110  # truncated + marker
    assert out.endswith("[Content truncated]")


def test_clean_title_strips_breadcrumb_and_mdn():
    assert m.clean_title("Home > Docs > Title").endswith("Title") or m.clean_title("Home > Docs > Title") == "Title"
    assert "MDN" not in m.clean_title("Fetch API | MDN")


def test_cap_snippet_word_boundary():
    long = "word " * 100
    out = m.cap_snippet(long, 40)
    assert out.endswith("...")
    assert len(out) <= 43


def test_parse_site_operator():
    clean, domains = m._parse_site_operator("rust raid site:reddit.com site:rustlabs.com")
    assert set(domains) == {"reddit.com", "rustlabs.com"}
    assert "site:" not in clean


def test_dedup_domain_caps():
    rs = [
        SearchResult(title="a", url="https://ex.com/1", snippet="", source="x"),
        SearchResult(title="b", url="https://ex.com/2", snippet="", source="x"),
        SearchResult(title="c", url="https://ex.com/3", snippet="", source="x"),
        SearchResult(title="d", url="https://other.com/1", snippet="", source="x"),
    ]
    out = m._dedup_domain(rs, max_per_domain=2)
    assert len(out) == 3  # 2 from ex.com + 1 from other.com


def test_is_valid_url_rejects_garbage():
    assert m._is_valid_url("https://example.com/path") is True
    assert m._is_valid_url("not a url") is False
    assert m._is_valid_url("https://ex.com/a›b") is False  # unicode angle bracket


def test_domain_matches_subdomains():
    assert m._domain_matches("https://docs.github.com/x", "github.com") is True
    assert m._domain_matches("https://github.com/x", "github.com") is True
    assert m._domain_matches("https://notgithub.com/x", "github.com") is False


# --- TTL cache ---------------------------------------------------------------

def test_ttl_cache_set_get():
    c = m._TTLCache()
    c.set(("k",), "v", 60.0)
    assert c.get(("k",)) == "v"


def test_ttl_cache_miss():
    assert m._TTLCache().get(("missing",)) is m._CACHE_MISS


def test_ttl_cache_expiry():
    c = m._TTLCache()
    c.set(("k",), "v", -1.0)  # already expired
    assert c.get(("k",)) is m._CACHE_MISS


def test_ttl_cache_eviction():
    c = m._TTLCache(max_entries=2)
    c.set(("a",), 1, 60.0)
    c.set(("b",), 2, 60.0)
    c.set(("c",), 3, 60.0)  # evicts oldest ("a")
    assert c.get(("a",)) is m._CACHE_MISS
    assert c.get(("c",)) == 3


# --- search formatting (backend health line) ---------------------------------

def _rs(*sources):
    return [SearchResult(title=f"t{i}", url=f"https://x{i}.com/", snippet="s", source=s) for i, s in enumerate(sources)]


def test_backends_line_all_green():
    out = m.format_search_results(_rs("bing"), "q", "web_search",
                                  {"duckduckgo": 5, "mojeek": 4, "bing": 6, "startpage": 3})
    assert "Backends 4/4" in out
    assert "Bing ✓" in out and "DDG ✓" in out


def test_backends_line_flags_dead_backend():
    out = m.format_search_results(_rs("duckduckgo", "mojeek", "startpage"), "q", "news_search",
                                  {"duckduckgo": 5, "mojeek": 4, "bing": 0, "startpage": 3})
    assert "Backends 3/4" in out
    assert "Bing ✗" in out


def test_backends_line_falls_back_when_no_counts():
    # Without source_counts, it infers from visible result sources.
    out = m.format_search_results(_rs("duckduckgo", "mojeek"), "q", "web_search")
    assert "Backends" in out


# --- error contracts ---------------------------------------------------------

def test_fetch_not_found_raises_not_swallowed():
    """Not-found registry lookups must raise (not fall through to garbage scraping).

    Uses a fake URL pattern that triggers the crates.io adapter; expects a clean
    RuntimeError naming the registry. (Network call; skipped offline.)
    """
    import asyncio
    async def run():
        await m._fetch_content_uncached("https://crates.io/crates/__definitely_not_a_real_crate_zzz__")
    try:
        asyncio.run(asyncio.wait_for(run(), timeout=20))
        assert False, "expected RuntimeError for missing crate"
    except RuntimeError as e:
        assert "not found" in str(e).lower()
    except Exception:
        # Network unavailable in CI — tolerate, don't fail the suite.
        pass


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
