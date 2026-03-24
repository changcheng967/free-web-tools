# Free Web Search MCP Server

Zero-cost web search and content extraction via MCP protocol. No API keys, no accounts, no limits.

## Setup

```bash
# Install dependencies
pip install -e .

# Add to Claude Code
claude mcp add free-web-search -- python -m mcp_server

# Or add to Claude Desktop — edit claude_desktop_config.json:
# {
#   "mcpServers": {
#     "free-web-search": {
#       "command": "python",
#       "args": ["-m", "mcp_server"],
#       "cwd": "/path/to/free_web_tools"
#     }
#   }
# }
```

## Tools (8)

| Tool | Purpose | Backend |
|------|---------|---------|
| `web_search` | General web search with time-range, language, domain filtering | DDG + Mojeek + Qwant (parallel) |
| `news_search` | News search (default time_range=week) | DDG + Mojeek + Qwant (parallel) |
| `fetch_url` | Extract content from URL with metadata, format options, links | Jina Reader JSON -> trafilatura |
| `deep_search` | Search + full content from top results (research tool) | Parallel search + parallel fetch |
| `instant_answer` | Factual/definitional answers (infobox, key facts) | DDG Instant Answer API |
| `wiki_summary` | Wikipedia article summary (all languages) | Wikipedia REST API |
| `auto_answer` | Comprehensive answer from multiple sources at once | DDG + Wikipedia + web search (parallel) |
| `related_searches` | Related/expanded query suggestions | DDG autocomplete |

### Tool Details

**`web_search`** — Search the web. Supports `site:` operator, time-range filtering (`day`, `week`, `month`, `year`), 25 language codes, and domain include/exclude (subdomain-aware).

**`news_search`** — Like `web_search` but defaults to `time_range="week"`.

**`fetch_url`** — Extract content from any URL. Returns Markdown (or plain text) with metadata header (Title | Author | Site | Date | Language | Method). Optional `with_links=true` appends all hyperlinks found on the page. Optional `return_format="text"` for plain text output.

**`deep_search`** — One-shot research: searches the web, fetches full content from top 3 results in parallel. Returns combined Markdown with numbered references. Now supports `time_range` and `language` params.

**`instant_answer`** — Best for factual queries ("What is X?", "Capital of Y?"). Returns summary, answer, key facts (infobox), definition. Shows related topics and actionable suggestions when no direct answer is found.

**`wiki_summary`** — Best for encyclopedic topics. Supports all Wikipedia languages via `lang` param (e.g. `lang="de"`, `lang="zh"`). Returns extract, description, thumbnail, and timestamp.

**`auto_answer`** — Comprehensive answer engine: fires DDG Instant Answer, Wikipedia, and web search in parallel. Synthesizes a combined answer with key facts and web references. Best for complex questions that benefit from multiple sources.

**`related_searches`** — Get related/expanded query suggestions.

## Architecture

- **Search**: DuckDuckGo Lite + Mojeek + Qwant (3-way parallel race, first-wins)
- **Answer engine**: DuckDuckGo Instant Answer API (structured factual data)
- **Encyclopedia**: Wikipedia REST Summary API (all languages)
- **Comprehensive**: `auto_answer` combines instant answer + Wikipedia + web search
- **Content extraction**: Jina AI Reader JSON (with format/links options) -> trafilatura `bare_extraction()` -> BeautifulSoup
- **Reliability**: Retry with exponential backoff (callable factory, not coroutine), parallel backend racing
- **Quality**: URL normalization (30+ tracking params), domain dedup, snippet capping, title cleaning, subdomain-aware filtering, smart truncation
- **Performance**: Persistent HTTP/2 client, parallel search backends, parallel content fetch
- **Protocol**: MCP spec via official Python SDK, STDIO transport

## Dependencies

All free, pip-installable:
- `mcp[cli]` — MCP protocol SDK
- `httpx[http2]` — Async HTTP client
- `beautifulsoup4` + `lxml` — HTML parsing
- `trafilatura` — Content extraction
