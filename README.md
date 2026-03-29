# Free Web Search MCP Server

Zero-cost web search, content extraction, and developer tools via MCP protocol. No API keys, no accounts, no limit.

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

## Tools (14)

| Tool | Purpose | Backend |
|------|---------|---------|
| `web_search` | General web search with time-range, language, domain filtering | DDG + Mojeek + Bing + Startpage (parallel) |
| `news_search` | News search (default time_range=week) | DDG + Mojeek + Bing + Startpage (parallel) |
| `fetch_url` | Extract content from URL with metadata, format options, links | Jina Reader JSON -> trafilatura |
| `deep_search` | Search + full content from top results (research tool) | Parallel search + parallel fetch |
| `instant_answer` | Factual/definitional answers (infobox, key facts) | DDG Instant Answer API + Wikipedia fallback |
| `wiki_summary` | Wikipedia article summary (all languages) | Wikipedia REST API |
| `auto_answer` | Comprehensive answer from multiple sources at once | DDG + Wikipedia + web search (parallel, fault-tolerant) |
| `related_searches` | Related/expanded query suggestions | DDG autocomplete + Bing Suggest fallback |
| `github_repo_info` | GitHub repo README + metadata (stars, language, license) | GitHub REST API + raw.githubusercontent.com |
| `github_file_content` | Fetch specific file content from GitHub repo | raw.githubusercontent.com (unlimited) |
| `github_search_repos` | Search GitHub for repositories | GitHub Search API |
| `github_issues` | List/fetch GitHub issues and pull requests | GitHub Issues/Search API |
| `code_search` | Search code across GitHub repos (grep.app) | grep.app API (free, unlimited) |
| `package_info` | Package metadata from PyPI, npm, crates.io | Registry JSON APIs |

### Tool Details

**`web_search`** — Search the web. Supports `site:` operator, time-range filtering (`day`, `week`, `month`, `year`), 25 language codes, and domain include/exclude (subdomain-aware).

**`news_search`** — Like `web_search` but defaults to `time_range="week"`.

**`fetch_url`** — Extract content from any URL. Returns Markdown (or plain text) with metadata header (Title | Author | Site | Date | Language | Method). Optional `with_links=true` appends all hyperlinks found on the page.

**`deep_search`** — One-shot research: searches the web, fetches full content from top 3 results in parallel. Returns combined Markdown with numbered references.

**`instant_answer`** — Best for factual queries ("What is X?", "Capital of Y?"). Returns summary, answer, key facts (infobox), definition.

**`wiki_summary`** — Best for encyclopedic topics. Supports all Wikipedia languages via `lang` param.

**`auto_answer`** — Comprehensive answer engine: fires DDG Instant Answer, Wikipedia, and web search in parallel.

**`related_searches`** — Get related/expanded query suggestions.

**`github_repo_info`** — Fetch a GitHub repository's README and structured metadata (stars, forks, language, license, topics, last push date). Supports `owner` + `repo` params.

**`github_file_content`** — Fetch a specific file from a GitHub repository. Returns raw file content. Supports branch selection and auto-detects default branch. Rejects binary files.

**`github_search_repos`** — Search GitHub for repositories. Supports GitHub search syntax (language, stars, topics). Sort by stars, forks, or recently updated.

**`github_issues`** — List, search, or fetch individual GitHub issues and pull requests. Filter by state (open/closed), type (issue/PR), search text. Fetch specific issue by number with full details and comments.

**`code_search`** — Search code across open-source GitHub repositories. Powered by grep.app (free, no API key). Supports language and repo filters.

**`package_info`** — Look up package metadata from PyPI, npm, or crates.io. Returns version, description, dependencies, license, and links.

## Architecture

- **Search**: DuckDuckGo Lite + Mojeek + Bing + Startpage (4-way parallel race, first-wins)
- **Answer engine**: DuckDuckGo Instant Answer API + Wikipedia fallback (structured factual data)
- **Encyclopedia**: Wikipedia REST Summary API (all languages)
- **Comprehensive**: `auto_answer` combines instant answer + Wikipedia + web search (fault-tolerant)
- **GitHub**: REST API (60 req/hr) + raw.githubusercontent.com (unlimited) for repo info, file content, issues/PRs, repo search
- **Code search**: grep.app API (free, unlimited) for cross-repo code search
- **Package registries**: PyPI JSON + npm Registry + crates.io API (all free, no keys)
- **Content extraction**: Jina AI Reader JSON (with format/links options) -> trafilatura bare_extraction() -> BeautifulSoup
- **Reliability**: Retry with exponential backoff, parallel backend racing, fallback chains
- **Quality**: URL normalization (30+ tracking params), domain dedup, snippet capping, title cleaning, subdomain-aware filtering, smart truncation
- **Performance**: Persistent HTTP/2 client, parallel search backends, parallel content fetch
- **Protocol**: MCP spec via official Python SDK, STDIO transport

## Dependencies

All free, pip-installable:
- `mcp[cli]` — MCP protocol SDK
- `httpx[http2]` — Async HTTP client
- `beautifulsoup4` + `lxml` — HTML parsing
- `trafilatura` — Content extraction
