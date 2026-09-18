"""Web tools for non-bridge providers (Anthropic API, OpenAI, Gemini, Ollama).

BridgeProvider (Claude SDK / Codex subprocess) has WebSearch/WebFetch
built in and gets them via the "sdk_builtin" grant. Generic providers have
no native web access at all, so these Python equivalents ride the "web"
tool category the same way shell_tools.py rides "shell":

  - WEB_TOOLS:          tool definition dicts (name, description, input_schema)
  - WEB_TOOL_EXECUTORS: name -> async executor (input dict -> result dict)

Both executors take only an input dict (no Runtime), so like the shell
bundle they are LOCAL tools: the worker runs them in-process instead of
RPC-ing back to the daemon (see ExecutionEngine.is_authority_tool).

No API key is required. Search goes through DuckDuckGo's HTML endpoint,
which is keyless but unofficial: when its markup changes the search tool
returns an error rather than a wrong answer, and web_fetch keeps working.
"""
from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 atn-daemon")
_FETCH_TIMEOUT = 25.0
_SEARCH_TIMEOUT = 15.0
_DEFAULT_MAX_CHARS = 20_000
_HARD_MAX_CHARS = 100_000
_DDG_HTML = "https://html.duckduckgo.com/html/"

# ---------------------------------------------------------------------------
# Tool definitions (JSON-schema format expected by all providers)
# ---------------------------------------------------------------------------

WEB_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web. Returns a ranked list of results with title, "
            "url and snippet. Follow up with web_fetch on the urls worth "
            "reading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (default 8, max 20)",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch a URL and return its readable text (HTML is stripped to "
            "text; JSON and plain text come back as-is). Use for reading "
            "pages, docs and APIs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL"},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate the text to this many characters (default 20000)",
                    "default": _DEFAULT_MAX_CHARS,
                },
            },
            "required": ["url"],
        },
    },
]


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "td", "th", "table", "section", "article", "header", "footer",
    "nav", "aside", "pre", "blockquote", "hr", "dt", "dd", "form",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._skip == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.split("\n")]
        out: list[str] = []
        blank = 0
        for ln in lines:
            if not ln:
                blank += 1
                if blank <= 1:
                    out.append("")
                continue
            blank = 0
            out.append(ln)
        return "\n".join(out).strip()


def html_to_text(markup: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML document."""
    p = _TextExtractor()
    try:
        p.feed(markup)
        p.close()
    except Exception:  # malformed markup: keep whatever was parsed
        pass
    return p.title.strip(), p.text()


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

def _clamp(value: Any, default: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, hi))


def _check_url(url: str) -> str | None:
    u = urlparse(url or "")
    if u.scheme not in ("http", "https") or not u.netloc:
        return f"web_fetch needs an absolute http(s) URL, got {url!r}"
    return None


async def _web_fetch(inp: dict[str, Any]) -> dict[str, Any]:
    import httpx

    url = str(inp.get("url", "")).strip()
    err = _check_url(url)
    if err:
        return {"error": err}
    max_chars = _clamp(inp.get("max_chars"), _DEFAULT_MAX_CHARS, _HARD_MAX_CHARS)
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT, follow_redirects=True,
            headers={"User-Agent": _UA, "Accept": "text/html,application/json,text/*;q=0.9,*/*;q=0.5"},
        ) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        return {"error": f"web_fetch failed: {exc}"}

    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    title = ""
    if "html" in ctype or (not ctype and resp.text.lstrip()[:1] == "<"):
        title, text = html_to_text(resp.text)
    elif ctype.startswith("text/") or ctype in ("application/json", "application/xml",
                                               "application/javascript"):
        text = resp.text
    else:
        return {
            "error": f"web_fetch: unsupported content-type {ctype or 'unknown'} "
                     f"({len(resp.content)} bytes)",
            "status": resp.status_code, "url": str(resp.url),
        }
    truncated = len(text) > max_chars
    return {
        "url": str(resp.url),
        "status": resp.status_code,
        "content_type": ctype,
        "title": title,
        "text": text[:max_chars],
        "truncated": truncated,
        "total_chars": len(text),
    }


_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.S,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>'
    r'|<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip(s: str) -> str:
    return _html.unescape(_TAG_RE.sub("", s or "")).strip()


def _ddg_target(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<url>."""
    if href.startswith("//"):
        href = "https:" + href
    u = urlparse(href)
    if u.netloc.endswith("duckduckgo.com") and u.path.startswith("/l/"):
        target = parse_qs(u.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


def parse_ddg_html(markup: str, limit: int) -> list[dict[str, str]]:
    """Pull (title, url, snippet) triples out of a DuckDuckGo HTML page."""
    links = _RESULT_RE.findall(markup)
    snippets = [a or b for a, b in _SNIPPET_RE.findall(markup)]
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, (href, title_html) in enumerate(links):
        url = _ddg_target(_html.unescape(href))
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        results.append({
            "title": _strip(title_html),
            "url": url,
            "snippet": _strip(snippets[i]) if i < len(snippets) else "",
        })
        if len(results) >= limit:
            break
    return results


async def _web_search(inp: dict[str, Any]) -> dict[str, Any]:
    import httpx

    query = str(inp.get("query", "")).strip()
    if not query:
        return {"error": "web_search needs a non-empty query"}
    limit = _clamp(inp.get("max_results"), 8, 20)
    try:
        async with httpx.AsyncClient(
            timeout=_SEARCH_TIMEOUT, follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            resp = await client.post(_DDG_HTML, data={"q": query, "kl": "wt-wt"})
    except httpx.HTTPError as exc:
        return {"error": f"web_search failed: {exc}"}
    if resp.status_code != 200:
        return {"error": f"web_search: search backend replied {resp.status_code}"}
    results = parse_ddg_html(resp.text, limit)
    if not results:
        if "anomaly" in resp.text.lower() or "captcha" in resp.text.lower():
            return {"error": "web_search: search backend rate-limited this daemon; retry later"}
        return {"query": query, "results": [], "note": "no results parsed"}
    return {"query": query, "results": results}


WEB_TOOL_EXECUTORS: dict[str, Any] = {
    "web_search": _web_search,
    "web_fetch": _web_fetch,
}
