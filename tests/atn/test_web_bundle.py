"""The "web" tool bundle: daemon-side web_search/web_fetch for non-bridge providers.

Bridges (Claude SDK / Codex) carry WebSearch/WebFetch natively under
"sdk_builtin". API and local providers have nothing until the owner grants
"web"; these tests pin the pure parsing helpers and the grant plumbing.
"""
from __future__ import annotations

import pytest

from atn.agent_tools import _TOOL_CATEGORIES, resolve_tool_surface
from atn.delegate_prompts import build_common_base
from atn.web_tools import (
    WEB_TOOL_EXECUTORS,
    WEB_TOOLS,
    _web_fetch,
    html_to_text,
    parse_ddg_html,
)


def test_html_to_text_strips_chrome_and_keeps_title():
    markup = """<html><head><title> Hello  Page </title>
    <style>body{}</style><script>var x=1;</script></head>
    <body><nav>menu</nav><h1>Heading</h1><p>First &amp; second</p>
    <script>ignored()</script><div>Tail</div></body></html>"""
    title, text = html_to_text(markup)
    assert title == "Hello  Page".strip()
    assert "var x" not in text and "ignored" not in text
    assert text.split("\n")[0] == "menu"
    assert "Heading" in text and "First & second" in text and "Tail" in text
    assert "\n\n\n" not in text  # blank runs collapse


def test_parse_ddg_html_unwraps_redirects_and_dedups():
    page = """
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">One <b>bold</b></a>
    <a class="result__snippet" href="x">First snippet</a>
    <a class="result__a" href="https://example.org/b">Two</a>
    <div class="result__snippet">Second snippet</div>
    <a class="result__a" href="https://example.org/b">Two again</a>
    """
    rows = parse_ddg_html(page, limit=10)
    assert [r["url"] for r in rows] == ["https://example.com/a", "https://example.org/b"]
    assert rows[0]["title"] == "One bold"
    assert rows[0]["snippet"] == "First snippet"
    assert rows[1]["snippet"] == "Second snippet"
    assert parse_ddg_html(page, limit=1) == rows[:1]


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http():
    out = await _web_fetch({"url": "file:///etc/passwd"})
    assert "error" in out
    out = await _web_fetch({"url": "example.com"})
    assert "error" in out


@pytest.mark.asyncio
async def test_web_fetch_strips_html(monkeypatch):
    import httpx

    async def fake_get(self, url):
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"},
            text="<html><title>T</title><body><p>Body text</p></body></html>",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    out = await _web_fetch({"url": "https://example.com", "max_chars": 4})
    assert out["title"] == "T"
    assert out["text"] == "Body" and out["truncated"] is True
    assert out["total_chars"] == len("Body text")


def test_web_grant_plumbing():
    names = {t["name"] for t in WEB_TOOLS}
    assert names == set(WEB_TOOL_EXECUTORS) == {"web_search", "web_fetch"}
    # "web" is a category the resolver leaves to the execution engine.
    assert "web" in _TOOL_CATEGORIES and _TOOL_CATEGORIES["web"] == set()
    surface = resolve_tool_surface(["web", "observation"])
    assert not names & {t["name"] for t in surface}
    # The engine treats both web tools as LOCAL (worker-side), like shell.
    from atn.runtime.execution_engine import ExecutionEngine
    from atn.runtime.worker_loop import _is_authority_tool
    for n in names:
        assert ExecutionEngine.is_authority_tool(n) is False
        assert _is_authority_tool(n) is False
    assert ExecutionEngine.is_authority_tool("create_agent") is True


def test_prompt_note_only_with_grant():
    with_web = build_common_base(tool_categories=["web"])
    without = build_common_base(tool_categories=["observation"])
    assert "## Web tools" in with_web and "## Web tools" not in without
    # Every agent is told that file/web access are explicit grants.
    assert 'separate grants ("shell", "web")' in without
