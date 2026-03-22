"""MCP server entry point for the Browser Control connector.

Wraps the existing BrowserClient + CDP relay stack as MCP tools,
communicating over stdio (JSON-RPC).  The ATN ConnectorManager launches
this as a subprocess and speaks MCP client protocol to discover and
invoke tools.

Usage (standalone):
    python server.py          # starts relay+Chrome, serves MCP over stdio

Usage (via ATN):
    Launched automatically by ConnectorManager.start("browser_relay")
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Ensure all logging goes to stderr -- stdout is reserved for MCP protocol.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("browser_relay.mcp")

# Import bootstrap and BrowserClient from the existing browser.py
# (lives alongside this file in the connector package)
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from browser import BrowserClient, bootstrap  # noqa: E402

# -- Shared state --------------------------------------------------------------

_client: BrowserClient | None = None


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Lifespan handler: bootstrap the relay stack before serving tools."""
    global _client
    log.info("bootstrapping relay + Chrome...")
    bootstrap()  # sync: starts relay server + Chrome w/ extension
    _client = BrowserClient()
    await _client.connect()
    log.info("BrowserClient connected to relay -- MCP server ready")
    try:
        yield
    finally:
        log.info("MCP server shutting down")
        _client = None


# -- MCP Server ----------------------------------------------------------------

mcp = FastMCP(
    "Browser Control",
    lifespan=_lifespan,
)


def _require_client() -> BrowserClient:
    """Get the active BrowserClient or raise."""
    if _client is None:
        raise RuntimeError("BrowserClient not connected")
    return _client


# -- Tools ---------------------------------------------------------------------


@mcp.tool()
async def browser_status() -> str:
    """Get connection status of the browser relay."""
    c = _require_client()
    return json.dumps({
        "connected": c.ws is not None and not c.ws.closed,
        "session_id": c.session_id,
        "target_id": c.target_id,
    })


@mcp.tool()
async def browser_tabs() -> str:
    """List all open browser tabs with their IDs, titles, and URLs."""
    c = _require_client()
    resp = await c.cdp("Target.getTargets", session_id=False)
    targets = resp.get("result", {}).get("targetInfos", [])
    tabs = [
        {"id": t["targetId"], "title": t.get("title", ""), "url": t.get("url", "")}
        for t in targets
        if t.get("type") == "page"
    ]
    return json.dumps(tabs)


@mcp.tool()
async def browser_navigate(url: str, new_tab: bool = True) -> str:
    """Navigate to a URL. Opens a new tab by default, or navigates the current tab.

    Args:
        url: The URL to navigate to
        new_tab: If True, open in a new tab. If False, navigate the current tab.
    """
    c = _require_client()
    if new_tab or not c.session_id:
        resp = await c.cdp("Target.createTarget", {"url": url}, session_id=False)
        target_id = resp.get("result", {}).get("targetId")
        if not target_id:
            return json.dumps({"error": "Failed to create tab", "detail": str(resp)})
        resp = await c.cdp(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
            session_id=False,
        )
        sid = resp.get("result", {}).get("sessionId")
        c.session_id = sid
        c.target_id = target_id
        return json.dumps({"target_id": target_id, "session_id": sid, "url": url})
    else:
        resp = await c.cdp("Page.navigate", {"url": url})
        return json.dumps(resp.get("result", {}))


@mcp.tool()
async def browser_attach(target_id: str) -> str:
    """Attach the debugger to an existing browser tab by its target ID.

    Args:
        target_id: The target ID from browser_tabs()
    """
    c = _require_client()
    resp = await c.cdp(
        "Target.attachToTarget",
        {"targetId": target_id, "flatten": True},
        session_id=False,
    )
    sid = resp.get("result", {}).get("sessionId")
    if sid:
        c.session_id = sid
        c.target_id = target_id
    return json.dumps({"session_id": sid, "target_id": target_id})


@mcp.tool()
async def browser_eval(expression: str) -> str:
    """Evaluate a JavaScript expression in the current browser tab.

    Args:
        expression: JavaScript code to evaluate
    """
    c = _require_client()
    result = await c.evaluate(expression)
    return json.dumps(result)


@mcp.tool()
async def browser_click(selector: str) -> str:
    """Click an element matching a CSS selector.

    Args:
        selector: CSS selector for the element to click
    """
    c = _require_client()
    js = f"""
        (function() {{
            var el = document.querySelector({json.dumps(selector)});
            if (!el) return 'NOT_FOUND: ' + {json.dumps(selector)};
            el.scrollIntoView({{block: 'center'}});
            el.click();
            return 'clicked';
        }})()
    """
    result = await c.evaluate(js)
    return json.dumps(result)


@mcp.tool()
async def browser_type(
    selector: str,
    text: str,
    clear: bool = True,
    submit: bool = False,
) -> str:
    """Type text into an input element. Works with plain inputs, React/Vue, and contenteditable.

    Args:
        selector: CSS selector for the input element
        text: Text to type
        clear: Clear existing value before typing (default True)
        submit: Click the submit button after typing (default False)
    """
    c = _require_client()
    js = f"""
        (function() {{
            var el = document.querySelector({json.dumps(selector)});
            if (!el) return JSON.stringify({{value: 'NOT_FOUND', selector: {json.dumps(selector)}}});
            el.scrollIntoView({{block: 'center'}});
            el.focus();
            el.dispatchEvent(new Event('focus', {{bubbles: true}}));
            var tag = el.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {{
                var proto = tag === 'TEXTAREA'
                    ? Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')
                    : Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                var newVal = {json.dumps(clear)} ? {json.dumps(text)} : (el.value + {json.dumps(text)});
                if (proto && proto.set) {{ proto.set.call(el, newVal); }}
                else {{ el.value = newVal; }}
                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                return JSON.stringify({{value: 'typed', method: 'nativeSetter'}});
            }} else if (el.contentEditable === 'true' || el.getAttribute('contenteditable') !== null) {{
                if ({json.dumps(clear)}) {{ el.innerHTML = ''; }}
                document.execCommand('insertText', false, {json.dumps(text)});
                if (!el.textContent.includes({json.dumps(text[:20] if len(text) > 20 else text)})) {{
                    el.innerHTML += '<p>' + {json.dumps(text)} + '</p>';
                    el.dispatchEvent(new Event('input', {{bubbles: true}}));
                }}
                return JSON.stringify({{value: 'typed', method: 'contentEditable'}});
            }} else {{
                return JSON.stringify({{value: 'NOT_EDITABLE', tag: tag}});
            }}
        }})()
    """
    result = await c.evaluate(js)
    val = result.get("value")
    if isinstance(val, str):
        try:
            result = json.loads(val)
        except json.JSONDecodeError:
            pass

    if submit and result.get("value") == "typed":
        await asyncio.sleep(0.3)
        submit_js = """
            (function() {
                var btn = document.querySelector('button[data-testid="send-button"]')
                    || document.querySelector('button[type="submit"]')
                    || document.querySelector('form button:last-of-type');
                if (btn && !btn.disabled) { btn.click(); return 'submitted'; }
                var active = document.activeElement;
                if (active) {
                    active.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
                    active.dispatchEvent(new KeyboardEvent('keypress', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
                    active.dispatchEvent(new KeyboardEvent('keyup', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
                    return 'submitted_enter';
                }
                return 'no submit button';
            })()
        """
        submit_result = await c.evaluate(submit_js)
        result["submit"] = submit_result.get("value")

    return json.dumps(result)


@mcp.tool()
async def browser_screenshot(save_to_file: bool = False) -> str:
    """Capture a screenshot of the current tab.

    Args:
        save_to_file: If True, save to a temp file and return the path.
                      If False, return base64-encoded PNG data (truncated in text).
    """
    import base64
    import tempfile

    c = _require_client()
    if not c.session_id:
        return json.dumps({"error": "No session attached. Use browser_attach first."})
    resp = await c.cdp("Page.captureScreenshot", {"format": "png"})
    data = resp.get("result", {}).get("data", "")
    if save_to_file:
        import os
        path = os.path.join(tempfile.gettempdir(), "screenshot.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))
        return json.dumps({"path": path})
    return json.dumps({"data_length": len(data), "data": data[:200] + "..."})


@mcp.tool()
async def browser_text() -> str:
    """Get the visible text content of the current page."""
    c = _require_client()
    result = await c.evaluate("document.body.innerText")
    return json.dumps(result)


@mcp.tool()
async def browser_url() -> str:
    """Get the current page URL."""
    c = _require_client()
    result = await c.evaluate("window.location.href")
    return json.dumps(result)


@mcp.tool()
async def browser_title() -> str:
    """Get the current page title."""
    c = _require_client()
    result = await c.evaluate("document.title")
    return json.dumps(result)


@mcp.tool()
async def browser_wait(
    selector: str,
    timeout_ms: int = 5000,
    visible: bool = False,
) -> str:
    """Wait for a CSS selector to appear in the DOM. Polls every 200ms.

    Args:
        selector: CSS selector to wait for
        timeout_ms: Maximum wait time in milliseconds (default 5000)
        visible: If True, also require the element to be visible
    """
    c = _require_client()
    if not c.session_id:
        return json.dumps({"error": "No session attached"})
    interval = 0.2
    elapsed = 0.0
    timeout_s = timeout_ms / 1000.0

    while elapsed < timeout_s:
        if visible:
            js = f"""(function() {{
                var el = document.querySelector({json.dumps(selector)});
                if (!el) return JSON.stringify({{found: false}});
                var r = el.getBoundingClientRect();
                var vis = r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden';
                return JSON.stringify({{found: true, visible: vis, tag: el.tagName, text: (el.textContent || '').trim().substring(0, 200)}});
            }})()"""
        else:
            js = f"""(function() {{
                var el = document.querySelector({json.dumps(selector)});
                if (!el) return JSON.stringify({{found: false}});
                return JSON.stringify({{found: true, tag: el.tagName, text: (el.textContent || '').trim().substring(0, 200)}});
            }})()"""
        result = await c.evaluate(js)
        val = result.get("value")
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if parsed.get("found"):
                    if visible and not parsed.get("visible"):
                        pass
                    else:
                        parsed["elapsed_ms"] = int(elapsed * 1000)
                        return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
        await asyncio.sleep(interval)
        elapsed += interval

    return json.dumps({
        "found": False,
        "timeout": True,
        "elapsed_ms": int(elapsed * 1000),
        "selector": selector,
    })


@mcp.tool()
async def browser_observe() -> str:
    """Inject a MutationObserver into the current page to track DOM changes.

    Call browser_events() afterwards to drain buffered events.
    """
    c = _require_client()
    if not c.session_id:
        return json.dumps({"error": "No session attached"})
    observer_js = """
    (function() {
        if (window.__agentObserver) {
            return 'already_active (events: ' + window.__agentEvents.length + ')';
        }
        window.__agentEvents = [];
        var MAX = 200;
        function classify(el) {
            var tag = el.tagName ? el.tagName.toLowerCase() : '';
            var role = el.getAttribute ? (el.getAttribute('role') || '') : '';
            var cls = el.className && typeof el.className === 'string' ? el.className : '';
            var txt = (el.textContent || '').trim().substring(0, 150);
            if (role === 'dialog' || role === 'alertdialog' || tag === 'dialog')
                return {type: 'dialog', tag: tag, role: role, text: txt};
            if (cls.match(/modal|popup|overlay|toast|snackbar|banner|notification|drawer|dropdown-menu|lightbox/i))
                return {type: 'overlay', tag: tag, class: cls.substring(0,100), text: txt};
            if (el.getAttribute && (el.getAttribute('aria-live') || el.getAttribute('role') === 'status' || el.getAttribute('role') === 'alert'))
                return {type: 'live-region', role: role || el.getAttribute('aria-live'), text: txt};
            if (tag === 'form')
                return {type: 'form', action: (el.action || '').substring(0,120), text: txt.substring(0,80)};
            if (tag === 'iframe')
                return {type: 'iframe', src: (el.src || '').substring(0, 150)};
            if (tag === 'video' || tag === 'audio')
                return {type: 'media', tag: tag, src: (el.src || el.currentSrc || '').substring(0,120)};
            if (tag === 'img' && el.src && !el.src.startsWith('data:image/svg'))
                return {type: 'image', src: el.src.substring(0,200), alt: (el.alt || '').substring(0,100)};
            if (tag === 'canvas')
                return {type: 'canvas', w: el.width, h: el.height};
            return null;
        }
        var INTERESTING = '[role="dialog"], [role="alertdialog"], [role="alert"], [role="status"], ' +
            '[aria-live], dialog, form, iframe, video, audio, img, canvas, ' +
            '[class*="modal"], [class*="popup"], [class*="toast"], [class*="overlay"], ' +
            '[class*="snackbar"], [class*="banner"], [class*="notification"], ' +
            '[class*="drawer"], [class*="lightbox"]';
        window.__agentObserver = new MutationObserver(function(mutations) {
            for (var i = 0; i < mutations.length; i++) {
                var m = mutations[i];
                for (var j = 0; j < m.addedNodes.length; j++) {
                    var node = m.addedNodes[j];
                    if (node.nodeType !== 1) continue;
                    var info = classify(node);
                    if (info) { info.time = Date.now(); info.action = 'added'; window.__agentEvents.push(info); }
                    if (node.querySelectorAll) {
                        var children = node.querySelectorAll(INTERESTING);
                        for (var k = 0; k < children.length; k++) {
                            var ci = classify(children[k]);
                            if (ci) { ci.time = Date.now(); ci.action = 'added'; window.__agentEvents.push(ci); }
                        }
                    }
                }
                for (var j2 = 0; j2 < m.removedNodes.length; j2++) {
                    var rnode = m.removedNodes[j2];
                    if (rnode.nodeType !== 1) continue;
                    var ri = classify(rnode);
                    if (ri) { ri.time = Date.now(); ri.action = 'removed'; window.__agentEvents.push(ri); }
                }
            }
            if (window.__agentEvents.length > MAX) {
                window.__agentEvents = window.__agentEvents.slice(-MAX);
            }
        });
        window.__agentObserver.observe(document.body, { childList: true, subtree: true });
        return 'observer_started';
    })()
    """
    result = await c.evaluate(observer_js)
    return json.dumps(result)


@mcp.tool()
async def browser_events() -> str:
    """Drain buffered DOM events from the MutationObserver.

    Call browser_observe() first to start collecting events.
    """
    c = _require_client()
    if not c.session_id:
        return json.dumps({"error": "No session attached"})
    drain_js = """
    (function() {
        if (!window.__agentEvents) return JSON.stringify({active: false, events: []});
        var evts = window.__agentEvents.splice(0, window.__agentEvents.length);
        return JSON.stringify({active: true, count: evts.length, events: evts});
    })()
    """
    result = await c.evaluate(drain_js)
    val = result.get("value")
    if isinstance(val, str):
        try:
            return val  # already JSON
        except Exception:
            pass
    return json.dumps(result)


@mcp.tool()
async def browser_detach() -> str:
    """Detach the debugger from the current tab (keeps the tab open)."""
    c = _require_client()
    if not c.session_id:
        return json.dumps({"error": "No session attached"})
    sid = c.session_id
    tid = c.target_id
    # Remove marker
    try:
        await c.evaluate("""(function() {
            var M = "\\uD83D\\uDD34 ";
            if (window.__agentMarkerObserver) {
                window.__agentMarkerObserver.disconnect();
                delete window.__agentMarkerObserver;
            }
            if (document.title.startsWith(M)) document.title = document.title.slice(M.length);
        })()""")
    except Exception:
        pass
    resp = await c.cdp(
        "Target.detachFromTarget",
        {"sessionId": sid},
        session_id=False,
    )
    c.session_id = None
    c.target_id = None
    return json.dumps({"detached": tid, "session_id": sid})


@mcp.tool()
async def browser_close(target_id: str = "") -> str:
    """Close a browser tab.

    Args:
        target_id: The tab to close. If empty, closes the currently attached tab.
    """
    c = _require_client()
    tid = target_id or c.target_id
    if not tid:
        return json.dumps({"error": "No target to close"})
    resp = await c.cdp("Target.closeTarget", {"targetId": tid}, session_id=False)
    if tid == c.target_id:
        c.session_id = None
        c.target_id = None
    return json.dumps(resp.get("result", {}))


@mcp.tool()
async def browser_cdp(method: str, params: str = "{}", session: bool = True) -> str:
    """Execute a raw Chrome DevTools Protocol command.

    Args:
        method: CDP method name (e.g. "Page.reload", "DOM.getDocument")
        params: JSON string of CDP parameters (default "{}")
        session: If True, send with the current session. If False, send globally.
    """
    c = _require_client()
    try:
        p = json.loads(params)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON params: {params}"})
    sid = c.session_id if session else False
    resp = await c.cdp(method, p, session_id=sid)
    return json.dumps(resp)


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
