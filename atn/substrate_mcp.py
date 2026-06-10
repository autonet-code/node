"""Substrate capsule MCP server — populate the autonet substrate from Claude Code.

A deliberately thin, standalone stdio MCP server (no ATN Runtime, no package
imports — runnable as a plain script). It gives any MCP client (Claude Code
via ``cc.bat --substrate``) three tools:

  substrate_capsule  Record a distilled work insight ("the essence of how to
                     do something") as a capsule. Written as a two-line
                     conversation JSONL in the ATN agent-conversation schema,
                     so the daemon's world_substrate_feed picks it up as one
                     (problem, resolution, outcome) work unit — no new
                     ingestion path exists; capsules ride the same rail as
                     ATN agent work.

  substrate_query    Read the equilibrated world: fetches nodes from the
                     running daemon (rpb_substrate_nodes over the local WS)
                     and keyword-ranks them against the query.

  substrate_status   Diagnostics: capsule count, daemon reachability, and
                     whether the capsule agent is provisioned/registered
                     (capsules only FLOW into the substrate once the agent
                     id exists in the daemon registry with an on-chain
                     identity — until then they accumulate on disk).

Identity: the capsule agent id comes from ``ATN_SUBSTRATE_AGENT`` or, like
the voice service, from ``CLAUDE_NAME`` (the ``--name`` passed to cc.bat),
prefixed ``cc-``. Falls back to ``claude-code``.

Design notes (v1, deliberate):
  - The WORKER distills. The capsule text is synthesized by the Claude
    instance that did the work, from hot context — no second parse, no
    separate distiller agent. The human triggers it ("log that to the
    substrate") and judges worthiness.
  - File write is offline-capable; the daemon nudge is best-effort. A
    daemon restart re-walks all conversations, so nothing is ever lost.
  - No auto-provisioning: creating an agent in the daemon triggers an
    immediate run (costs tokens). substrate_status tells the user what is
    missing instead.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

log = logging.getLogger("substrate_mcp")

DAEMON_WS_URL = os.environ.get("ATN_DAEMON_WS", "ws://127.0.0.1:7700")
ATN_DATA_DIR = Path(os.environ.get("ATN_DATA_DIR", str(Path.home() / ".atn")))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def capsule_agent_id() -> str:
    """Resolve the agent id capsules are attributed to.

    Priority: ATN_SUBSTRATE_AGENT > cc-<CLAUDE_NAME> > claude-code.
    Sanitized to registry-safe characters.
    """
    explicit = os.environ.get("ATN_SUBSTRATE_AGENT", "").strip()
    if explicit:
        raw = explicit
    else:
        name = os.environ.get("CLAUDE_NAME", "").strip()
        raw = f"cc-{name}" if name else "claude-code"
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-").lower()
    return sanitized or "claude-code"


# ---------------------------------------------------------------------------
# Capsule write (offline-capable)
# ---------------------------------------------------------------------------

def write_capsule(problem: str, insight: str, context: str = "") -> Path:
    """Write one capsule as a two-line conversation JSONL.

    Schema matches atn.conversation.ConversationTurn / the files
    world_substrate_feed walks: the user turn is the problem (feed's
    ``_extract_problem``), the assistant turn is the distilled insight
    (``_extract_resolution``). One file == one work unit.
    """
    agent_id = capsule_agent_id()
    conv_dir = ATN_DATA_DIR / "agents" / agent_id / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    path = conv_dir / f"capsule_{stamp}_{uuid.uuid4().hex[:8]}.jsonl"

    problem_text = problem.strip()
    if context.strip():
        problem_text += f"\n\n[Context]\n{context.strip()}"
    # All local Claude instances share one capsule agent (one wallet = one
    # on-chain identity), so per-instance provenance rides in the capsule
    # body instead of the agent id.
    instance = os.environ.get("CLAUDE_NAME", "").strip()
    if instance:
        problem_text += f"\n\n[Instance: {instance}]"

    lines = [
        {"role": "user", "content": problem_text, "timestamp": now.isoformat()},
        {"role": "assistant", "content": insight.strip(), "timestamp": now.isoformat()},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def capsule_count() -> int:
    conv_dir = ATN_DATA_DIR / "agents" / capsule_agent_id() / "conversations"
    if not conv_dir.is_dir():
        return 0
    return sum(1 for _ in conv_dir.glob("capsule_*.jsonl"))


# ---------------------------------------------------------------------------
# Daemon WS client (best-effort)
# ---------------------------------------------------------------------------

async def daemon_request(payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any] | None:
    """One request/response against the daemon's privileged local listener.

    The local listener pre-auths the connection, but the server pushes
    unsolicited frames (snapshot on connect, event stream) — so match
    replies by msg_id rather than taking the next frame.

    Returns the reply dict, or None if the daemon is unreachable.
    """
    try:
        import websockets
    except ImportError:
        return None
    msg_id = uuid.uuid4().hex[:12]
    payload = {**payload, "msg_id": msg_id}
    try:
        async with websockets.connect(DAEMON_WS_URL, open_timeout=5) as ws:
            await ws.send(json.dumps(payload))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, remaining))
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("msg_id") == msg_id:
                    return msg
            return None
    except Exception as exc:
        log.debug("daemon_request failed: %s", exc)
        return None


async def nudge_feed() -> bool:
    """Tell the daemon a capsule landed so the substrate feed counts it
    toward its next cycle. Best-effort — False just means 'daemon will
    find it later' (next cycle threshold or restart re-walk)."""
    reply = await daemon_request({
        "type": "substrate_feed_nudge",
        "agent_id": capsule_agent_id(),
    }, timeout=5.0)
    return bool(reply and reply.get("ok"))


# ---------------------------------------------------------------------------
# Query (keyword rank over daemon-provided nodes)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2}


async def query_substrate(query: str, max_results: int = 8) -> dict[str, Any]:
    reply = await daemon_request(
        {"type": "rpb_substrate_nodes", "max_nodes": 500},
        timeout=20.0,
    )
    if reply is None:
        return {"error": "Daemon not reachable on " + DAEMON_WS_URL +
                         " — start it with `python -m atn` to query the substrate."}
    if not reply.get("ok"):
        return {"error": reply.get("error", "substrate query failed")}
    items = (reply.get("result") or {}).get("items", [])
    q_terms = _terms(query)
    if not q_terms:
        return {"error": "Query has no usable terms."}

    scored: list[tuple[float, dict[str, Any]]] = []
    seen_nodes: set[str] = set()
    for item in items:
        node_id = str(item.get("node_id") or item.get("id") or "")
        if node_id and node_id in seen_nodes:
            continue        # same node can appear under multiple tendencies
        content = str(item.get("content", "") or item.get("label", ""))
        if not content:
            continue
        overlap = len(q_terms & _terms(content))
        if overlap == 0:
            continue
        if node_id:
            seen_nodes.add(node_id)
        scored.append((overlap / len(q_terms), item))
    scored.sort(key=lambda t: t[0], reverse=True)

    return {
        "query": query,
        "nodes_searched": len(items),
        "matches": [
            {
                "relevance": round(score, 3),
                "content": str(item.get("content", "") or item.get("label", ""))[:600],
                "node_id": item.get("node_id") or item.get("id"),
                "score": item.get("score"),
            }
            for score, item in scored[:max_results]
        ],
    }


async def status() -> dict[str, Any]:
    agent_id = capsule_agent_id()
    out: dict[str, Any] = {
        "agent_id": agent_id,
        "capsules_on_disk": capsule_count(),
        "capsule_dir": str(ATN_DATA_DIR / "agents" / agent_id / "conversations"),
        "daemon_url": DAEMON_WS_URL,
    }
    reply = await daemon_request({"type": "snapshot"}, timeout=10.0)
    if reply is None:
        out["daemon"] = "unreachable"
        out["note"] = ("Capsules are written to disk regardless; the daemon "
                       "ingests them when running.")
        return out
    out["daemon"] = "running"
    # snapshot "agents" is a dict keyed by agent id (see runtime/snapshot.py)
    agents = (reply.get("result") or {}).get("agents", {})
    entry = agents.get(agent_id) if isinstance(agents, dict) else None
    if entry is None:
        out["agent_provisioned"] = False
        out["note"] = (f"Agent '{agent_id}' is not in the daemon registry. "
                       "Capsules accumulate on disk but only flow into the "
                       "substrate once the agent exists and has an on-chain "
                       "identity (register it via the frontend or CLI).")
    else:
        out["agent_provisioned"] = True
        out["agent_address"] = entry.get("agent_address", "")
        out["agent_registered_on_chain"] = bool(entry.get("registered_on_chain"))
        if not entry.get("registered_on_chain"):
            out["note"] = (f"Agent '{agent_id}' exists but is not registered "
                           "on-chain — the substrate feed skips unregistered "
                           "agents; capsules wait on disk until registration.")
    return out


# ---------------------------------------------------------------------------
# MCP plumbing
# ---------------------------------------------------------------------------

TOOLS = [
    types.Tool(
        name="substrate_capsule",
        description=(
            "Record a distilled work insight into the autonet substrate. Call "
            "when the user asks to log something to the substrate, or when "
            "they flag a breakthrough worth keeping. Write the capsule as "
            "generalized, reusable knowledge — the essence of HOW to do the "
            "thing, not a log of what happened: state the problem class, the "
            "approach that works, WHY it works, and failure modes to avoid. "
            "Strip session-specific noise (paths, names) unless essential."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem class this insight addresses, stated generally (1-3 sentences).",
                },
                "insight": {
                    "type": "string",
                    "description": "The distilled how-to: approach, why it works, failure modes. This is the substrate entry's substance.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional: domain/tech context that scopes where the insight applies.",
                },
            },
            "required": ["problem", "insight"],
        },
    ),
    types.Tool(
        name="substrate_query",
        description=(
            "Search the equilibrated substrate world for existing knowledge "
            "relevant to a problem. Use before starting non-trivial work to "
            "check whether the network already knows how."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to know how to do."},
                "max_results": {"type": "integer", "description": "Max matches to return (default 8)."},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="substrate_status",
        description="Diagnostics: capsule agent identity, capsules on disk, daemon reachability, provisioning/registration state.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


def create_server() -> Server:
    server = Server("substrate")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        try:
            if name == "substrate_capsule":
                problem = (arguments.get("problem") or "").strip()
                insight = (arguments.get("insight") or "").strip()
                if not problem or not insight:
                    result: dict[str, Any] = {"error": "Both 'problem' and 'insight' are required."}
                else:
                    path = write_capsule(problem, insight, arguments.get("context", "") or "")
                    nudged = await nudge_feed()
                    result = {
                        "stored": str(path),
                        "agent_id": capsule_agent_id(),
                        "daemon_notified": nudged,
                    }
                    if not nudged:
                        result["note"] = ("Daemon not reachable — capsule saved to disk; "
                                          "it will be ingested when the daemon runs.")
            elif name == "substrate_query":
                result = await query_substrate(
                    (arguments.get("query") or "").strip(),
                    int(arguments.get("max_results", 8)),
                )
            elif name == "substrate_status":
                result = await status()
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            log.exception("Tool %s failed", name)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server


async def run() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [substrate-mcp] %(levelname)s: %(message)s")
    log.info("Substrate MCP server starting (agent_id=%s, data_dir=%s)",
             capsule_agent_id(), ATN_DATA_DIR)
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
