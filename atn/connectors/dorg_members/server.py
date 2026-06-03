"""MCP server for the dОrg member-agent roster.

Wraps the dОrg member API (the hackathon backend) so an autonet agent — the
support agent ("Kevin") — can register a member's external agent on request
and hand back its API token, all through conversation. This is the
"conversational registration" path: a Discord member asks Kevin to register
their agent; Kevin calls register_member_agent here.

Member agents are NOT autonet agents and NOT on-chain — they're external,
member-operated agents that report work into dОrg via the member API (token +
the claim_lead / surface_lead / send_message tools). This connector only covers
registration + lookup; the member's agent uses its own token against the member
API directly to push updates.

Config (env, injected by ConnectorManager):
  - DORG_API_BASE   base URL of the dОrg member API (e.g. http://127.0.0.1:8077)

Runs as a stdio MCP subprocess launched by ConnectorManager.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# All logging to stderr — stdout is the MCP protocol channel.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dorg_members.mcp")

API_BASE = os.environ.get("DORG_API_BASE", "").rstrip("/")

mcp = FastMCP("dОrg Member Agents")


async def _request(method: str, path: str, *, json_body: Any = None) -> dict:
    """Call the dОrg member API. Returns the parsed JSON (or an error dict)."""
    if not API_BASE:
        return {"error": "DORG_API_BASE is not configured"}
    url = f"{API_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.request(method, url, json=json_body)
    except Exception as exc:
        return {"error": f"request to {url} failed: {exc}"}
    if resp.status_code >= 400:
        # Surface the API's error message (e.g. 409 already-registered).
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text
        return {"error": f"{resp.status_code}: {detail or resp.reason_phrase}"}
    try:
        return resp.json()
    except Exception:
        return {"error": "non-JSON response from member API"}


@mcp.tool()
async def register_member_agent(
    discord_user_id: str,
    name: str,
    owner_address: str = "",
) -> str:
    """Register a member's external agent into the dОrg roster and return its
    API token (shown ONCE).

    Use this when a Discord member asks you to register their agent. The member
    operates the agent themselves on their own machine; this just issues them
    an identity + token so the agent can report work into dОrg.

    Args:
        discord_user_id: The requesting member's Discord user id (the numeric id).
        name: A short name for the agent.
        owner_address: Optional on-chain wallet address of the owner.

    Returns JSON with agent_id, name, and token. Give the member the token
    privately and tell them their agent uses it (Authorization header) to call
    the member API tools: claim_lead, surface_lead, send_message. The token is
    not retrievable again — if lost, re-register.
    """
    result = await _request(
        "POST", "/admin/agents",
        json_body={
            "discord_user_id": str(discord_user_id),
            "name": name,
            "owner_address": owner_address,
        },
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_member_agents() -> str:
    """List the registered member agents and their current scores/activity.

    Use this to check whether a member already has an agent, or to report on
    member-agent standings. Returns the roster with per-agent lead/score stats.
    """
    result = await _request("GET", "/scores")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    if not API_BASE:
        log.warning("DORG_API_BASE not set — register/list tools will return errors")
    mcp.run(transport="stdio")
