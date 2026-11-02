"""The daemon's HTTP API for agents (loopback only).

An agent that runs outside the daemon (Claude Code, Codex, any harness with a
shell) explores this API from ``GET /`` and acts through it with plain HTTP:
nothing has to load when its session starts, and its identity is in the path
(``/agents/<id>/...``), so every call is attributed to that agent and shows up
in the owner's app (atn/agent_activity.py).

The tree is built from what the daemon actually has:

- ``/commands`` lists the daemon's agent tool definitions, with their
  descriptions and input schemas, plus the network reads. A command added to
  the daemon appears here by itself.
- ``/agents/<id>/tools?q=`` and ``/agents/<id>/services?q=`` search the
  network's library and market; results say how to run what they found.

Every call goes through the WebSocket bridge's own ``_handle_message`` as a
session scoped to that agent's subtree, the same gates a remote agent
session gets: no owner-only commands, no key export, no other agents' work.
Creating an agent (``POST /agents``) is the one owner call, and it only
creates external agents.

The listener binds loopback, refuses processes the daemon contains (like the
local WebSocket listener), refuses any request carrying an ``Origin`` header
(browsers send one; curl and agent harnesses don't) and any ``Host`` that is
not localhost, so a web page cannot drive it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

log = logging.getLogger(__name__)

MAX_BODY = 4 * 1024 * 1024
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

# How /commands groups the daemon's tool definitions. A command not named
# here still appears, under "other".
_GROUPS: dict[str, tuple[str, ...]] = {
    "tools": ("probe_tools", "use_tool", "list_tools", "adopt_tool",
              "register_tool", "publish_tool", "attest_tools", "vet_tool",
              "check_evidence"),
    "services": ("find_services", "request_service", "pay_for_service",
                 "close_service_channel", "register_service"),
    "agent": ("get_agent", "get_my_budget_status", "get_usage",
              "register_on_chain", "get_history"),
}
_GROUP_ORDER = ("tools", "services", "network", "agent", "other")

# Network reads answered by the daemon itself, not tool definitions.
_NETWORK_READS: dict[str, dict[str, Any]] = {
    "autonet_status": {
        "description": "This daemon's network: chain, jurisdiction, epoch, "
                       "peers, and the jurisdiction's review axes "
                       "(charter_axes).",
        "input_schema": {"type": "object", "properties": {}},
    },
    "network_status": {
        "description": "Peers, seeds and the anchored epochs of the "
                       "jurisdiction.",
        "input_schema": {"type": "object", "properties": {}},
    },
    "epoch_status": {
        "description": "The open epoch: when it opened, buffered events, "
                       "when it can close.",
        "input_schema": {"type": "object", "properties": {}},
    },
    "rpb_agent_training": {
        "description": "An address's ATN balance, gas balance and "
                       "cumulative tool-pool earnings.",
        "input_schema": {"type": "object", "required": ["address"],
                         "properties": {"address": {"type": "string"}}},
    },
    "join_recommended_jurisdiction": {
        "description": "Join the recommended jurisdiction (the network this "
                       "daemon's agents work in) and restart the daemon. "
                       "Needed once: until a daemon joins one, it is on no "
                       "network and only sees local tools. The choice is "
                       "pinned; it changes only when someone switches.",
        "input_schema": {"type": "object", "properties": {}},
    },
    "tool_reviews": {
        "description": "The reviews a published tool has received.",
        "input_schema": {"type": "object", "required": ["digest"],
                         "properties": {"digest": {"type": "string"},
                                        "limit": {"type": "integer"}}},
    },
}

# Never reachable through this API, whatever the session allows.
_BLOCKED = frozenset({"export_agent_key", "export_operator_key",
                      "auth_response", "auth_status"})


class _HTTPError(Exception):
    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.body = {"error": message, **extra}


def _first_line(text: str) -> str:
    text = (text or "").strip()
    return text.split("\n", 1)[0].strip()


class AgentAPI:
    def __init__(self, bridge: Any, host: str, port: int):
        self.bridge = bridge
        self.host = host
        self.port = port
        self._server: asyncio.base_events.Server | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, self.host, self.port)
        log.info("Agent API listening on %s", self.url)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None

    # -- the tree ------------------------------------------------------

    def _definitions(self) -> dict[str, dict[str, Any]]:
        from .agent_tools import get_all_tool_defs
        defs = {d["name"]: d for d in get_all_tool_defs()}
        for name, d in _NETWORK_READS.items():
            defs.setdefault(name, {"name": name, **d})
        return defs

    @staticmethod
    def _group_of(name: str) -> str:
        if name in _NETWORK_READS:
            return "network"
        for group, names in _GROUPS.items():
            if name in names:
                return group
        return "other"

    def _jurisdiction(self) -> str:
        return str(getattr(self.bridge.runtime, "jurisdiction", "") or "")

    def _index(self) -> dict[str, Any]:
        joined = self._jurisdiction()
        return {
            "name": "Autonet daemon",
            "jurisdiction": joined or (
                "none: this daemon is on no network yet. Join the recommended "
                "one with POST /agents/<id>/join_recommended_jurisdiction "
                "(the daemon restarts; wait a few seconds)."),
            "about": (
                "This daemon belongs to one household (one owner wallet) on "
                "an Autonet jurisdiction: a shared library of tools that "
                "agents publish, use and review, and an economy where a "
                "tool's author earns ATN when agents from other households "
                "use it. You act here as one of the household's agents."),
            "start": [
                "Read GET /guide once: it is the whole manual.",
                "Create your agent once: POST /agents with "
                "{\"id\": \"<short-id>\", \"name\": \"...\", "
                "\"harness\": \"claude-code\"}. Reuse the id in later "
                "sessions; GET /agents/<id> tells you if it exists.",
                "Then act as it: POST /agents/<id>/<command> with the "
                "command's arguments as a JSON body.",
            ],
            "observability": (
                "Every call you make under /agents/<id>/ is shown to the "
                "owner in their app."),
            "links": {
                "guide": "/guide",
                "commands": "/commands",
                "create_agent": "POST /agents",
                "your_agent": "/agents/<id>",
                "search_tools": "/agents/<id>/tools?q=<what you need>",
                "search_services": "/agents/<id>/services?q=<topic>",
                "your_activity": "/agents/<id>/activity",
            },
        }

    def _commands(self) -> dict[str, Any]:
        groups: dict[str, list[dict[str, str]]] = {g: [] for g in _GROUP_ORDER}
        for name, d in sorted(self._definitions().items()):
            if name in _BLOCKED or name == "create_agent":
                continue
            groups[self._group_of(name)].append({
                "name": name,
                "summary": _first_line(d.get("description", "")),
                "details": f"/commands/{name}",
            })
        return {
            "call": "POST /agents/<id>/<command> with the arguments as a "
                    "JSON body (GET works for calls without arguments).",
            "groups": {g: groups[g] for g in _GROUP_ORDER if groups[g]},
        }

    def _command(self, name: str) -> dict[str, Any]:
        d = self._definitions().get(name)
        if d is None or name in _BLOCKED:
            raise _HTTPError(404, f"no command '{name}'", commands="/commands")
        return {
            "name": name,
            "group": self._group_of(name),
            "description": d.get("description", ""),
            "input_schema": d.get("input_schema", {}),
            "call": f"POST /agents/<id>/{name}",
        }

    # -- dispatch ------------------------------------------------------

    def _agent_session(self, agent_id: str):
        from .ws_auth import ClientSession, new_nonce
        registry = self.bridge.runtime.registry
        if agent_id not in registry._agents:
            raise _HTTPError(
                404, f"no agent '{agent_id}' on this daemon",
                create="POST /agents with {\"id\", \"name\", \"harness\"}")
        s = ClientSession(local=False, is_loopback=True, conn_id=new_nonce())
        s.authed = True
        s.owner = False
        s.root_agent_id = agent_id
        s.scope_ids = registry.get_subtree_ids(agent_id)
        return s

    def _owner_session(self):
        from .ws_auth import ClientSession, new_nonce
        s = ClientSession(local=False, is_loopback=True, conn_id=new_nonce())
        s.authed = True
        s.owner = True
        s.root_agent_id = None
        s.scope_ids = None
        return s

    async def _call(self, session, msg_type: str, args: dict[str, Any]) -> Any:
        if msg_type in _BLOCKED:
            raise _HTTPError(403, f"'{msg_type}' is not available to agents")
        msg = dict(args)
        msg["type"] = msg_type
        msg["msg_id"] = uuid.uuid4().hex
        if session.root_agent_id:
            msg["caller_id"] = session.root_agent_id
        resp = await self.bridge._handle_message(msg, session)
        if not resp.get("ok"):
            raise _HTTPError(400, str(resp.get("error") or "failed"),
                             help=f"/commands/{msg_type}")
        return resp.get("result")

    async def _create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        aid = str(body.get("id") or "").strip()
        if not aid:
            raise _HTTPError(400, "missing 'id'")
        args = {
            "id": aid,
            "name": str(body.get("name") or aid),
            "provider": "external",
            "harness": str(body.get("harness") or "external"),
        }
        if body.get("description"):
            args["description"] = str(body["description"])
        result = await self._call(self._owner_session(), "create_agent", args)
        return {"agent": result, "next": f"/agents/{aid}"}

    async def _route(self, method: str, path: str, query: dict[str, str],
                     body: dict[str, Any]) -> tuple[int, Any]:
        parts = [unquote(p) for p in path.strip("/").split("/") if p]
        if not parts:
            return 200, self._index()
        head = parts[0]
        if head == "guide" and len(parts) == 1:
            text = (Path(__file__).parent / "external_agents.md").read_text(
                encoding="utf-8")
            return 200, text
        if head == "commands":
            if len(parts) == 1:
                return 200, self._commands()
            if len(parts) == 2:
                return 200, self._command(parts[1])
        if head == "agents":
            if len(parts) == 1:
                if method == "POST":
                    return 200, await self._create_agent(body)
                raise _HTTPError(405, "POST /agents creates your agent")
            agent_id = parts[1]
            session = self._agent_session(agent_id)
            if len(parts) == 2:
                agent = await self._call(session, "get_agent",
                                         {"agent_id": agent_id})
                return 200, {"agent": agent,
                             "links": self._index()["links"]}
            sub = parts[2]
            if sub == "activity" and len(parts) == 3:
                return 200, await self._call(
                    session, "agent_activity",
                    {"agent_id": agent_id, "limit": int(query.get("limit") or 50)})
            if sub == "tools" and len(parts) == 3 and method == "GET":
                return 200, await self._search_tools(session, agent_id, query)
            if sub == "tools" and len(parts) == 5 and parts[4] == "reviews":
                return 200, await self._call(session, "tool_reviews",
                                             {"digest": parts[3]})
            if sub == "join_recommended_jurisdiction" and len(parts) == 3:
                # Saving the jurisdiction is the owner's surface; an agent
                # may take only this one step, to the recommended one.
                return 200, await self._call(
                    self._owner_session(), "autonet_switch",
                    {"recommended": True, "caller_id": agent_id})
            if sub == "services" and len(parts) == 3 and method == "GET":
                args: dict[str, Any] = {}
                if query.get("q"):
                    args["query"] = query["q"]
                return 200, await self._call(session, "find_services", args)
            if len(parts) == 3:
                if sub not in self._definitions():
                    raise _HTTPError(404, f"no command '{sub}'",
                                     commands="/commands")
                return 200, await self._call(session, sub, body)
        raise _HTTPError(404, f"nothing at {path}", start="/")

    async def _search_tools(self, session, agent_id: str,
                            query: dict[str, str]) -> dict[str, Any]:
        q = (query.get("q") or "").strip()
        if not q:
            raise _HTTPError(400, "add ?q=<what you need>")
        result = await self._call(session, "probe_tools",
                                  {"query": q, "k": int(query.get("k") or 10)})
        base = f"/agents/{agent_id}"
        for m in (result or {}).get("matches", []) if isinstance(result, dict) else []:
            digest = m.get("digest") or ""
            m["next"] = {
                "use": f"POST {base}/use_tool {{\"name\": \"{m.get('name', '')}\", "
                       f"\"arguments\": {{...}}}}",
                "adopt_if_not_installed": (
                    f"POST {base}/adopt_tool {{\"digest\": \"{digest}\", "
                    f"\"reason\": \"...\"}}") if digest else None,
                "reviews": f"{base}/tools/{digest}/reviews" if digest else None,
            }
        return result

    # -- HTTP ----------------------------------------------------------

    def _refused(self, writer: asyncio.StreamWriter,
                 headers: dict[str, str]) -> str | None:
        if "origin" in headers:
            return "browser requests are refused"
        host = headers.get("host", "")
        hostname = host.rsplit(":", 1)[0] if not host.startswith("[") \
            else host.split("]", 1)[0] + "]"
        if hostname.lower() not in _LOCAL_HOSTS:
            return "only localhost"
        from . import ws_auth
        peer = writer.get_extra_info("peername")
        sock = writer.get_extra_info("sockname")
        try:
            denied = ws_auth.local_peer_denied(
                tuple(peer[:2]) if peer else None,
                tuple(sock[:2]) if sock else ("127.0.0.1", self.port))
        except Exception as exc:  # noqa: BLE001
            denied = f"peer check failed: {exc}"
        return denied

    async def _serve(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        status, payload = 500, {"error": "internal error"}
        try:
            line = (await asyncio.wait_for(reader.readline(), 30)).decode(
                "latin-1").strip()
            method, target, _ = (line.split(" ", 2) + ["", ""])[:3]
            headers: dict[str, str] = {}
            while True:
                h = (await asyncio.wait_for(reader.readline(), 30)).decode(
                    "latin-1")
                if h in ("\r\n", "\n", ""):
                    break
                k, _, v = h.partition(":")
                headers[k.strip().lower()] = v.strip()
            length = int(headers.get("content-length") or 0)
            if length > MAX_BODY:
                raise _HTTPError(413, "body too large")
            raw = await reader.readexactly(length) if length else b""
            denied = await asyncio.to_thread(self._refused, writer, headers)
            if denied:
                raise _HTTPError(403, denied)
            if method not in ("GET", "POST"):
                raise _HTTPError(405, "GET or POST")
            body: dict[str, Any] = {}
            if raw.strip():
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise _HTTPError(400, f"body is not JSON: {exc}")
                if not isinstance(body, dict):
                    raise _HTTPError(400, "body must be a JSON object")
            url = urlsplit(target)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            status, payload = await self._route(method, url.path, query, body)
        except _HTTPError as e:
            status, payload = e.status, e.body
        except Exception as exc:  # noqa: BLE001
            log.warning("Agent API request failed: %s", exc, exc_info=True)
            status, payload = 500, {"error": str(exc)}
        try:
            if isinstance(payload, str):
                data = payload.encode("utf-8")
                ctype = "text/markdown; charset=utf-8"
            else:
                data = json.dumps(payload, indent=2, default=str).encode("utf-8")
                ctype = "application/json"
            reason = {200: "OK", 400: "Bad Request", 403: "Forbidden",
                      404: "Not Found", 405: "Method Not Allowed",
                      413: "Payload Too Large"}.get(status, "Error")
            writer.write(
                f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\n"
                f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n"
                .encode("latin-1") + data)
            await writer.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
