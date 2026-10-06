#!/usr/bin/env python3
"""Deterministic model + registry stub for the live E2E harness.

One stdlib HTTP server, two jobs:

  * ``POST /v1/chat/completions`` — an OpenAI-compatible endpoint the isolated
    daemon registers as a custom provider (``providers.e2e_stub`` in its
    config.yaml; ``atn/runtime/provider_manager.py`` turns any unknown name with
    a ``base_url`` into an ``OpenAICompatibleProvider``). By default it never
    calls tools and replies ``"<MARKER> <last user message>"``, both plain JSON
    and ``stream: true`` SSE, so UI assertions on agent output are
    deterministic. Trigger words in the newest user message switch modes, so
    the app's tool-call, stall and provider-error paths can be driven live:

      - ``E2E-TOOL`` (or ``E2E-TOOL:<name>``): the first turn answers with one
        tool call (to ``<name>``, else the first tool the request offers) with
        empty arguments; once the tool result is back, a normal text reply.
      - ``E2E-STALL``: never answers (holds the request until the stub stops
        or ``--stall-seconds`` elapse, then drops the connection).
      - ``E2E-FAIL``: HTTP 503 with an error body (``retry-after: 0`` so the
        daemon's retries fail fast).
  * ``GET /registry.json`` — the network registry the daemon reads through
    ``ATN_REGISTRY_URL`` when it joins the network. It names ONLY the local
    hardhat chain and the contracts the harness deployed, so a join can never
    resolve Etherlink Shadownet addresses.

``GET /health`` returns ``{"ok": true, "requests": N}`` for the status check.

No dependency beyond the standard library: the harness launches it with the
same interpreter as the daemon, detached, and kills it on ``down``.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

MARKER = "E2E-STUB-REPLY"
MODEL_ID = "echo-1"
TRIGGER_TOOL = "E2E-TOOL"
TRIGGER_STALL = "E2E-STALL"
TRIGGER_FAIL = "E2E-FAIL"
FAIL_MESSAGE = "E2E stub forced failure: service unavailable"


def last_user_text(messages: List[Dict[str, Any]]) -> str:
    """The text of the newest user message, flattening content parts."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(p for p in parts if p)
    return ""


def reply_text(body: Dict[str, Any]) -> str:
    """The deterministic completion for a chat request."""
    said = last_user_text(body.get("messages") or []).strip()
    return f"{MARKER} {said}".strip()


def mode_for(body: Dict[str, Any]) -> str:
    """Which canned behaviour this request triggers: text, tool, stall, fail."""
    said = last_user_text(body.get("messages") or [])
    if TRIGGER_FAIL in said:
        return "fail"
    if TRIGGER_STALL in said:
        return "stall"
    if TRIGGER_TOOL in said:
        # Only the first turn calls the tool: once a tool result is the newest
        # message, finish with text so the run completes.
        msgs = body.get("messages") or []
        if msgs and msgs[-1].get("role") == "tool":
            return "text"
        return "tool"
    return "text"


def tool_call_for(body: Dict[str, Any]) -> Dict[str, Any]:
    """The single tool call an ``E2E-TOOL`` turn makes."""
    said = last_user_text(body.get("messages") or [])
    name = ""
    for word in said.split():
        if word.startswith(TRIGGER_TOOL + ":"):
            name = word.split(":", 1)[1].strip()
            break
    if not name:
        for tool in body.get("tools") or []:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                name = fn["name"]
                break
    return {
        "id": f"call_e2e_{int(time.time() * 1000)}",
        "type": "function",
        "function": {"name": name or "unknown_tool", "arguments": "{}"},
    }


def tool_completion(body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-e2e-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": None,
                        "tool_calls": [tool_call_for(body)]},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1,
                  "total_tokens": 11},
    }


def tool_stream_chunks(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SSE events for a streamed tool call: the call in one delta, then a
    finish chunk with ``finish_reason: tool_calls``."""
    model = body.get("model") or MODEL_ID
    call = tool_call_for(body)
    return [
        {"object": "chat.completion.chunk", "model": model,
         "choices": [{"index": 0, "delta": {"tool_calls": [{
             "index": 0, "id": call["id"], "type": "function",
             "function": call["function"]}]}, "finish_reason": None}]},
        {"object": "chat.completion.chunk", "model": model,
         "choices": [{"index": 0, "delta": {},
                      "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 1,
                   "total_tokens": 11}},
    ]


def completion(body: Dict[str, Any]) -> Dict[str, Any]:
    text = reply_text(body)
    return {
        "id": f"chatcmpl-e2e-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(text.split()),
                  "total_tokens": 10 + len(text.split())},
    }


def stream_chunks(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SSE events for a streamed completion: one content delta per word, then
    a finish chunk carrying usage (the shape openai_compat._recv_stream reads)."""
    text = reply_text(body)
    model = body.get("model") or MODEL_ID
    words = text.split(" ")
    chunks: List[Dict[str, Any]] = []
    for i, w in enumerate(words):
        piece = w if i == 0 else " " + w
        chunks.append({"object": "chat.completion.chunk", "model": model,
                       "choices": [{"index": 0, "delta": {"content": piece},
                                    "finish_reason": None}]})
    chunks.append({"object": "chat.completion.chunk", "model": model,
                   "choices": [{"index": 0, "delta": {},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10,
                             "completion_tokens": len(words),
                             "total_tokens": 10 + len(words)}})
    return chunks


class StubState:
    def __init__(self, registry_path: Path | None,
                 stall_seconds: float = 600.0) -> None:
        self.registry_path = registry_path
        self.stall_seconds = stall_seconds
        self.requests = 0
        self.lock = threading.Lock()
        # Set on shutdown so a held E2E-STALL request lets go at once.
        self.stopping = threading.Event()

    def bump(self) -> None:
        with self.lock:
            self.requests += 1


def make_handler(state: StubState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            return

        def _json(self, code: int, payload: Any) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._json(200, {"ok": True, "requests": state.requests})
            elif path == "/registry.json":
                if state.registry_path is None or not state.registry_path.exists():
                    self._json(404, {"error": "no registry"})
                    return
                self._json(200, json.loads(
                    state.registry_path.read_text(encoding="utf-8")))
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model"}]})
            else:
                self._json(404, {"error": f"no route {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid json"}})
                return
            if not path.endswith("/chat/completions"):
                self._json(404, {"error": f"no route {path}"})
                return
            state.bump()
            mode = mode_for(body)
            if mode == "fail":
                data = json.dumps({"error": {"message": FAIL_MESSAGE,
                                             "type": "server_error"}}).encode()
                self.send_response(503)
                self.send_header("content-type", "application/json")
                self.send_header("retry-after", "0")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if mode == "stall":
                state.stopping.wait(state.stall_seconds)
                self.close_connection = True
                return
            if not body.get("stream"):
                self._json(200, tool_completion(body) if mode == "tool"
                           else completion(body))
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            chunks = (tool_stream_chunks(body) if mode == "tool"
                      else stream_chunks(body))
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


class StubServer(ThreadingHTTPServer):
    """Releases held E2E-STALL requests when the server shuts down."""

    stub_state: StubState

    def shutdown(self) -> None:
        self.stub_state.stopping.set()
        super().shutdown()


def serve(host: str, port: int, registry_path: Path | None,
          stall_seconds: float = 600.0) -> ThreadingHTTPServer:
    state = StubState(registry_path, stall_seconds)
    server = StubServer((host, port), make_handler(state))
    server.stub_state = state
    server.daemon_threads = True
    return server


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--registry", default="",
                    help="path to the registry.json served at /registry.json")
    ap.add_argument("--stall-seconds", type=float, default=600.0,
                    help="how long an E2E-STALL request is held")
    args = ap.parse_args()
    server = serve(args.host, args.port,
                   Path(args.registry) if args.registry else None,
                   stall_seconds=args.stall_seconds)
    print(f"stub listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
