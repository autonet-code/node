"""The tool-side secret API — what a PINNED TOOL imports to read its secrets.

docs/tool_secret_binding.md. A tool that declared ``capabilities.secrets`` runs
with its OWN broker session bound to its PID, and reads values through it:

    from atn.tool_secret_api import get_secret, list_secrets

    token = get_secret("GITHUB_TOKEN")   # the value, in the tool's memory only

WHY THE TOOL GETS THE VALUE AND THE AGENT DOES NOT. The agent's worker receives
only ``{var_name, path}`` — an opaque handle — because the agent is a language
model whose whole output surface is a transcript. A tool is deterministic code
that was content-hashed before it ran; giving it the value is the point of the
binding. The exfiltration path this closes is the AGENT holding a secret it
only needs a TOOL to use.

SELF-CONTAINED BY NECESSITY. Adopted tools run under ``tool_guard`` with a
scrubbed environment and a deny-by-default audit hook, so this module must not
import from the ``atn`` package at call time or drag in optional dependencies.
It speaks the broker's newline-JSON protocol directly.

THE GUARD INTERACTION. Reading a secret needs the broker pipe/socket, which the
audit hook sees as a ``socket.*`` (POSIX) event. An adopted tool that declares
secrets therefore also needs ``net``. Weigh that when approving an adoption: a
tool with secrets AND net is a tool that can send them somewhere. The narrower
review is a tool whose declared ``authorized_hosts`` constrain where net may
go (change 2 in the spec).
"""
from __future__ import annotations

import json
import os
import sys

__all__ = ["get_secret", "list_secrets", "SecretUnavailable"]

_PIPE_NAME = r"\\.\pipe\vault-broker"
_TIMEOUT_S = 5.0


class SecretUnavailable(RuntimeError):
    """Raised when a declared secret cannot be obtained.

    Always a hard failure, never a silent empty string: a tool that proceeds
    with an empty credential produces a confusing downstream error, and worse,
    may fall back to some other path the author did not intend.
    """


def list_secrets() -> list[str]:
    """The service names bound to THIS tool call.

    Read from ``ATN_TOOL_SECRETS`` (names only, set by the daemon). This is an
    advertisement, not an authorization — the broker authorizes from the PID
    session. If the env var is absent the tool declared no secrets.
    """
    raw = os.environ.get("ATN_TOOL_SECRETS", "")
    return [s for s in (p.strip() for p in raw.split(",")) if s]


def _call(req: dict) -> dict:
    """One newline-JSON round-trip to the broker. Never raises; returns a dict."""
    payload = (json.dumps(req) + "\n").encode("utf-8")
    if sys.platform == "win32":
        return _call_win(payload)
    return _call_posix(payload)


def _call_win(payload: bytes) -> dict:
    try:
        with open(_PIPE_NAME, "r+b", buffering=0) as pipe:
            pipe.write(payload)
            pipe.flush()
            buf = bytearray()
            while b"\n" not in buf:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > (1 << 20):
                    break
    except OSError as exc:
        return {"ok": False, "error": f"broker unreachable: {exc}"}
    return _decode(buf)


def _call_posix(payload: bytes) -> dict:
    import socket

    path = os.environ.get("VAULT_BROKER_SOCK") or "/tmp/vault-broker.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_TIMEOUT_S)
            sock.connect(path)
            sock.sendall(payload)
            buf = bytearray()
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > (1 << 20):
                    break
    except OSError as exc:
        return {"ok": False, "error": f"broker unreachable: {exc}"}
    return _decode(buf)


def _decode(buf: bytearray) -> dict:
    if not buf:
        return {"ok": False, "error": "empty broker reply"}
    try:
        out = json.loads(bytes(buf).split(b"\n", 1)[0].decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return {"ok": False, "error": f"bad broker reply: {exc}"}
    return out if isinstance(out, dict) else {"ok": False, "error": "bad reply"}


def get_secret(service: str) -> str:
    """The VALUE of one bound secret. Raises SecretUnavailable if not bound.

    The broker stages the value to a nameless 0600 file for this PID and
    returns its path; we read it and return the contents. The staged file is
    unlinked by the daemon's owner-gated ``release_session`` when the tool
    process ends — the value does not outlive the call.
    """
    service = str(service or "").strip()
    if not service:
        raise SecretUnavailable("service name is required")

    res = _call({"op": "request", "service": service})
    if not res.get("ok"):
        raise SecretUnavailable(
            f"secret {service!r} not available to this tool: "
            f"{res.get('error', 'denied')}. Declare it in "
            f"capabilities.secrets and ensure the calling agent's "
            f"allowance covers it.")

    path = str(res.get("path") or "")
    if not path:
        raise SecretUnavailable(f"broker staged no file for {service!r}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        raise SecretUnavailable(
            f"staged file for {service!r} unreadable: {exc}") from exc
