"""Deny-by-default capability guard for ADOPTED pinned tools.

docs/tool_substrate.md — Adoption rail, containment layer. Launched as

    python tool_guard.py <tool_script.py>

with the policy JSON in the ATN_TOOL_POLICY environment variable
(popped before the tool code runs). The parent (ToolStore) has already
scrubbed the environment and pointed cwd at a per-tool sandbox
directory; this runner adds the in-process floor: a sys.addaudithook
that hard-fails on any capability the manifest did not declare.

Policy shape (all deny by default):

    {"net": bool, "fs": bool, "spawn": bool, "hosts": [str, ...]}

  - net   — socket use ("socket.*" audit events)
  - fs    — file access outside the sandbox cwd + the Python runtime
            ("open" audit events; fd-based and in-tree access passes)
  - spawn — subprocess / exec / fork ("subprocess.Popen", "os.system",
            "os.posix_spawn", "os.spawn", "os.exec", "os.fork",
            "os.forkpty")
  - hosts — WHERE net may go. Empty/absent with net=True means
            "anywhere" (the pre-existing behavior). Non-empty narrows
            net to those destinations: the union of authorized_hosts
            for the secrets this tool was bound (see
            docs/tool_secret_binding.md). A tool holding a credential
            AND unrestricted egress is the shape this narrows.

            HONEST LIMITS, read them before trusting this: matching is
            on the destination as WRITTEN by the caller. A hostname is
            checked at socket.getaddrinfo; a literal IP is checked at
            socket.connect against the same list, so connecting to a
            raw IP passes ONLY if that IP string is itself listed.
            What this does NOT stop: a resolver-level indirection (the
            tool resolves an allowed name itself, then connects to
            whatever address it got), a proxy/CDN that fronts other
            origins, DNS-record exfiltration, or native code bypassing
            the audit hook entirely. It raises the cost of a stealth
            call in ordinary Python. It is NOT an egress firewall.

Two GUEST-only fields (origin="integration" tools, atn/guest_sandbox.py):

  - deny_loopback — refuse any connect/sendto to a loopback, unspecified
            or link-local address, "localhost", or an AF_UNIX path, even
            with net=True. Keeps a guest tool off the daemon's own
            listeners (owner socket :7700) at the Python layer. The
            OS-level check is the owner listener's peer-credential test
            (ws_auth.local_peer_denied); this is the tripwire before it.
  - deny_paths    — path prefixes refused for open/listdir/scandir even
            with fs=True (the daemon data dir, keystore, /proc). Matters
            on the same-uid fallback; under uid isolation the OS already
            refuses them.

An undeclared capability raises PermissionError: the tool process dies
with a traceback naming the capability, the ToolStore surfaces it as an
error frame, and the mismatch between manifest and behavior is exactly
the reproducible evidence a CON claim wants.

HONESTY, NOT HERMETICS: an audit hook is bypassable by sufficiently
determined native code (ctypes, extension modules). It makes
capability declarations honest for straightforward Python and turns
evasion attempts into deliberate, evidenced acts. The OS-level
isolated runner (vault track) is the wall; this is the tripwire in
front of it. Defense stays layered: consent gate, provenance
friction, and the CON evidence rail sit around this.
"""
from __future__ import annotations

import ipaddress
import json
import os
import runpy
import sys


def _real(path: str) -> str:
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return path


def main() -> None:
    policy = json.loads(os.environ.pop("ATN_TOOL_POLICY", "{}"))
    script = sys.argv[1]

    allow_net = bool(policy.get("net"))
    allow_fs = bool(policy.get("fs"))
    allow_spawn = bool(policy.get("spawn"))
    # Destination allowlist. Empty => net is unrestricted (prior behavior);
    # non-empty => every destination must match. Lowercased once here so the
    # hot path is a plain set/suffix test.
    allow_hosts = {
        str(h).strip().lower()
        for h in (policy.get("hosts") or [])
        if str(h).strip()
    }

    deny_loopback = bool(policy.get("deny_loopback"))
    deny_paths = tuple(
        _real(str(p)) for p in (policy.get("deny_paths") or []) if str(p).strip()
    )

    allowed_prefixes = tuple({
        _real(os.getcwd()),
        _real(sys.prefix),
        _real(sys.base_prefix),
        _real(os.path.dirname(_real(script))),
    })

    # os.fork / os.forkpty too: a forked child can setsid() out of the
    # process group the parent kills, so "no spawn" must mean no fork.
    spawn_events = ("subprocess.Popen", "os.system", "os.posix_spawn",
                    "os.spawn", "os.exec", "os.startfile", "os.fork",
                    "os.forkpty")

    def _host_ok(host: str) -> bool:
        """True iff ``host`` is covered by the allowlist.

        Exact match, or a subdomain of a listed name (``api.x.com`` matches a
        listed ``x.com``). Never the reverse: listing ``api.x.com`` does not
        permit ``x.com`` or ``evil-x.com``.
        """
        h = host.strip().lower().rstrip(".")
        if not h:
            return False
        if h in allow_hosts:
            return True
        return any(h.endswith("." + allowed) for allowed in allow_hosts)

    def _check_dest(event: str, args: tuple) -> None:
        """Enforce the destination allowlist on the two events that carry one.

        socket.getaddrinfo -> args[0] is the host being resolved.
        socket.connect     -> args[1] is the address tuple; args[1][0] is the
                              host/IP as written by the caller.
        Any other socket.* event carries no destination and is left to the
        blanket net check.
        """
        if not allow_hosts:
            return
        dest = None
        if event == "socket.getaddrinfo" and args:
            dest = args[0]
        elif event == "socket.connect" and len(args) > 1:
            addr = args[1]
            if isinstance(addr, (tuple, list)) and addr:
                dest = addr[0]
        if dest is None:
            return
        if isinstance(dest, bytes):
            dest = dest.decode("utf-8", errors="replace")
        if not isinstance(dest, str):
            return  # AF_UNIX path / unknown shape: not a network destination
        if not _host_ok(dest):
            raise PermissionError(
                f"destination not authorized: {dest} "
                f"(allowed: {', '.join(sorted(allow_hosts))})")

    def _is_local_dest(dest) -> bool:
        if isinstance(dest, bytes):
            dest = dest.decode("utf-8", errors="replace")
        if not isinstance(dest, str):
            return False
        h = dest.strip().lower().rstrip(".").strip("[]")
        if h in ("", "localhost", "localhost.localdomain", "ip6-localhost",
                 "ip6-loopback") or h.endswith(".localhost"):
            return True
        try:
            ip = ipaddress.ip_address(h.split("%", 1)[0])
        except ValueError:
            return False
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
        return ip.is_loopback or ip.is_unspecified or ip.is_link_local

    def _check_local(event: str, args: tuple) -> None:
        """deny_loopback: no connect/sendto toward this host's own sockets."""
        if event == "socket.getaddrinfo" and args:
            if _is_local_dest(args[0]):
                raise PermissionError(
                    f"guest tool may not reach local addresses ({args[0]})")
            return
        if event not in ("socket.connect", "socket.sendto", "socket.sendmsg"):
            return
        addr = args[1] if len(args) > 1 else None
        if isinstance(addr, (str, bytes)):
            # AF_UNIX path (or abstract socket): a local IPC endpoint.
            raise PermissionError("guest tool may not connect to unix sockets")
        if isinstance(addr, (tuple, list)) and addr:
            if _is_local_dest(addr[0]):
                raise PermissionError(
                    f"guest tool may not reach local addresses ({addr[0]})")

    def _denied_path(path) -> bool:
        if not deny_paths or path is None or isinstance(path, int):
            return False
        p = _real(os.fsdecode(path) if isinstance(path, bytes) else str(path))
        return any(p == d or p.startswith(d.rstrip("/\\") + os.sep)
                   for d in deny_paths)

    # Evidence capture (docs/tool_substrate.md, Decision 2026-10-12). When the
    # daemon runs a CORE-REPLACING tool it passes trace_path; every observed
    # file open, network destination and spawn argv is appended as one JSON
    # line. The fd is opened BEFORE the hook is installed (so it is not itself
    # traced) and written with os.write (not an audited event). Interpreter
    # internals (stdlib / site-packages / the tool's own code dir) are skipped.
    # What a spawned child does is NOT visible here: only the spawn's argv.
    trace_fd = -1
    trace_path = str(policy.get("trace_path") or "")
    if trace_path:
        try:
            trace_fd = os.open(trace_path,
                               os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        except OSError:
            trace_fd = -1
    trace_state = {"n": 0}
    trace_cap = 5000
    internal_prefixes = tuple({
        _real(sys.prefix), _real(sys.base_prefix),
        _real(os.path.dirname(_real(script))),
    })
    import time as _time

    def _emit(rec: dict) -> None:
        if trace_state["n"] >= trace_cap:
            return
        trace_state["n"] += 1
        rec["ts"] = round(_time.time(), 3)
        try:
            os.write(trace_fd, (json.dumps(rec, default=str) + "\n")
                     .encode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001 — tracing never breaks the tool
            pass

    def _observe(event: str, args: tuple) -> None:
        if event == "open":
            path = args[0] if args else None
            if path is None or isinstance(path, int):
                return
            p = _real(os.fsdecode(path) if isinstance(path, bytes) else str(path))
            if p.startswith(internal_prefixes):
                return
            mode = args[1] if len(args) > 1 else ""
            _emit({"kind": "file", "path": p, "mode": str(mode)})
        elif event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
            addr = args[1] if len(args) > 1 else None
            _emit({"kind": "net", "event": event, "dest": addr})
        elif event == "socket.getaddrinfo" and args:
            _emit({"kind": "net", "event": event, "dest": [args[0],
                   args[1] if len(args) > 1 else None]})
        elif event == "subprocess.Popen":
            _emit({"kind": "spawn", "executable": args[0] if args else None,
                   "argv": args[1] if len(args) > 1 else None,
                   "cwd": args[2] if len(args) > 2 else None})
        elif event in ("os.system", "os.posix_spawn", "os.spawn", "os.exec",
                       "os.startfile"):
            _emit({"kind": "spawn", "event": event, "argv": list(args)[:3]})

    def _refuse(reason: str) -> None:
        raise PermissionError(reason)

    def _check(event: str, args: tuple) -> None:
        if not allow_net and event.startswith("socket."):
            _refuse(f"undeclared capability: net ({event})")
        if allow_net and event.startswith("socket."):
            if deny_loopback:
                _check_local(event, args)
            _check_dest(event, args)
        if deny_paths and event in fs_path_events:
            if _denied_path(args[0] if args else None):
                _refuse(f"tool may not access {args[0]}")
        if not allow_spawn and event in spawn_events:
            _refuse(f"undeclared capability: spawn ({event})")
        if not allow_fs and event in fs_path_events:
            path = args[0] if args else None
            if path is None or isinstance(path, int):
                return  # fd re-open: the fd was already policy-checked
            p = _real(os.fsdecode(path) if isinstance(path, bytes)
                      else str(path))
            if not p.startswith(allowed_prefixes):
                _refuse(f"undeclared capability: fs ({p})")

    # Path-taking events checked for fs and deny_paths: reads, listings and
    # the mutations (delete, rename, mkdir) an open-only check missed.
    fs_path_events = ("open", "os.listdir", "os.scandir", "os.remove",
                      "os.rename", "os.rmdir", "os.mkdir", "shutil.rmtree")

    def hook(event: str, args: tuple) -> None:
        if trace_fd >= 0:
            _observe(event, args)
        try:
            _check(event, args)
        except PermissionError as exc:
            # Recorded BEFORE it propagates, so a tool that swallows the
            # error still leaves the refusal in the trace (the daemon reads
            # it back and raises the alarm, Decision 2026-10-27).
            if trace_fd >= 0:
                _emit({"kind": "denied", "reason": str(exc)[:300]})
            raise

    sys.addaudithook(hook)
    sys.argv = [script]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
