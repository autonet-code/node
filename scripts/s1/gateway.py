#!/usr/bin/env python3
"""S1 network pieces: an allowlisting HTTP CONNECT proxy and TCP forwards.

The cloud VMs reach the internet only through an agent proxy: HTTPS CONNECT
to allowlisted hostnames, no raw TCP, no direct DNS. S1 recreates that with
this script on a gateway container that straddles the daemons' internal
Docker network and the edge network where Caddy (standing in for
Cloudflare) serves the relay and the RPC.

    python gateway.py proxy --port 3128 --allow relay.autonet.test:443,rpc.autonet.test:443
    python gateway.py forward 7801=h1:7800 7802=h2:7800

Every CONNECT is logged as ``ALLOW host:port`` or ``DENY host:port``, so a
run shows exactly which hosts a daemon needs on the real allowlist.
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def log(msg: str) -> None:
    print(msg, flush=True)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def splice(r1, w1, r2, w2) -> None:
    await asyncio.gather(pipe(r1, w2), pipe(r2, w1))


async def proxy_client(reader, writer, allow: set[str]) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
    except Exception:
        writer.close()
        return
    line = head.split(b"\r\n", 1)[0].decode(errors="replace")
    parts = line.split()
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        # Plain HTTP through the proxy: the cloud proxy is HTTPS-only.
        log(f"DENY non-CONNECT {line[:120]}")
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    target = parts[1].lower()
    if target not in allow:
        log(f"DENY {target}")
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    host, _, port = target.rpartition(":")
    try:
        ur, uw = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 15)
    except Exception as exc:  # noqa: BLE001
        log(f"FAIL {target}: {exc!r}")
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    log(f"ALLOW {target}")
    writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await writer.drain()
    await splice(reader, writer, ur, uw)


async def run_proxy(port: int, allow: set[str]) -> None:
    server = await asyncio.start_server(lambda r, w: proxy_client(r, w, allow), "0.0.0.0", port)
    log(f"proxy on :{port}, allow {sorted(allow)}")
    async with server:
        await server.serve_forever()


async def forward_client(reader, writer, host: str, port: int) -> None:
    try:
        ur, uw = await asyncio.wait_for(asyncio.open_connection(host, port), 15)
    except Exception:
        writer.close()
        return
    await splice(reader, writer, ur, uw)


async def run_forwards(specs: list[str]) -> None:
    servers = []
    for spec in specs:
        listen, _, target = spec.partition("=")
        host, _, port = target.rpartition(":")
        srv = await asyncio.start_server(
            lambda r, w, h=host, p=int(port): forward_client(r, w, h, p), "0.0.0.0", int(listen))
        servers.append(srv)
        log(f"forward :{listen} -> {target}")
    await asyncio.gather(*(s.serve_forever() for s in servers))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proxy")
    p.add_argument("--port", type=int, default=3128)
    p.add_argument("--allow", default="")
    f = sub.add_parser("forward")
    f.add_argument("specs", nargs="+")
    a = ap.parse_args()
    if a.cmd == "proxy":
        allow = {x.strip().lower() for x in a.allow.split(",") if x.strip()}
        asyncio.run(run_proxy(a.port, allow))
    else:
        asyncio.run(run_forwards(a.specs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
