"""Proxied and guarded dialing for the libp2p WebSocket transport (0.8.1).

Machines whose only egress is a proxy (cloud sandboxes, corporate networks)
dial WebSocket peers through it: HTTP CONNECT (``http://`` / ``https://``
proxy URLs) or SOCKS5 (``socks5://`` / ``socks5h://``), named by
``HTTPS_PROXY`` / ``HTTP_PROXY`` / ``ALL_PROXY`` (``NO_PROXY`` honoured).
The tunnel carries TLS for ``wss`` and then the WebSocket handshake.
py-libp2p's own SOCKS path waits on a task group that never ends, so both
proxy kinds go through the code here.

``install()`` patches the transport once:

- every dial (direct or proxied) runs its WebSocket connection in its own
  guarded task group (``_guarded``): py-libp2p starts each connection's
  reader in one shared background nursery, so a rejected handshake (a
  proxy answering 502 for a dead relay) or a reader error used to cancel
  every connection on the node;
- the proxy choice is made here rather than in py-libp2p's
  ``_create_connection`` (which prints to stdout on every proxied dial);
- the remote address of a ws connection is read from trio-websocket (the
  library returns a placeholder), so the reach check sees a ws peer's IP.

See docs/p2p_networking.md.
"""

from __future__ import annotations

import base64
import ipaddress
import logging
import ssl
import struct
from typing import Any, Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

_installed = False
_env_proxy: Any = None   # py-libp2p's get_proxy_from_environment (NO_PROXY aware)


async def open_connect_tunnel(proxy_url: str, host: str, port: int) -> Any:
    """A trio stream to ``host:port`` through the HTTP proxy at ``proxy_url``."""
    import trio
    p = urlparse(proxy_url)
    stream: Any = await trio.open_tcp_stream(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    if p.scheme == "https":
        ctx = ssl.create_default_context()
        stream = trio.SSLStream(stream, ctx, server_hostname=p.hostname)
        await stream.do_handshake()
    target = f"{host}:{port}"
    lines = [f"CONNECT {target} HTTP/1.1", f"Host: {target}"]
    if p.username:
        cred = f"{unquote(p.username)}:{unquote(p.password or '')}"
        lines.append("Proxy-Authorization: Basic "
                     + base64.b64encode(cred.encode()).decode())
    await stream.send_all(("\r\n".join(lines) + "\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await stream.receive_some(4096)
        if not chunk:
            raise OSError(f"proxy {p.hostname} closed during CONNECT")
        head += chunk
        if len(head) > 16384:
            raise OSError("proxy CONNECT response too long")
    status = head.split(b"\r\n", 1)[0].decode(errors="replace")
    if " 200" not in status:
        raise OSError(f"proxy refused CONNECT {target}: {status}")
    return stream


async def _read_exactly(stream: Any, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = await stream.receive_some(n - len(buf))
        if not chunk:
            raise OSError("SOCKS proxy closed the connection")
        buf += chunk
    return buf


async def open_socks5_tunnel(proxy_url: str, host: str, port: int) -> Any:
    """A trio stream to ``host:port`` through the SOCKS5 proxy at
    ``proxy_url`` (RFC 1928; no auth or username/password, RFC 1929).
    Names are sent to the proxy unresolved, so the dialer needs no DNS."""
    import trio
    p = urlparse(proxy_url)
    stream = await trio.open_tcp_stream(p.hostname, p.port or 1080)
    try:
        methods = b"\x00\x02" if p.username else b"\x00"
        await stream.send_all(b"\x05" + bytes([len(methods)]) + methods)
        ver, method = await _read_exactly(stream, 2)
        if ver != 5 or method == 0xFF:
            raise OSError("SOCKS proxy accepts none of our auth methods")
        if method == 2:
            user = unquote(p.username or "").encode()
            pw = unquote(p.password or "").encode()
            await stream.send_all(b"\x01" + bytes([len(user)]) + user + bytes([len(pw)]) + pw)
            _, status = await _read_exactly(stream, 2)
            if status != 0:
                raise OSError("SOCKS proxy refused our credentials")
        try:
            ip = ipaddress.ip_address(host)
            addr = (b"\x01" if ip.version == 4 else b"\x04") + ip.packed
        except ValueError:
            name = host.encode("idna")
            addr = b"\x03" + bytes([len(name)]) + name
        await stream.send_all(b"\x05\x01\x00" + addr + struct.pack(">H", port))
        ver, rep, _, atyp = await _read_exactly(stream, 4)
        if rep != 0:
            raise OSError(f"SOCKS proxy refused CONNECT {host}:{port} (code {rep})")
        skip = {1: 4, 4: 16}.get(atyp)
        if skip is None:
            skip = (await _read_exactly(stream, 1))[0]
        await _read_exactly(stream, skip + 2)
        return stream
    except BaseException:
        await stream.aclose()
        raise


async def open_tunnel(proxy_url: str, host: str, port: int) -> Any:
    scheme = urlparse(proxy_url).scheme.lower()
    if scheme in ("http", "https"):
        return await open_connect_tunnel(proxy_url, host, port)
    if scheme in ("socks5", "socks5h", "socks"):
        return await open_socks5_tunnel(proxy_url, host, port)
    raise OSError(f"unsupported proxy scheme {scheme!r} in {proxy_url}")


def proxy_for(url: str) -> Optional[str]:
    """The proxy a dial to ``url`` goes through (environment, NO_PROXY)."""
    if _env_proxy is None:
        return None
    import os
    try:
        found = _env_proxy(url)
        if found:
            return found
        # A machine whose only way out is a proxy tunnels every dial, so a
        # ws:// dial with only HTTPS_PROXY (or ALL_PROXY) set uses it too.
        from libp2p.transport.websocket import proxy_env
        parsed = urlparse(url)
        if proxy_env._should_bypass_proxy(parsed.hostname, parsed.port):
            return None
        for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            if os.environ.get(name):
                return os.environ[name]
    except Exception:  # noqa: BLE001
        return None
    return None


def install() -> None:
    """Patch the WebSocket transport (see the module docstring)."""
    global _installed, _env_proxy
    if _installed:
        return
    try:
        from libp2p.transport.websocket import transport as wst
        from libp2p.transport.websocket import proxy_env
    except Exception:  # noqa: BLE001 - no websocket transport, nothing to do
        return
    _install_remote_address()
    _env_proxy = proxy_env.get_proxy_from_environment
    # py-libp2p's _create_connection asks this and prints the answer; we
    # choose the proxy in _create_direct_connection instead.
    proxy_env.get_proxy_from_environment = lambda url: None

    async def _open(self: Any, proto_info: Any, ssl_context: Optional[ssl.SSLContext],
                    proxy_url: Optional[str]) -> Any:
        import trio
        from trio_websocket import wrap_client_stream
        from libp2p.utils.multiaddr_utils import extract_host_from_multiaddr
        host = extract_host_from_multiaddr(proto_info.rest_multiaddr) or "localhost"
        if isinstance(host, tuple):
            host = host[0]
        port = int(proto_info.rest_multiaddr.value_for_protocol("tcp") or "80")
        if self._background_nursery is None:
            raise wst.OpenConnectionError(
                "No background nursery available for the WebSocket transport")
        with trio.fail_after(self._config.handshake_timeout):
            if proxy_url:
                stream: Any = await open_tunnel(proxy_url, host, port)
            else:
                stream = await trio.open_tcp_stream(host, port)
            try:
                if proto_info.is_wss:
                    stream = trio.SSLStream(stream, ssl_context or ssl.create_default_context(),
                                            server_hostname=host, https_compatible=True)
                    await stream.do_handshake()
                host_header = host if port in (80, 443) else f"{host}:{port}"
                ws = await self._background_nursery.start(_guarded(
                    lambda n: wrap_client_stream(
                        n, stream, host_header, "/", message_queue_size=1024,
                        max_message_size=self._config.max_message_size)))
            except BaseException:
                with trio.CancelScope(shield=True):
                    await trio.aclose_forcefully(stream)
                raise
        if proxy_url:
            logger.info("WebSocket dial to %s:%s through proxy %s",
                        host, port, urlparse(proxy_url).hostname)
        return wst.P2PWebSocketConnection(
            ws, None, is_secure=proto_info.is_wss,
            max_buffered_amount=self._config.max_buffered_amount)

    async def _create_proxy_connection(self: Any, proto_info: Any, proxy_url: str,
                                       ssl_context: Optional[ssl.SSLContext]) -> Any:
        return await _open(self, proto_info, ssl_context, proxy_url)

    async def _create_direct_connection(self: Any, proto_info: Any,
                                        ssl_context: Optional[ssl.SSLContext]) -> Any:
        from libp2p.utils.multiaddr_utils import extract_host_from_multiaddr, format_host_for_url
        host = extract_host_from_multiaddr(proto_info.rest_multiaddr) or "localhost"
        if isinstance(host, tuple):
            host = host[0]
        port = int(proto_info.rest_multiaddr.value_for_protocol("tcp") or "80")
        url = f"{'wss' if proto_info.is_wss else 'ws'}://{format_host_for_url(host)}:{port}/"
        return await _open(self, proto_info, ssl_context, proxy_for(url))

    wst.WebsocketTransport._create_proxy_connection = _create_proxy_connection
    wst.WebsocketTransport._create_direct_connection = _create_direct_connection
    _installed = True


def _guarded(opener: Any) -> Any:
    """A WebSocket connection in its own task group inside the transport's
    shared one. py-libp2p runs every connection's reader directly in one
    background nursery, so a rejected handshake (a proxy answering 502 for
    a dead relay) or a reader error cancelled every connection on the node.
    Here a failure before the connection is up raises in the dial alone, and
    a failure after it stays with that connection."""
    import trio

    async def run(task_status: Any = trio.TASK_STATUS_IGNORED) -> None:
        up = False
        try:
            async with trio.open_nursery() as n:
                ws = await opener(n)
                up = True
                task_status.started(ws)
        except* Exception as group:
            if not up:
                raise
            logger.debug("WebSocket connection ended: %r", group.exceptions[0])
    return run


def _install_remote_address() -> None:
    """py-libp2p's WebSocket connection returns no remote address (a
    placeholder), so the reach check could never see a ws peer's IP. Read
    it from trio-websocket's endpoint instead."""
    try:
        from libp2p.transport.websocket import connection as wsc
    except Exception:  # noqa: BLE001
        return

    def get_remote_address(self: Any) -> Optional[tuple]:
        remote = getattr(getattr(self, "_ws_connection", None), "remote", None)
        address = getattr(remote, "address", None)
        if address is None:
            return None
        return (str(address), int(getattr(remote, "port", 0) or 0))

    wsc.P2PWebSocketConnection.get_remote_address = get_remote_address
