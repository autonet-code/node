"""HTTP CONNECT proxies for the libp2p WebSocket transport.

py-libp2p's WebSocket client honours ``HTTP_PROXY`` / ``HTTPS_PROXY`` but
only speaks SOCKS5 to them. Machines whose only egress is an HTTP proxy
(cloud sandboxes, corporate networks) need HTTP CONNECT: open a tunnel
through the proxy, add TLS for ``wss``, then run the WebSocket handshake
over that stream. ``install()`` patches the transport once; SOCKS proxy URLs
keep the library's own path. See docs/p2p_networking.md.
"""

from __future__ import annotations

import base64
import logging
import ssl
from typing import Any, Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

_installed = False


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


def install() -> None:
    """Teach the WebSocket transport to dial through HTTP(S) CONNECT proxies."""
    global _installed
    if _installed:
        return
    try:
        from libp2p.transport.websocket import transport as wst
    except Exception:  # noqa: BLE001 — no websocket transport, nothing to do
        return
    original = wst.WebsocketTransport._create_proxy_connection
    _install_remote_address()

    async def _create_proxy_connection(self: Any, proto_info: Any, proxy_url: str,
                                       ssl_context: Optional[ssl.SSLContext]) -> Any:
        if urlparse(proxy_url).scheme not in ("http", "https"):
            return await original(self, proto_info, proxy_url, ssl_context)
        import trio
        from trio_websocket import wrap_client_stream
        from libp2p.utils.multiaddr_utils import extract_host_from_multiaddr
        host = extract_host_from_multiaddr(proto_info.rest_multiaddr) or "localhost"
        if isinstance(host, tuple):
            host = host[0]
        port = int(proto_info.rest_multiaddr.value_for_protocol("tcp") or "80")
        with trio.fail_after(self._config.handshake_timeout):
            stream = await open_connect_tunnel(proxy_url, host, port)
            if proto_info.is_wss:
                stream = trio.SSLStream(stream, ssl_context or ssl.create_default_context(),
                                        server_hostname=host)
                await stream.do_handshake()
            if self._background_nursery is None:
                raise wst.OpenConnectionError(
                    "No background nursery available for the WebSocket transport")
            ws = await wrap_client_stream(
                self._background_nursery, stream, host, "/",
                message_queue_size=1024,
                max_message_size=self._config.max_message_size)
        logger.info("WebSocket dial to %s:%s through HTTP proxy %s",
                    host, port, urlparse(proxy_url).hostname)
        return wst.P2PWebSocketConnection(
            ws, None, is_secure=proto_info.is_wss,
            max_buffered_amount=self._config.max_buffered_amount)

    wst.WebsocketTransport._create_proxy_connection = _create_proxy_connection
    _installed = True


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
