"""The ws transport (docs/p2p_networking.md): a relay that can sit behind an
HTTPS proxy, and daemons that can only dial out through an HTTP proxy.

  - gossip and blobs cross a /ws relay;
  - a daemon dials the relay through an HTTP CONNECT proxy;
  - the Cloudflare shape: wss by HOSTNAME, through the proxy, to a
    TLS-terminating front that forwards plain ws to the relay.

Run: python -m pytest tests/test_p2p_ws.py -v
"""

import datetime
import functools
import hashlib
import ssl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trio

from nodes.common.blob_resolver import host_blob_server
from nodes.common.p2p import AutonetHost

TOPIC = "/autonet/world-events/ws-test"


def trio_test(fn):
    """See tests/test_p2p.py: pytest-asyncio claims bare async tests."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        trio.run(functools.partial(fn, *args, **kwargs))
    return run


def _ws(name):
    return AutonetHost(node_id=name, listen_host="127.0.0.1", transport="ws")


async def _pipe(src, dst):
    try:
        while True:
            data = await src.receive_some(65536)
            if not data:
                break
            await dst.send_all(data)
    except Exception:
        pass


async def _connect_proxy(stream, seen):
    """Minimal HTTP CONNECT proxy (the cloud sandboxes' egress shape)."""
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await stream.receive_some(4096)
        if not chunk:
            return
        head += chunk
    target = head.split(b"\r\n", 1)[0].decode().split(" ")[1]
    seen.append(target)
    host, port = target.rsplit(":", 1)
    upstream = await trio.open_tcp_stream(host, int(port))
    await stream.send_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    async with trio.open_nursery() as n:
        n.start_soon(_pipe, stream, upstream)
        n.start_soon(_pipe, upstream, stream)


def _self_signed(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                           critical=False)
            .sign(key, hashes.SHA256()))
    cert_p, key_p = tmp_path / "c.pem", tmp_path / "k.pem"
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                        serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(cert_p, key_p)
    return ctx


async def _relay_traffic(a, c, relay):
    """Gossip a -> c and a blob c -> a, both only through the relay."""
    got = []
    blob = b"blob-over-ws" * 500
    cid = hashlib.sha256(blob).hexdigest()
    host_blob_server(relay)
    await relay.subscribe_topic(TOPIC, lambda _t, _d: None)
    await c.subscribe_topic(TOPIC, lambda _t, d: got.append(d))
    await a.subscribe_topic(TOPIC, lambda _t, _d: None)
    with trio.fail_after(20):
        while not got:
            await a.publish_topic(TOPIC, b"event")
            await trio.sleep(0.5)
    host_blob_server(c).put(blob)
    with trio.fail_after(15):
        assert await a.fetch_blob_from_peers([relay.peer_id], cid) == blob
    return got[0]


@trio_test
async def test_gossip_and_blobs_cross_a_ws_relay():
    r, a, c = _ws("relay"), _ws("a"), _ws("c")
    async with r.run(), a.run(), c.run():
        assert "/ws/" in r.addrs[0]
        assert await a.connect_to_peer(r.addrs[0])
        assert await c.connect_to_peer(r.addrs[0])
        assert await _relay_traffic(a, c, r) == b"event"


@trio_test
async def test_dial_through_http_connect_proxy(monkeypatch=None):
    import os
    seen = []
    r, a, c = _ws("relay"), _ws("a"), _ws("c")
    async with trio.open_nursery() as nursery:
        listeners = await nursery.start(
            functools.partial(trio.serve_tcp, lambda s: _connect_proxy(s, seen), 0,
                              host="127.0.0.1"))
        proxy_port = listeners[0].socket.getsockname()[1]
        async with r.run(), a.run(), c.run():
            os.environ["HTTP_PROXY"] = f"http://127.0.0.1:{proxy_port}"
            try:
                assert await a.connect_to_peer(r.addrs[0])
            finally:
                os.environ.pop("HTTP_PROXY", None)
            assert seen, "the dial did not go through the proxy"
            assert await c.connect_to_peer(r.addrs[0])
            assert await _relay_traffic(a, c, r) == b"event"
        nursery.cancel_scope.cancel()


@trio_test
async def test_wss_by_hostname_through_proxy_to_a_tls_front(tmp_path=None):
    """Cloudflare shape: the relay serves plain ws; a TLS front terminates
    wss on another port; the daemon dials /dns4/localhost/.../wss through an
    HTTP proxy, which resolves the name (the dialer never does DNS)."""
    import os
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    server_ctx = _self_signed(tmp)
    seen = []
    r, a = _ws("relay"), _ws("a")
    async with trio.open_nursery() as nursery:
        proxy = await nursery.start(
            functools.partial(trio.serve_tcp, lambda s: _connect_proxy(s, seen), 0,
                              host="127.0.0.1"))
        proxy_port = proxy[0].socket.getsockname()[1]
        async with r.run(), a.run():
            relay_port = int(r.addrs[0].split("/tcp/")[1].split("/")[0])

            async def _front(stream):
                tls = trio.SSLStream(stream, server_ctx, server_side=True)
                await tls.do_handshake()
                upstream = await trio.open_tcp_stream("127.0.0.1", relay_port)
                async with trio.open_nursery() as n:
                    n.start_soon(_pipe, tls, upstream)
                    n.start_soon(_pipe, upstream, tls)
            front = await nursery.start(functools.partial(
                trio.serve_tcp, _front, 0, host="127.0.0.1"))
            front_port = front[0].socket.getsockname()[1]
            addr = f"/dns4/localhost/tcp/{front_port}/wss/p2p/{r.peer_id}"
            os.environ["HTTPS_PROXY"] = f"http://127.0.0.1:{proxy_port}"
            try:
                assert await a.connect_to_peer(addr)
            finally:
                os.environ.pop("HTTPS_PROXY", None)
            assert seen == [f"localhost:{front_port}"]
            assert r.peer_id in a.get_connected_peers()
        nursery.cancel_scope.cancel()
