"""
Node-Native Blob Store for Autonet

Content-addressed storage backed by local disk. Zero external infrastructure.
Nodes store blobs locally and serve them to peers over HTTP or P2P.
Content integrity is guaranteed by hash verification.

Interface: add_json, get_json, add_bytes, get_bytes.
Async interface: get_bytes_async (uses P2P when available).
"""

import hashlib
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .p2p import AutonetHost

logger = logging.getLogger(__name__)


class BlobStore:
    """Content-addressed blob storage for Autonet nodes.

    Stores data as files named by their sha256 hash. Retrieval checks
    local disk first, then queries peer nodes over HTTP with hash
    verification on every fetch.
    """

    def __init__(
        self,
        data_dir: str = "~/.autonet/blobs",
        peer_urls: Optional[List[str]] = None,
        p2p_host: Optional["AutonetHost"] = None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.peer_urls = peer_urls or []
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._p2p_host: Optional["AutonetHost"] = None

        if p2p_host is not None:
            self.set_p2p_host(p2p_host)

        logger.info(f"BlobStore initialized at {self.data_dir}")

    # ============ Write Operations ============

    def add_json(self, data: Dict[str, Any]) -> Optional[str]:
        """Store JSON data and return content hash."""
        content = json.dumps(data, sort_keys=True)
        return self.add_bytes(content.encode("utf-8"))

    def add_bytes(self, data: bytes) -> Optional[str]:
        """Store raw bytes and return sha256 hex hash."""
        content_hash = hashlib.sha256(data).hexdigest()
        blob_path = self.data_dir / content_hash
        if not blob_path.exists():
            blob_path.write_bytes(data)
            logger.info(f"Stored blob: {content_hash[:16]}... ({len(data)} bytes)")
        else:
            logger.debug(f"Blob already exists: {content_hash[:16]}...")
        return content_hash

    def add_file(self, filepath: Union[str, Path]) -> Optional[str]:
        """Store a file and return content hash."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
        return self.add_bytes(filepath.read_bytes())

    # ============ Read Operations ============

    def get_json(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve JSON data by content hash."""
        data = self.get_bytes(content_hash)
        if data:
            try:
                return json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.error(f"Failed to decode JSON from blob: {content_hash[:16]}...")
        return None

    def get_bytes(self, content_hash: str) -> Optional[bytes]:
        """Retrieve raw bytes by content hash. Local first, then peers."""
        # 1. Check local store
        blob_path = self.data_dir / content_hash
        if blob_path.exists():
            logger.debug(f"Local hit: {content_hash[:16]}...")
            return blob_path.read_bytes()

        # 2. Try peer nodes
        for peer_url in self.peer_urls:
            try:
                import requests
                url = f"{peer_url}/blob/{content_hash}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    # Verify hash before accepting
                    actual_hash = hashlib.sha256(resp.content).hexdigest()
                    if actual_hash == content_hash:
                        # Cache locally
                        blob_path.write_bytes(resp.content)
                        logger.info(
                            f"Fetched from peer {peer_url}: {content_hash[:16]}..."
                        )
                        return resp.content
                    else:
                        logger.warning(
                            f"Hash mismatch from {peer_url}: "
                            f"expected {content_hash[:16]}, got {actual_hash[:16]}"
                        )
            except Exception as e:
                logger.debug(f"Peer {peer_url} unreachable: {e}")
                continue

        logger.warning(f"Blob not found: {content_hash[:16]}...")
        return None

    # ============ P2P Integration ============

    def set_p2p_host(self, host: "AutonetHost") -> None:
        """
        Wire this BlobStore to a running AutonetHost.

        Registers a blob request handler so peers can fetch blobs from us,
        and enables async P2P fallback in get_bytes_async().
        """
        self._p2p_host = host
        host.set_blob_handler(self._serve_blob_for_p2p)
        logger.info("BlobStore wired to P2P host — serving blobs to peers")

    async def _serve_blob_for_p2p(self, content_hash: str) -> Optional[bytes]:
        """P2P blob handler: serve a locally-stored blob to a requesting peer."""
        return self.get_bytes_local(content_hash)

    async def get_bytes_async(self, content_hash: str) -> Optional[bytes]:
        """
        Retrieve blob: local disk first, HTTP peers second, P2P network third.

        Must be called from an async (trio) context.
        Returns None if the blob cannot be found anywhere.
        """
        # 1. Local cache
        blob_path = self.data_dir / content_hash
        if blob_path.exists():
            logger.debug(f"Local hit: {content_hash[:16]}...")
            return blob_path.read_bytes()

        # 2. HTTP peers (sync, run in thread to not block trio)
        if self.peer_urls:
            import trio
            data = await trio.to_thread.run_sync(
                lambda: self._fetch_from_http_peers(content_hash), cancellable=True
            )
            if data is not None:
                blob_path.write_bytes(data)
                return data

        # 3. P2P peers
        if self._p2p_host and self._p2p_host._running:
            data = await self._p2p_host.fetch_blob_from_any_peer(content_hash)
            if data is not None:
                blob_path.write_bytes(data)
                logger.info(f"Fetched blob {content_hash[:16]}... from P2P network")
                return data

        logger.warning(f"Blob not found anywhere: {content_hash[:16]}...")
        return None

    def _fetch_from_http_peers(self, content_hash: str) -> Optional[bytes]:
        """Synchronous HTTP peer fetch (called from trio thread executor)."""
        for peer_url in self.peer_urls:
            try:
                import requests
                url = f"{peer_url}/blob/{content_hash}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    actual_hash = hashlib.sha256(resp.content).hexdigest()
                    if actual_hash == content_hash:
                        logger.info(f"Fetched from peer {peer_url}: {content_hash[:16]}...")
                        return resp.content
                    logger.warning(
                        f"Hash mismatch from {peer_url}: "
                        f"expected {content_hash[:16]}, got {actual_hash[:16]}"
                    )
            except Exception as e:
                logger.debug(f"Peer {peer_url} unreachable: {e}")
        return None

    async def announce_blobs_to_peers(self) -> int:
        """
        Advertise all locally-stored blobs to connected P2P peers.

        This is a capability hint — peers learn what blobs we can serve them.
        Returns the number of peers notified.
        """
        if not self._p2p_host or not self._p2p_host._running:
            return 0

        local_hashes = self.list_hashes()
        if not local_hashes:
            return 0

        # Advertise via capability update (blob_count in modules_hosted)
        self._p2p_host.update_capability(
            modules_hosted=list(self._p2p_host._capability.modules_hosted) + [
                f"blob:{h}" for h in local_hashes[:50]  # cap at 50 to keep capability small
            ]
        )
        await self._p2p_host.advertise_capability()
        logger.info(f"Announced {len(local_hashes)} blobs to P2P peers")
        return len(self._p2p_host.get_connected_peers())

    # ============ Utility ============

    def compute_hash(self, data: bytes) -> str:
        """Compute sha256 hex digest without storing."""
        return hashlib.sha256(data).hexdigest()

    # Deprecated alias — the original name implied IPFS CIDs but the
    # blob store has only ever produced sha256 hex digests over libp2p.
    compute_cid = compute_hash

    def has(self, content_hash: str) -> bool:
        """Check if a blob exists locally."""
        return (self.data_dir / content_hash).exists()

    def list_hashes(self) -> List[str]:
        """List all locally stored content hashes."""
        return [f.name for f in self.data_dir.iterdir() if f.is_file()]

    def delete(self, content_hash: str) -> bool:
        """Delete a blob from local store."""
        blob_path = self.data_dir / content_hash
        if blob_path.exists():
            blob_path.unlink()
            return True
        return False

    # ============ HTTP Server for Peer Retrieval ============

    def start_server(self, port: int = 9100) -> int:
        """Start HTTP server so peers can fetch our blobs.

        Returns the actual port the server is listening on.
        """
        store = self

        class BlobHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Expected path: /blob/<content_hash>
                if not self.path.startswith("/blob/"):
                    self.send_response(404)
                    self.end_headers()
                    return

                content_hash = self.path[len("/blob/"):]
                data = store.get_bytes_local(content_hash)
                if data is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress default logging
                pass

        self._server = HTTPServer(("0.0.0.0", port), BlobHandler)
        actual_port = self._server.server_address[1]
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()
        logger.info(f"Blob server listening on port {actual_port}")
        return actual_port

    def stop_server(self):
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._server_thread = None
            logger.info("Blob server stopped")

    def get_bytes_local(self, content_hash: str) -> Optional[bytes]:
        """Get bytes from local store only (no peer fetch). Used by HTTP server."""
        blob_path = self.data_dir / content_hash
        if blob_path.exists():
            return blob_path.read_bytes()
        return None
