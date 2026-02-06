"""
IPFS Client for Autonet Nodes

Handles content-addressed storage for model weights, task specs, and solutions.
"""

import logging
import hashlib
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class IPFSClient:
    """Client for interacting with IPFS for content-addressed storage."""

    # Class-level store shared across all instances (important for multi-node tests)
    _mock_store: Dict[str, bytes] = {}

    def __init__(
        self,
        gateway_url: str = "https://ipfs.io/ipfs/",
        api_url: str = "http://127.0.0.1:5001",
    ):
        self.gateway_url = gateway_url
        self.api_url = api_url
        self.client = None

        self._connect()

    def _connect(self) -> None:
        """Connect to IPFS daemon."""
        try:
            import ipfshttpclient
            self.client = ipfshttpclient.connect(self.api_url)
            logger.info("Connected to IPFS daemon")
        except ImportError:
            logger.warning("ipfshttpclient not installed. Using mock mode.")
        except Exception as e:
            logger.warning(f"Could not connect to IPFS daemon: {e}. Using mock mode.")

    def is_connected(self) -> bool:
        """Check if connected to IPFS."""
        return self.client is not None

    def add_json(self, data: Dict[str, Any]) -> Optional[str]:
        """Add JSON data to IPFS and return the CID."""
        content = json.dumps(data, sort_keys=True)
        return self.add_bytes(content.encode())

    def add_bytes(self, data: bytes) -> Optional[str]:
        """Add raw bytes to IPFS and return the CID."""
        if self.client:
            try:
                result = self.client.add_bytes(data)
                logger.info(f"Added to IPFS: {result}")
                return result
            except Exception as e:
                logger.error(f"Failed to add to IPFS: {e}")

        # Mock mode: store data in-memory and return a fake CID
        hash_digest = hashlib.sha256(data).hexdigest()
        mock_cid = f"Qm{hash_digest[:44]}"
        IPFSClient._mock_store[mock_cid] = data
        logger.info(f"Mock IPFS add: {mock_cid}")
        return mock_cid

    def add_file(self, filepath: Union[str, Path]) -> Optional[str]:
        """Add a file to IPFS and return the CID."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        if self.client:
            try:
                result = self.client.add(str(filepath))
                cid = result['Hash'] if isinstance(result, dict) else result[0]['Hash']
                logger.info(f"Added file to IPFS: {cid}")
                return cid
            except Exception as e:
                logger.error(f"Failed to add file to IPFS: {e}")

        # Mock mode
        with open(filepath, 'rb') as f:
            return self.add_bytes(f.read())

    def get_json(self, cid: str) -> Optional[Dict[str, Any]]:
        """Get JSON data from IPFS by CID."""
        data = self.get_bytes(cid)
        if data:
            try:
                return json.loads(data.decode())
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from CID: {cid}")
        return None

    def get_bytes(self, cid: str) -> Optional[bytes]:
        """Get raw bytes from IPFS by CID."""
        if self.client:
            try:
                return self.client.cat(cid)
            except Exception as e:
                logger.error(f"Failed to get from IPFS: {e}")

        # Check mock store
        if cid in IPFSClient._mock_store:
            logger.debug(f"Mock IPFS get: {cid}")
            return IPFSClient._mock_store[cid]

        # Try HTTP gateway as fallback
        try:
            import requests
            url = f"{self.gateway_url}{cid}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.error(f"Failed to get from IPFS gateway: {e}")

        return None

    def pin(self, cid: str) -> bool:
        """Pin a CID to keep it available."""
        if self.client:
            try:
                self.client.pin.add(cid)
                logger.info(f"Pinned: {cid}")
                return True
            except Exception as e:
                logger.error(f"Failed to pin: {e}")
        return False

    def unpin(self, cid: str) -> bool:
        """Unpin a CID."""
        if self.client:
            try:
                self.client.pin.rm(cid)
                logger.info(f"Unpinned: {cid}")
                return True
            except Exception as e:
                logger.error(f"Failed to unpin: {e}")
        return False

    def compute_cid(self, data: bytes) -> str:
        """Compute what the CID would be without uploading."""
        # Simplified: just return a hash-based mock CID
        hash_digest = hashlib.sha256(data).hexdigest()
        return f"Qm{hash_digest[:44]}"
