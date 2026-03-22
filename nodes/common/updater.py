"""
Auto-Update Mechanism for Autonet Nodes.

Checks for new versions of the node software and applies updates.
Supports multiple sources: git, HTTP, or blob store.

Story 6.5: Auto-update mechanism.
"""

import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .config import AutonetConfig, UpdateConfig, load_config

logger = logging.getLogger(__name__)


class UpdateSource(Enum):
    GIT = "git"
    HTTP = "http"
    BLOB_STORE = "blob_store"


@dataclass
class UpdateInfo:
    """Describes an available update."""
    current_version: str
    available_version: str
    source: str
    changelog: str = ""
    download_url: str = ""
    sha256: str = ""

    @property
    def has_update(self) -> bool:
        return self.available_version != self.current_version


class AutonetUpdater:
    """
    Checks for and applies node software updates.

    Lifecycle:
        updater = AutonetUpdater(config, current_version="0.1.0")
        info = updater.check_update()
        if info and info.has_update:
            updater.apply_update(info)
    """

    def __init__(
        self,
        config: Optional[AutonetConfig] = None,
        current_version: str = "0.1.0",
        repo_path: Optional[str] = None,
    ):
        self.config = config or load_config()
        self.update_config: UpdateConfig = self.config.update
        self.current_version = current_version
        self.repo_path = repo_path or str(Path.cwd())
        self._last_check: float = 0.0
        self._last_info: Optional[UpdateInfo] = None

    def check_update(self) -> Optional[UpdateInfo]:
        """
        Check for available updates from the configured source.

        Returns UpdateInfo if an update is available, None on error.
        """
        source = self.update_config.update_source

        try:
            if source == UpdateSource.GIT.value:
                return self._check_git()
            elif source == UpdateSource.HTTP.value:
                return self._check_http()
            elif source == UpdateSource.BLOB_STORE.value:
                return self._check_blob_store()
            else:
                logger.warning(f"Unknown update source: {source}")
                return None
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return None

    def should_check(self) -> bool:
        """Whether enough time has elapsed since the last check."""
        return time.time() - self._last_check >= self.update_config.check_interval

    def apply_update(self, info: UpdateInfo) -> bool:
        """
        Apply an update. Returns True on success.

        For git: performs git pull.
        For HTTP: downloads and verifies the package.
        For blob_store: fetches from blob store.
        """
        if not info.has_update:
            logger.info("No update to apply")
            return True

        source = self.update_config.update_source

        try:
            if source == UpdateSource.GIT.value:
                return self._apply_git()
            elif source == UpdateSource.HTTP.value:
                return self._apply_http(info)
            elif source == UpdateSource.BLOB_STORE.value:
                return self._apply_blob_store(info)
            else:
                logger.warning(f"Unknown update source: {source}")
                return False
        except Exception as e:
            logger.error(f"Update apply failed: {e}")
            return False

    # -- Git source --

    def _check_git(self) -> Optional[UpdateInfo]:
        """Check for updates via git fetch."""
        self._last_check = time.time()

        try:
            # Fetch latest
            subprocess.run(
                ["git", "fetch", "--quiet"],
                cwd=self.repo_path,
                capture_output=True,
                timeout=30,
            )

            # Compare local HEAD with remote
            local = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()

            remote = subprocess.run(
                ["git", "rev-parse", "origin/master"],
                cwd=self.repo_path,
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()

            # Get changelog
            changelog = ""
            if local != remote:
                result = subprocess.run(
                    ["git", "log", "--oneline", f"{local}..{remote}"],
                    cwd=self.repo_path,
                    capture_output=True, text=True, timeout=10,
                )
                changelog = result.stdout.strip()

            info = UpdateInfo(
                current_version=local[:8],
                available_version=remote[:8],
                source="git",
                changelog=changelog,
            )

            self._last_info = info

            if info.has_update:
                logger.info(
                    f"Update available: {info.current_version} -> {info.available_version}"
                )
            else:
                logger.debug("No git updates available")

            return info

        except FileNotFoundError:
            logger.warning("git not found on PATH")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("git fetch timed out")
            return None

    def _apply_git(self) -> bool:
        """Apply update via git pull."""
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=self.repo_path,
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                logger.info(f"Git update applied: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"Git pull failed: {result.stderr.strip()}")
                return False
        except Exception as e:
            logger.error(f"Git pull error: {e}")
            return False

    # -- HTTP source --

    def _check_http(self) -> Optional[UpdateInfo]:
        """Check for updates via HTTP version endpoint."""
        self._last_check = time.time()
        url = self.update_config.update_url
        if not url:
            logger.warning("No update_url configured for HTTP source")
            return None

        try:
            import requests
            resp = requests.get(f"{url}/version.json", timeout=10)
            resp.raise_for_status()
            data = resp.json()

            info = UpdateInfo(
                current_version=self.current_version,
                available_version=data.get("version", self.current_version),
                source="http",
                changelog=data.get("changelog", ""),
                download_url=data.get("download_url", ""),
                sha256=data.get("sha256", ""),
            )
            self._last_info = info
            return info

        except Exception as e:
            logger.error(f"HTTP update check failed: {e}")
            return None

    def _apply_http(self, info: UpdateInfo) -> bool:
        """Download and verify an HTTP update package."""
        if not info.download_url:
            logger.error("No download URL in update info")
            return False

        try:
            import requests
            resp = requests.get(info.download_url, timeout=120, stream=True)
            resp.raise_for_status()

            content = resp.content

            # Verify SHA-256 if provided
            if info.sha256:
                actual = hashlib.sha256(content).hexdigest()
                if actual != info.sha256:
                    logger.error(
                        f"SHA-256 mismatch: expected {info.sha256}, got {actual}"
                    )
                    return False

            # Write to temp file (actual application would extract/install)
            update_path = Path(self.repo_path) / ".update_package"
            update_path.write_bytes(content)
            logger.info(f"Update downloaded to {update_path} ({len(content)} bytes)")
            return True

        except Exception as e:
            logger.error(f"HTTP update download failed: {e}")
            return False

    # -- Blob store source --

    def _check_blob_store(self) -> Optional[UpdateInfo]:
        """Check for updates in the blob store."""
        self._last_check = time.time()

        try:
            from .blob_store import BlobStore
            store = BlobStore()
            version_data = store.get_json("autonet_latest_version")
            if not version_data:
                return None

            info = UpdateInfo(
                current_version=self.current_version,
                available_version=version_data.get("version", self.current_version),
                source="blob_store",
                changelog=version_data.get("changelog", ""),
                sha256=version_data.get("sha256", ""),
            )
            self._last_info = info
            return info

        except Exception as e:
            logger.error(f"Blob store update check failed: {e}")
            return None

    def _apply_blob_store(self, info: UpdateInfo) -> bool:
        """Fetch update from blob store."""
        try:
            from .blob_store import BlobStore
            store = BlobStore()
            update_data = store.get(info.sha256)
            if not update_data:
                logger.error("Failed to fetch update from blob store")
                return False

            update_path = Path(self.repo_path) / ".update_package"
            update_path.write_bytes(update_data)
            logger.info(f"Update fetched from blob store ({len(update_data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Blob store update failed: {e}")
            return False
