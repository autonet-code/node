"""
Auto-Update Mechanism for Autonet Nodes.

Checks for new versions of the node software and applies updates.
Supports multiple sources: PyPI (default for the daemon), git, HTTP, or
blob store.

Story 6.5: Auto-update mechanism.

Daemon staging model (see docs/auto_update_design.md)
-----------------------------------------------------
The running daemon never restarts itself. ``stage_update`` downloads and
verifies a release wheel, then writes it to ``{data_dir}/staged_update/``
with a ``pending.json`` marker. The next daemon boot (``atn/update_boot.py``,
before any heavy imports) pip-installs the staged wheel and re-execs once.
This module does NOT install into the running process.
"""

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .config import AutonetConfig, UpdateConfig, load_config

logger = logging.getLogger(__name__)


# Staged-update layout under the daemon data dir (~/.atn by default).
STAGED_DIRNAME = "staged_update"
PENDING_MARKER = "pending.json"


class UpdateSource(Enum):
    PYPI = "pypi"
    GIT = "git"
    HTTP = "http"
    BLOB_STORE = "blob_store"


def _parse_version(v: str) -> tuple:
    """Best-effort PEP 440-ish version tuple for ordering.

    Prefers ``packaging.version`` when available; falls back to a numeric
    dotted-segment tuple so a comparison is always possible. Non-numeric
    suffixes (rc, dev) sort before the bare release under the fallback —
    good enough for "is X strictly newer than Y"; packaging handles the
    real cases.
    """
    try:
        from packaging.version import Version
        return (1, Version(v))
    except Exception:
        parts: list[int] = []
        for seg in str(v).replace("-", ".").split("."):
            num = "".join(ch for ch in seg if ch.isdigit())
            parts.append(int(num) if num else 0)
        return (0, tuple(parts))


def version_is_newer(candidate: str, current: str) -> bool:
    """True if ``candidate`` is a strictly newer version than ``current``."""
    if not candidate or candidate == current:
        return False
    try:
        c, cur = _parse_version(candidate), _parse_version(current)
        # Only compare when both used the same backend (both packaging or both fallback).
        if c[0] == cur[0]:
            return c[1] > cur[1]
        # Mixed backends — fall back to string inequality as "newer".
        return candidate != current
    except Exception:
        return candidate != current


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
        return version_is_newer(self.available_version, self.current_version)


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
        *,
        source: str = "",
        package_name: str = "autonet-computer",
        pypi_index_url: str = "",
        data_dir: Optional[str] = None,
        check_interval: Optional[int] = None,
        rpc_url: str = "",
        registry_address: str = "",
    ):
        self.config = config or load_config()
        self.update_config: UpdateConfig = self.config.update
        self.current_version = current_version
        self.repo_path = repo_path or str(Path.cwd())
        # Daemon-facing overrides (atn AutoUpdateConfig) take precedence over
        # the nodes-layer UpdateConfig so the daemon controls its own updates.
        self.source = source or self.update_config.update_source
        self.package_name = package_name
        self.pypi_index_url = (pypi_index_url or "https://pypi.org/pypi").rstrip("/")
        self.check_interval = (
            check_interval if check_interval is not None
            else self.update_config.check_interval
        )
        # Where staged wheels land. Default ~/.atn (the daemon data dir).
        self.data_dir = Path(data_dir) if data_dir else (Path.home() / ".atn")
        # Optional on-chain integrity anchor (advisory in V1).
        self.rpc_url = rpc_url
        self.registry_address = registry_address

        self._last_check: float = 0.0
        self._last_info: Optional[UpdateInfo] = None
        self._last_error: str = ""
        self._state: str = "idle"  # idle|available|staged|error
        # Seed in-memory last-check from the persisted timestamp so "have we
        # checked within the interval?" survives daemon restarts.
        self._last_check = self._read_last_check()

    @property
    def staged_dir(self) -> Path:
        return self.data_dir / STAGED_DIRNAME

    @property
    def pending_path(self) -> Path:
        return self.staged_dir / PENDING_MARKER

    @property
    def last_check_path(self) -> Path:
        return self.staged_dir / "last_check.json"

    def _read_last_check(self) -> float:
        """Persisted epoch seconds of the last release check, or 0.0."""
        try:
            p = self.last_check_path
            if p.exists():
                return float(json.loads(p.read_text(encoding="utf-8")).get("ts", 0.0))
        except Exception:
            pass
        return 0.0

    def _write_last_check(self, ts: float) -> None:
        try:
            self.staged_dir.mkdir(parents=True, exist_ok=True)
            self.last_check_path.write_text(
                json.dumps({"ts": ts}), encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Could not persist last-check timestamp: %s", exc)

    def seconds_since_last_check(self) -> float:
        """Wall-clock seconds since the last persisted release check."""
        if self._last_check <= 0:
            return float("inf")
        return max(0.0, time.time() - self._last_check)

    def check_update(self) -> Optional[UpdateInfo]:
        """
        Check for available updates from the configured source.

        Returns UpdateInfo if an update is available, None on error.
        """
        source = self.source

        try:
            if source == UpdateSource.PYPI.value:
                info = self._check_pypi()
            elif source == UpdateSource.GIT.value:
                info = self._check_git()
            elif source == UpdateSource.HTTP.value:
                info = self._check_http()
            elif source == UpdateSource.BLOB_STORE.value:
                info = self._check_blob_store()
            else:
                logger.warning(f"Unknown update source: {source}")
                return None
            self._last_error = ""
            # Persist the check time so the interval survives restarts.
            self._write_last_check(self._last_check)
            if info is not None and info.has_update:
                self._state = "available"
            return info
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            self._last_error = str(e)
            self._state = "error"
            return None

    def should_check(self) -> bool:
        """Whether enough time has elapsed since the last check."""
        return time.time() - self._last_check >= self.check_interval

    # ------------------------------------------------------------------
    # Staging (daemon path) — download + verify a wheel, mark pending.
    # The running process is NOT modified; apply happens at next boot.
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Observability snapshot for the frontend / WS surface."""
        pending = self.read_pending()
        return {
            "state": self._state,
            "source": self.source,
            "current_version": self.current_version,
            "available_version": (
                self._last_info.available_version if self._last_info else ""
            ),
            "staged_version": (pending or {}).get("version", ""),
            "pending": pending is not None,
            "last_check": self._last_check,
            "last_error": self._last_error,
        }

    def read_pending(self) -> Optional[dict[str, Any]]:
        """Return the staged ``pending.json`` payload, or None."""
        try:
            if self.pending_path.exists():
                return json.loads(self.pending_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read pending update marker: %s", exc)
        return None

    def stage_update(self, info: UpdateInfo) -> bool:
        """Download + verify the release wheel and stage it for next boot.

        Steps: download wheel → verify sha256 → advisory on-chain core-hash
        check → write wheel to ``staged_update/<version>/`` + ``pending.json``.
        Never installs into the running process. Returns True when staged.
        """
        if not info.has_update:
            logger.info("No update to stage")
            return False
        if not info.download_url:
            logger.error("Cannot stage: no download_url in update info")
            self._last_error = "no download_url"
            self._state = "error"
            return False

        # Already staged at this version? Idempotent — don't re-download.
        existing = self.read_pending()
        if existing and existing.get("version") == info.available_version:
            logger.info("Update %s already staged", info.available_version)
            self._state = "staged"
            return True

        try:
            import requests
            resp = requests.get(info.download_url, timeout=180)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:
            logger.error("Update download failed: %s", exc)
            self._last_error = f"download failed: {exc}"
            self._state = "error"
            return False

        # 1. Wheel integrity — the artifact must match its published hash.
        if info.sha256:
            actual = hashlib.sha256(content).hexdigest()
            if actual.lower() != info.sha256.lower():
                logger.error(
                    "Refusing to stage %s: wheel sha256 mismatch "
                    "(expected %s, got %s)",
                    info.available_version, info.sha256, actual,
                )
                self._last_error = "wheel sha256 mismatch"
                self._state = "error"
                return False
        else:
            logger.warning(
                "Staging %s with NO published wheel hash — integrity unverified",
                info.available_version,
            )

        # 2. On-chain core-hash anchor (advisory in V1). Refuse only on a
        #    confirmed mismatch; fail open (with a log) when the chain or the
        #    Registry key is unavailable — matches atn/_cache.validate semantics.
        if not self._verify_core_hash(info.available_version):
            logger.error(
                "Refusing to stage %s: on-chain core-hash mismatch",
                info.available_version,
            )
            self._last_error = "on-chain core-hash mismatch"
            self._state = "error"
            return False

        # 3. Write the wheel + marker.
        try:
            version_dir = self.staged_dir / info.available_version
            version_dir.mkdir(parents=True, exist_ok=True)
            wheel_name = info.download_url.rsplit("/", 1)[-1] or f"{self.package_name}-{info.available_version}.whl"
            wheel_path = version_dir / wheel_name
            wheel_path.write_bytes(content)
            marker = {
                "version": info.available_version,
                "wheel_path": str(wheel_path),
                "sha256": info.sha256,
                "source": info.source,
                "staged_at": time.time(),
                "package_name": self.package_name,
            }
            # Atomic-ish: write to temp then replace.
            tmp = self.pending_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
            os.replace(tmp, self.pending_path)
        except Exception as exc:
            logger.error("Failed to write staged update: %s", exc)
            self._last_error = f"stage write failed: {exc}"
            self._state = "error"
            return False

        logger.info(
            "Staged update %s → %s (applies on next daemon restart)",
            info.available_version, wheel_path,
        )
        self._state = "staged"
        return True

    def _verify_core_hash(self, version: str) -> bool:
        """Advisory on-chain core-hash check. True = proceed.

        Returns True when verification is unavailable (no rpc/registry, key
        not published, chain unreachable) — but logs at WARNING so an
        unverified stage is visible rather than silent. Returns False only
        on a confirmed mismatch against the published on-chain hash.
        """
        if not self.rpc_url or not self.registry_address:
            logger.warning(
                "Staging %s without on-chain verification "
                "(no rpc_url/registry_address configured)", version,
            )
            return True
        try:
            from atn._cache import validate
            ok = validate(self.rpc_url, self.registry_address, version)
            if ok:
                logger.info("On-chain core-hash check passed (or unpublished) for %s", version)
            return ok
        except Exception as exc:
            logger.warning("On-chain verification unavailable for %s: %s", version, exc)
            return True

    def clear_pending(self) -> None:
        """Remove the staged-update marker (after apply or on failure)."""
        try:
            if self.pending_path.exists():
                self.pending_path.unlink()
        except Exception as exc:
            logger.warning("Failed to clear pending marker: %s", exc)

    # ------------------------------------------------------------------
    # PyPI source
    # ------------------------------------------------------------------

    def _check_pypi(self) -> Optional[UpdateInfo]:
        """Check PyPI's JSON API for the latest release of the package.

        Picks the newest non-prerelease version and its wheel (bdist_wheel)
        download URL + sha256 from the release files.
        """
        self._last_check = time.time()
        url = f"{self.pypi_index_url}/{self.package_name}/json"
        try:
            import requests
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("PyPI update check failed: %s", exc)
            self._last_error = str(exc)
            return None

        latest = (data.get("info") or {}).get("version", self.current_version)
        download_url = ""
        sha256 = ""
        # Find a wheel for the latest version among its release files.
        releases = data.get("releases") or {}
        files = releases.get(latest) or []
        # Prefer a py3 wheel; fall back to any wheel, then any file.
        wheel = None
        for f in files:
            if f.get("packagetype") == "bdist_wheel":
                wheel = f
                if "py3" in (f.get("python_version") or "") or "py3" in (f.get("filename") or ""):
                    break
        if wheel is None and files:
            wheel = files[0]
        if wheel:
            download_url = wheel.get("url", "")
            sha256 = (wheel.get("digests") or {}).get("sha256", "")

        info = UpdateInfo(
            current_version=self.current_version,
            available_version=latest,
            source="pypi",
            changelog="",
            download_url=download_url,
            sha256=sha256,
        )
        self._last_info = info
        if info.has_update:
            logger.info(
                "PyPI update available: %s → %s",
                info.current_version, info.available_version,
            )
        return info

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

        source = self.source

        try:
            if source == UpdateSource.PYPI.value:
                # PyPI uses the stage→boot-apply path, not in-process install.
                return self.stage_update(info)
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
