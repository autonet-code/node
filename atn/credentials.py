"""Cross-platform credential storage using the OS keychain.

Uses the ``keyring`` library which delegates to:
  - Windows: Windows Credential Locker (WinVault)
  - macOS:   macOS Keychain
  - Linux:   Secret Service (GNOME Keyring / KWallet)

Each integration's credentials are stored as a single JSON blob under the
service name ``atn`` with the integration ID as the username.

Fallback: if keyring is unavailable (headless CI, etc.) credentials are
stored as JSON files under ``data_dir/credentials/``.  Not encrypted — for
development only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SERVICE_NAME = "atn"


def _keyring_available() -> bool:
    """Return True if keyring is importable and has a usable backend."""
    try:
        import keyring
        backend = keyring.get_keyring()
        # The fail backend means no real backend is configured
        name = type(backend).__name__
        if "Fail" in name or "fail" in name:
            return False
        return True
    except Exception:
        return False


class CredentialStore:
    """Manages credentials for integrations and providers.

    Primary: OS keychain via ``keyring``.
    Fallback: plain JSON files (dev only).

    Usage:
        store = CredentialStore(data_dir)
        store.save("google_calendar", {"access_token": "...", "refresh_token": "..."})
        creds = store.load("google_calendar")  # -> dict or {}
        store.delete("google_calendar")
    """

    def __init__(self, data_dir: Path) -> None:
        self._use_keyring = _keyring_available()
        # Fallback directory (always available for list_integrations scan)
        self._dir = data_dir / "credentials"
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._use_keyring:
            log.debug("CredentialStore: using OS keychain")
        else:
            log.warning("CredentialStore: keyring unavailable, using plaintext JSON fallback")

    def save(self, integration_id: str, credentials: dict[str, Any]) -> None:
        """Store credentials for an integration."""
        payload = json.dumps(credentials)
        if self._use_keyring:
            import keyring
            keyring.set_password(_SERVICE_NAME, integration_id, payload)
        # Always write fallback file too — used by list_integrations()
        # and as a backup if keyring is lost.
        path = self._dir / f"{integration_id}.json"
        path.write_text(payload, encoding="utf-8")
        log.debug("Credentials saved for %s", integration_id)

    def load(self, integration_id: str) -> dict[str, Any]:
        """Load credentials.  Returns {} if missing or corrupted."""
        if self._use_keyring:
            try:
                import keyring
                raw = keyring.get_password(_SERVICE_NAME, integration_id)
                if raw:
                    return json.loads(raw)
            except Exception:
                log.warning("Failed to load from keyring for %s", integration_id, exc_info=True)
        # Fallback to file
        path = self._dir / f"{integration_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Failed to read credential file for %s", integration_id, exc_info=True)
        return {}

    def delete(self, integration_id: str) -> None:
        """Delete credentials for an integration."""
        if self._use_keyring:
            try:
                import keyring
                keyring.delete_password(_SERVICE_NAME, integration_id)
            except Exception:
                pass  # may not exist
        path = self._dir / f"{integration_id}.json"
        if path.exists():
            path.unlink()
        log.debug("Credentials deleted for %s", integration_id)

    def exists(self, integration_id: str) -> bool:
        """Check if credentials exist for an integration."""
        if self._use_keyring:
            try:
                import keyring
                raw = keyring.get_password(_SERVICE_NAME, integration_id)
                if raw:
                    return True
            except Exception:
                pass
        return (self._dir / f"{integration_id}.json").exists()

    def list_integrations(self) -> list[str]:
        """Return IDs of all integrations with stored credentials."""
        return [p.stem for p in self._dir.glob("*.json")]
