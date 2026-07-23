"""Credential storage for connector integrations and provider API keys.

Backend: the age-encrypted vault (``atn/_vendor/kevin/keystore.py``). Every
CredentialStore payload is stored under the reserved dotted prefix ``app.`` as a
JSON blob string, e.g. ``app.google_calendar`` -> ``{"access_token": ...}``.

DAEMON-PLANE / GRANTABILITY ISOLATION
-------------------------------------
Vault names containing a dot are DAEMON-PLANE and MUST NEVER be agent-grantable
or user-managed through the generic secrets surface. The ``app.`` prefix is
dotted, so CredentialStore payloads are structurally excluded from every
agent-facing allowance-resolution / policy-map path (see
``atn/vault_setup.generate_policy_map`` and the ws/worker choke points). Flat
names (no dot) remain the agent-grantable plane, unchanged.

NO PLAINTEXT FALLBACK
---------------------
This class NEVER writes plaintext JSON credential files and NEVER writes to the
OS keyring. The previous keyring-primary + always-written-plaintext design is
gone. On first construction it performs a one-shot migration of any legacy
``data_dir/credentials/*.json`` files into the vault and DELETES them (the whole
point is removing the plaintext copies), and best-effort clears the matching
keyring entries.

LIMITATION: keyring cannot be enumerated, so legacy credentials that lived ONLY
in the keyring (no fallback file) cannot be discovered here for migration. Only
file-based legacy entries are migrated. In practice the old code always wrote a
fallback file alongside any keyring entry, so this window is empty for normal
installs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Reserved daemon-plane prefix for CredentialStore payloads. Dotted => never
# agent-grantable (see module docstring).
PREFIX = "app."


def _keystore():
    """Import the vendored age keystore (authoritative in the wheel), falling
    back to a dev checkout of kevin on the path. Raises on total unavailability
    (caller degrades)."""
    try:
        from atn._vendor.kevin import keystore  # type: ignore
        return keystore
    except Exception:
        import keystore  # type: ignore
        return keystore


class CredentialStore:
    """Manages credentials for integrations and providers, backed by the vault.

    Usage:
        store = CredentialStore(data_dir)
        store.save("google_calendar", {"access_token": "...", "refresh_token": "..."})
        creds = store.load("google_calendar")  # -> dict or {}
        store.delete("google_calendar")

    All entries live in the vault under ``app.<integration_id>``. No plaintext
    file or keyring copy is ever written.
    """

    def __init__(self, data_dir: Path) -> None:
        # Legacy fallback directory — only read (for one-shot migration) and
        # deleted, NEVER written to going forward.
        self._dir = Path(data_dir) / "credentials"
        try:
            self._ks = _keystore()
        except Exception:
            self._ks = None
            log.warning(
                "CredentialStore: age keystore unavailable — credentials cannot "
                "be persisted this session (save will raise; load returns empty)",
                exc_info=True,
            )
        self._migrate_legacy_files()

    # ------------------------------------------------------------------
    # One-shot legacy migration
    # ------------------------------------------------------------------

    def _migrate_legacy_files(self) -> None:
        """Move any ``data_dir/credentials/*.json`` into the vault, then delete
        the plaintext file. If the vault already holds the entry, still delete
        the file. Best-effort keyring cleanup for each migrated id."""
        if not self._dir.exists():
            return
        try:
            files = list(self._dir.glob("*.json"))
        except OSError:
            return
        if not files:
            return
        migrated_ids: list[str] = []
        for path in files:
            stem = path.stem
            key = PREFIX + stem
            try:
                have = self._ks is not None and key in self._ks.list_services()
            except Exception:
                have = False
            if self._ks is not None and not have:
                # Vault lacks it — read the file and put it in.
                try:
                    payload = path.read_text(encoding="utf-8")
                    self._ks.put_secret(key, payload)
                except Exception:
                    # Could not migrate this one; leave the file in place so no
                    # data is lost, and move on.
                    log.warning("CredentialStore: legacy migration failed for %s",
                                stem, exc_info=True)
                    continue
            # Either the vault already had it or we just wrote it: drop the
            # plaintext file (the entire point of the migration).
            try:
                path.unlink()
            except OSError:
                log.debug("CredentialStore: could not remove legacy file for %s",
                          stem, exc_info=True)
            migrated_ids.append(stem)
        # Best-effort keyring cleanup for migrated ids. Swallow everything,
        # including keyring being unimportable.
        if migrated_ids:
            try:
                import keyring
                for cid in migrated_ids:
                    try:
                        keyring.delete_password("atn", cid)
                    except Exception:
                        pass
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    def save(self, integration_id: str, credentials: dict[str, Any]) -> None:
        """Store credentials for an integration in the vault. Raises if the
        vault is unavailable (the caller MUST know the write did not persist)."""
        if self._ks is None:
            raise RuntimeError("credential vault unavailable; cannot save credentials")
        payload = json.dumps(credentials)
        self._ks.put_secret(PREFIX + integration_id, payload)
        log.debug("Credentials saved for %s", integration_id)

    def load(self, integration_id: str) -> dict[str, Any]:
        """Load credentials. Returns {} if missing, corrupt, or vault down."""
        if self._ks is None:
            return {}
        try:
            raw = self._ks.get_secret(PREFIX + integration_id)
        except KeyError:
            return {}
        except Exception:
            log.warning("Failed to load credentials for %s", integration_id,
                        exc_info=True)
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            log.warning("Corrupt credential payload for %s", integration_id)
            return {}

    def delete(self, integration_id: str) -> None:
        """Delete credentials for an integration from the vault."""
        if self._ks is None:
            return
        try:
            self._ks.delete_secret(PREFIX + integration_id)
        except Exception:
            log.debug("Credentials delete failed for %s", integration_id,
                      exc_info=True)
        log.debug("Credentials deleted for %s", integration_id)

    def exists(self, integration_id: str) -> bool:
        """Check if credentials exist for an integration."""
        if self._ks is None:
            return False
        try:
            return (PREFIX + integration_id) in self._ks.list_services()
        except Exception:
            return False

    def list_integrations(self) -> list[str]:
        """Return IDs of all integrations with stored credentials (vault
        ``app.*`` names, plus any legacy file stems not yet migrated)."""
        ids: list[str] = []
        if self._ks is not None:
            try:
                ids = [name[len(PREFIX):] for name in self._ks.list_services()
                       if name.startswith(PREFIX)]
            except Exception:
                ids = []
        # Dedupe against any remaining legacy files (migration is best-effort).
        seen = set(ids)
        try:
            for p in self._dir.glob("*.json"):
                if p.stem not in seen:
                    ids.append(p.stem)
                    seen.add(p.stem)
        except OSError:
            pass
        return ids
