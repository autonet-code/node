"""Root test fixtures: keep every test away from the developer's real keystore.

The age keystore resolves its paths from ``KEYSTORE_DIR`` at import time and
defaults to ``~/.atn/keystore``. Any test that registers an agent, stores a
credential or puts a secret would otherwise write into the developer's REAL
vault. That happened for months (the vault carried 125 test-agent keys), and
on 2026-09-16 several test processes plus the running daemon wrote the vault
concurrently and tore it (a torn ``vault.age`` that no longer decrypted; it
was recovered by truncation). Two defences:

1. this file sets ``KEYSTORE_DIR`` to a per-session temp dir BEFORE any
   ``atn`` module is imported (conftest loads first), so late imports pick
   it up automatically;
2. the autouse fixture below also repoints the module-level path constants
   of any keystore module that is already imported, in case a plugin or an
   earlier conftest pulled one in.
"""

import os
import sys
import tempfile

import pytest

_SESSION_KEYSTORE = tempfile.mkdtemp(prefix="atn-test-keystore-")
os.environ["KEYSTORE_DIR"] = _SESSION_KEYSTORE


def _repoint(module, base: str) -> None:
    names = {
        "KEYSTORE_DIR": base,
        "KEY_PATH": os.path.join(base, "identity.age-key"),
        "VAULT_PATH": os.path.join(base, "vault.age"),
        "BUNDLES_PATH": os.path.join(base, "bundles.json"),
        "SECRET_META_PATH": os.path.join(base, "secret_meta.json"),
        "EXPORT_DIR": os.path.join(base, "exports"),
        "SESSIONS_DIR": os.path.join(base, "sessions"),
        "_KEYSTORE_DIR": base,
    }
    for name, value in names.items():
        if hasattr(module, name):
            setattr(module, name, value)


@pytest.fixture(autouse=True, scope="session")
def _isolated_keystore():
    """Never let a test touch ``~/.atn/keystore``."""
    for mod_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if mod_name in ("keystore", "atn._vendor.kevin.keystore",
                        "atn.runtime.broker_client") or mod_name.endswith(".keystore"):
            _repoint(module, _SESSION_KEYSTORE)
    yield
