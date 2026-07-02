r"""Vendored kevin vault: self-contained age keystore + PID-auth broker.

Source of truth: C:\code\kevin (keystore.py, vault/vault_broker.py). Ships in
core so `pip install autonet-computer` brings the vault; runtime activation is
gated by ATN_WORKER_ISOLATION. Backend is a local age-encrypted keystore — no
HashiCorp Vault dependency.
"""
