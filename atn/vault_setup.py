"""atn-vault-setup — initialize the self-contained agent secret vault.

Ships in core (``pip install autonet-computer`` brings it). Performs the
data-plane setup pip *can* do — the age identity + the service->policy identity
map the broker and daemon read — and prints the one elevated OS step pip
*cannot*: running the PID-auth broker as a separate, locked-down OS identity so
a same-user agent can't read the age key and decrypt the whole vault.

The vault backend is the self-contained age-encrypted keystore
(``atn/_vendor/kevin/keystore.py``); there is no HashiCorp Vault dependency.
Runtime activation of the whole isolation+vault path is still gated by
``ATN_WORKER_ISOLATION`` — this command only provisions the store.
"""
from __future__ import annotations

import json
import os
import sys


def _keystore():
    """Import the vendored age keystore (authoritative in the wheel), falling
    back to a dev checkout of kevin on the path."""
    try:
        from atn._vendor.kevin import keystore  # type: ignore
        return keystore
    except Exception:
        import keystore  # type: ignore
        return keystore


def _policy_map_path(ks) -> str:
    """Where the broker (_MAP_PATH) and daemon (broker_client._SERVICE_POLICY_MAP)
    both read the service->policy map — the keystore data dir, env-overridable."""
    return os.environ.get(
        "ATN_VAULT_POLICY_MAP",
        os.path.join(ks.KEYSTORE_DIR, "service_policy_map.json"))


def generate_policy_map(ks) -> dict:
    """Identity map (policy == service) over the vault's current services.

    The self-contained age backend has no HashiCorp policy layer, so the broker's
    policy indirection collapses to a pass-through: each service maps to a policy
    of the same name. Regenerate after adding secrets."""
    return {svc: svc for svc in ks.list_services()}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ks = _keystore()

    # 1. age identity + keystore dir (created if absent).
    pub = ks.init_identity()
    print(f"[vault] keystore dir : {ks.KEYSTORE_DIR}")
    print(f"[vault] identity     : ready (age public key {pub})")

    # 2. service->policy identity map — what the broker + daemon read to authorize.
    mapping = generate_policy_map(ks)
    mp = _policy_map_path(ks)
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[vault] policy map   : {mp} ({len(mapping)} service(s))")
    if not mapping:
        print("[vault] note         : vault is empty. Add secrets with "
              "keystore.put_secret(service, value), then re-run to refresh "
              "the map.")

    # 3. the elevated OS step pip cannot do (hardening against same-user agents).
    print()
    print("[vault] NEXT (elevated, one-time): run the PID-auth broker as a "
          "separate, locked-down OS identity so agents cannot read the age key")
    print("        (identity.age-key). Until that is done the vault WORKS but is "
          "NOT hardened against a same-user agent reading the key directly.")
    print("        Broker service setup: atn/_vendor/kevin/vault/RUNBOOK.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
