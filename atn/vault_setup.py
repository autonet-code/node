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


def write_policy_map(ks) -> str:
    """Regenerate + persist the service->policy map. Returns the map path.

    Called by ``main`` below and by the WS ``secrets_put``/``secrets_delete``
    handlers after a vault mutation. NOTE: the broker caches this map at
    import, so a newly ADDED service needs a broker restart to become
    grantable; rotation of an existing name works live."""
    mapping = generate_policy_map(ks)
    mp = _policy_map_path(ks)
    d = os.path.dirname(mp)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return mp


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ks = _keystore()

    # 1. age identity + keystore dir (created if absent).
    pub = ks.init_identity()
    print(f"[vault] keystore dir : {ks.KEYSTORE_DIR}")
    print(f"[vault] identity     : ready (age public key {pub})")

    # 2. service->policy identity map — what the broker + daemon read to authorize.
    mp = write_policy_map(ks)
    mapping = generate_policy_map(ks)
    print(f"[vault] policy map   : {mp} ({len(mapping)} service(s))")
    if not mapping:
        print("[vault] note         : vault is empty. Add secrets with "
              "keystore.put_secret(service, value), then re-run to refresh "
              "the map.")

    # 3. tighten perms on what pip CAN do (data-plane hardening).
    _tighten_perms(ks)

    # 4. the elevated OS step pip cannot do (hardening against same-user agents).
    if os.name == "nt":
        _print_windows_next()
    else:
        _print_posix_next(ks)
    return 0


def _tighten_perms(ks) -> None:
    """0700 the keystore dir and 0600 the age key — pip-doable hardening so the
    key is at least not world/group readable before the elevated uid split."""
    if os.name == "nt":
        return  # POSIX-mode bits are meaningless on Windows; ACLs are the model
    try:
        os.chmod(ks.KEYSTORE_DIR, 0o700)
    except OSError:
        pass
    # keystore.py stores the age identity as identity.age-key in KEYSTORE_DIR.
    for name in ("identity.age-key", "identity.key", "key.age"):
        p = os.path.join(ks.KEYSTORE_DIR, name)
        if os.path.exists(p):
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass


def _print_windows_next() -> None:
    print()
    print("[vault] NEXT (elevated, one-time): run the PID-auth broker as a "
          "separate, locked-down OS identity so agents cannot read the age key")
    print("        (identity.age-key). Until that is done the vault WORKS but is "
          "NOT hardened against a same-user agent reading the key directly.")
    print("        Broker service setup: atn/_vendor/kevin/vault/RUNBOOK.md")


def _print_posix_next(ks) -> None:
    """POSIX elevated guidance: create a separate broker uid, chown the key to it,
    and run the broker under a systemd unit. Emit a ready-to-install unit file."""
    unit_path = os.path.join(ks.KEYSTORE_DIR, "atn-vault-broker.service")
    try:
        with open(unit_path, "w", encoding="utf-8") as f:
            f.write(_SYSTEMD_UNIT.format(
                keystore_dir=ks.KEYSTORE_DIR,
                python=sys.executable,
            ))
        print(f"[vault] systemd unit : {unit_path} (template)")
    except OSError:
        unit_path = "(could not write unit template)"
    print()
    print("[vault] NEXT (root, one-time): run the PID-auth broker as a SEPARATE")
    print("        uid so a same-daemon-uid agent cannot read the age key. Until")
    print("        then the vault WORKS but is NOT hardened against a same-uid")
    print("        agent reading identity.age-key directly. Steps:")
    print("          1. sudo useradd --system --no-create-home vault-svc")
    print(f"          2. sudo chown -R vault-svc {ks.KEYSTORE_DIR}")
    print(f"          3. sudo chmod 700 {ks.KEYSTORE_DIR}")
    print("          4. ensure Yama ptrace_scope>=1 "
          "(sysctl kernel.yama.ptrace_scope=1) so agents can't ptrace the broker")
    print(f"          5. sudo cp {unit_path} /etc/systemd/system/ && \\")
    print("             sudo systemctl enable --now atn-vault-broker")
    print("        Full runbook: atn/_vendor/kevin/vault/RUNBOOK_POSIX.md")


# systemd unit template. Runs the broker as vault-svc with the owner secret from
# an EnvironmentFile (root-only, 0600) — never in the unit or the process argv.
_SYSTEMD_UNIT = """\
[Unit]
Description=ATN vault-broker (PID-auth secret broker)
After=network.target

[Service]
Type=simple
User=vault-svc
Group=vault-svc
# BROKER_OWNER_SECRET lives here, root-owned 0600 — NOT in this unit file.
EnvironmentFile=-/etc/atn/vault-broker.env
Environment=KEYSTORE_DIR={keystore_dir}
ExecStart={python} -m atn._vendor.kevin.vault.vault_broker
# Hardening: no new privileges, private tmp, read-only system, non-dumpable.
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths={keystore_dir}
Restart=on-failure

[Install]
WantedBy=multi-user.target
"""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
