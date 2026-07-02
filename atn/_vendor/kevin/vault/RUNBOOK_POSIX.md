# Vault credential broker — POSIX setup runbook

The vault ships in the wheel and WORKS out of the box (self-contained
age-encrypted keystore, no HashiCorp). `atn-vault-setup` provisions the data
plane (age identity + `service_policy_map.json`) and tightens perms it can.

What remains is the **privileged setup only root can do**: running the PID-auth
broker under a **separate OS uid** so a same-daemon-uid agent cannot read the age
key out of the keystore or the broker's memory. That uid split IS the security
boundary — without it the broker is just the same-user design again.

## The trust model in one paragraph
The broker runs as a dedicated low-priv user **`vault-svc`**. Agents run as the
daemon uid (e.g. `ubuntu`). Agents never hold the age key — they connect to the
broker's **AF_UNIX socket**; the broker reads their kernel PID+uid via
`SO_PEERCRED` (the POSIX analog of `GetNamedPipeClientProcessId`), looks up the
policies the *owner* granted that session, decrypts the secret with the age key
(readable only by `vault-svc`), and stages the value to a nameless file — it
returns `{var_name, path}`, never the value. Because the age key + owner-secret
live in a `vault-svc` process, an agent running as a *different* uid can't read
them.

## ⚠️ THREAT MODEL — read before deploying
This defends secrets against **misaligned local AI agents** you run that may
actively try to break out. The boundary is **OS uid separation**: a process
owned by uid A cannot `ptrace`/read the memory of a process owned by uid B.

The boundary is bypassed by **`CAP_SYS_PTRACE` / root**. Any process with it can
read `vault-svc`'s memory and lift the age key. Therefore the boundary holds only
if **agents do not run as root and cannot gain `CAP_SYS_PTRACE`**.

**REQUIRED for the boundary to hold:**
- Run all AI agents as a **non-root uid**, distinct from `vault-svc`.
- Keep the agent uid out of `sudo`/`wheel` and any capability grant.
- Set **Yama `ptrace_scope >= 1`** (`sysctl kernel.yama.ptrace_scope=1`) so even a
  same-uid ptrace is blocked to non-descendants; cross-uid is blocked outright.
- The broker additionally calls `PR_SET_DUMPABLE(0)` on itself at startup, so it
  is non-dumpable even to a same-uid peer (belt-and-suspenders).

## Steps that need root (one-time)

### 1. Create the vault-svc user
```
sudo useradd --system --no-create-home vault-svc
```

### 2. Own the keystore by vault-svc, lock it down
```
sudo chown -R vault-svc:vault-svc "$KEYSTORE_DIR"     # e.g. ~/.atn/keystore
sudo chmod 700 "$KEYSTORE_DIR"
sudo chmod 600 "$KEYSTORE_DIR/identity.age-key"
```
The daemon uid gets NOTHING on the keystore dir — that's the point. (The daemon
reaches secrets only via the broker socket, never by reading files.)

### 3. Owner secret in a root-only env file
```
sudo install -d -m 700 /etc/atn
printf 'BROKER_OWNER_SECRET=%s\n' "$(openssl rand -hex 32)" | \
  sudo tee /etc/atn/vault-broker.env >/dev/null
sudo chmod 600 /etc/atn/vault-broker.env
```
The daemon needs this SAME secret to `mint_nonce`/`release_session`. Provide it
to the daemon process out-of-band (its own env), NOT on any agent-readable path.

### 4. Harden ptrace
```
echo 'kernel.yama.ptrace_scope=1' | sudo tee /etc/sysctl.d/10-atn-ptrace.conf
sudo sysctl --system
```

### 5. Install + start the broker service
`atn-vault-setup` writes a unit template to `$KEYSTORE_DIR/atn-vault-broker.service`.
```
sudo cp "$KEYSTORE_DIR/atn-vault-broker.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now atn-vault-broker
systemctl status atn-vault-broker        # expect: listening on <sock> (age keystore, local)
```

### 6. The nonce-mint stays owner-side
The daemon (holding `BROKER_OWNER_SECRET`) mints a one-time nonce per worker via
`mint_nonce(services)` and passes it to the worker's `register`. The owner secret
must never enter agent space — the daemon process holds it, agents do not.

## What's already verified (no action needed)
- AF_UNIX broker + `SO_PEERCRED` peer-PID auth (kernel-authenticated, unforgeable).
- End-to-end on Linux: granted service served as `{var_name,path}` (value never in
  the reply); value-push tripwire delivers the raw value to the daemon monitor;
  staged file 0600.
- Fail-closed: unregistered request denied; no-tripwire request denied (M7);
  PID-reuse (`_identity` mismatch) denied; post-`release_session` denied + staged
  file unlinked.
