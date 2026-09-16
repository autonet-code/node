# Vault credential broker: Windows setup runbook

The vault ships in the wheel and WORKS out of the box (self-contained
age-encrypted keystore, no HashiCorp). `atn-vault-setup` provisions the data
plane (age identity + `service_policy_map.json`).

What remains is the **privileged setup only an administrator can do**: running
the PID-auth broker under a **separate OS account** so a same-user agent cannot
read the age key out of the keystore or out of the broker's memory. That account
split IS the security boundary; without it the broker is just the same-user
design again.

## The trust model in one paragraph
The broker runs as a dedicated low-priv user **`vault-svc`**. Agents run as the
daemon's own account. Agents never hold the age key: they connect to the
broker's **named pipe** (`\\.\pipe\vault-broker`); the broker reads their kernel
PID via `GetNamedPipeClientProcessId`, looks up the policies the *owner* granted
that session, decrypts the secret with the age key (readable only by
`vault-svc`), and stages the value to a nameless file. It returns
`{var_name, path}`, never the value. Because the age key and the owner secret
live in a `vault-svc` process, an agent running as a *different* OS user cannot
read them.

## ⚠️ THREAT MODEL: read this before deploying
This system defends secrets against **misaligned local AI agents**: agents you
run that may actively try to break out. The entire boundary is **OS user
separation**, a process owned by user A cannot `ReadProcessMemory` a process
owned by user B. That is the only thing keeping a misaligned agent out of the
broker's age key.

**This boundary is bypassed by `SeDebugPrivilege`.** Any process holding it can
read *any* process's memory regardless of owner, including `vault-svc`.
`SeDebugPrivilege` is available to **Administrators when elevated**.

Therefore the boundary is REAL only if **agents do not run as a local
administrator**. A misaligned agent that runs as an admin account, even
unelevated, has interaction-free elevation paths (UAC auto-elevation bypasses
like `fodhelper`/`computerdefaults`, service/scheduled-task creation as SYSTEM).
Once elevated it gets `SeDebugPrivilege` and reads the key out of `vault-svc`'s
memory. "Assume agents stay non-elevated" is an assumption a *misaligned* agent
will not honor.

**REQUIRED for the boundary to hold:**
- Run all AI agents as a **dedicated standard (non-admin) OS user**, NOT as your
  admin account, NOT as any account in the Administrators group.
- Keep that user out of Administrators, Backup Operators, and any group that
  confers debug/restore privileges.
- Optionally deny `SeDebugPrivilege` to that user outright via Local Security
  Policy (`secpol.msc` → Local Policies → User Rights Assignment → Debug
  programs), as belt-and-suspenders against a later misconfiguration.

If you run agents as an admin account, this system still *raises the cost* over
the naive same-user design (the agent must now know to elevate and run a
bypass), but it does **not** close the hole against a determined misaligned
agent. Do not deploy it that way and believe the boundary holds.

## Steps that need an administrator (one-time)

Placeholders used below:
- `<KEYSTORE_DIR>`: the keystore directory `atn-vault-setup` printed (default
  `%USERPROFILE%\.atn\keystore`).
- `<PYTHON>`: the full path to the interpreter autonet is installed into
  (`python -c "import sys; print(sys.executable)"`).
- `<AGENT-USER>`: the standard account the daemon and its agents run as.

### 1. Create the two accounts (elevated shell)
```
net user vault-svc <a-strong-password> /add
net user <AGENT-USER> <a-strong-password> /add
```
Keep BOTH out of Administrators. `vault-svc` only runs the service, it never
needs interactive logon.

Verify:
```
net localgroup Administrators            # neither account may appear
```

### 2. Lock the keystore to vault-svc (elevated)
```
icacls "<KEYSTORE_DIR>" /inheritance:r ^
  /grant vault-svc:(OI)(CI)F Administrators:(OI)(CI)F
icacls "<KEYSTORE_DIR>\identity.age-key" /inheritance:r ^
  /grant vault-svc:F Administrators:F
```
`<AGENT-USER>` gets NOTHING on the keystore directory, that is the point. The
daemon reaches secrets only over the broker pipe, never by reading files.

### 3. Owner secret in the service environment (elevated)
Generate one and keep it out of agent space:
```
powershell -Command "[guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')"
```
Set it (and the keystore location) as the service's own environment rather than
on the command line, so it never appears in any process's argv:
```
reg add "HKLM\SYSTEM\CurrentControlSet\Services\atn-vault-broker" ^
  /v Environment /t REG_MULTI_SZ ^
  /d "BROKER_OWNER_SECRET=<the value>\0KEYSTORE_DIR=<KEYSTORE_DIR>" /f
```
The daemon needs the SAME secret to `mint_nonce` / `release_session`. Provide it
to the daemon process out-of-band (its own environment), never on any
agent-readable path.

### 4. Install and start the broker service (elevated)
```
sc create atn-vault-broker ^
  binPath= "\"<PYTHON>\" -m atn._vendor.kevin.vault.vault_broker" ^
  obj= ".\vault-svc" password= "<vault-svc password>" start= auto
sc start atn-vault-broker
```
Expect the broker to log `listening on \\.\pipe\vault-broker (age keystore,
local)`.

### 5. Verify from the app
Open the daemon's Secrets tab. It should show **Broker push: armed** (the
`push_armed` status field). If it does not, the daemon could not reach the
broker: check that the service is running as `vault-svc` and that the daemon
holds the same `BROKER_OWNER_SECRET`.

### 6. The nonce-mint stays owner-side
The daemon (holding `BROKER_OWNER_SECRET`) mints a one-time nonce per worker via
`mint_nonce(services)` and passes it to the worker's `register`. The owner secret
must never enter agent space: the daemon process holds it, agents do not.

## What's already verified (no action needed)
- Named-pipe broker with `GetNamedPipeClientProcessId` peer-PID auth
  (kernel-authenticated, unforgeable).
- Granted service served as `{var_name, path}`, the value never in the reply;
  value-push tripwire delivers the raw value to the daemon monitor.
- Fail-closed: unregistered request denied, no-tripwire request denied, PID-reuse
  (identity mismatch) denied, post-`release_session` denied with the staged file
  unlinked.

See `RUNBOOK_POSIX.md` for the Linux/macOS equivalent (AF_UNIX socket,
`SO_PEERCRED`, systemd unit).
