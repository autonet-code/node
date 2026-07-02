# Vault credential broker — setup runbook

The code is built and tested (logic verified end-to-end against a dev Vault).
What remains is the **privileged setup only you can do** — creating the separate
OS user and handling your offline master key. That separation IS the security
boundary; without it the broker is just the old same-user design again.

## The trust model in one paragraph
Vault + the broker run as a dedicated low-priv user **`vault-svc`**. Agents run
as you (`astmo`). Agents never hold a Vault token — they ask the broker over a
named pipe; the broker reads their kernel PID, looks up the policies the *owner*
granted that session, fetches from Vault with the broker's OWN token, and stages
the value to a nameless file. Because the broker's token + owner-secret live in a
`vault-svc` process, an agent running as a *different* OS user can't read them.

## ⚠️ THREAT MODEL — read this before deploying
This system defends secrets against **misaligned local AI agents** — agents you
run that may actively try to break out. The entire boundary is **OS user
separation**: a process owned by user A cannot `ReadProcessMemory` a process
owned by user B. That is the only thing keeping a misaligned agent out of the
broker's Vault token.

**This boundary is bypassed by `SeDebugPrivilege`.** Any process holding it can
read *any* process's memory regardless of owner — including `vault-svc`, which
hands over the broker's Vault token directly and defeats everything below it.
`SeDebugPrivilege` is available to **Administrators when elevated**.

Therefore the boundary is REAL only if **agents do not run as a local
administrator**. A misaligned agent that runs as an admin account — even
unelevated — has interaction-free elevation paths (UAC auto-elevation bypasses
like `fodhelper`/`computerdefaults`, service/scheduled-task creation as SYSTEM).
Once elevated it gets `SeDebugPrivilege` and reads the token out of `vault-svc`'s
memory. "Assume agents stay non-elevated" is an assumption a *misaligned* agent
will not honor.

**REQUIRED for the boundary to hold:**
- Run all AI agents as a **dedicated standard (non-admin) OS user** — NOT as your
  admin account, NOT as any account in the Administrators group.
- Explicitly **deny `SeDebugPrivilege`** to that user (belt-and-suspenders: even a
  future misconfig or privilege grant won't reopen the hole). See step 0 below.
- Keep that user out of Administrators, Backup Operators, and any group that
  confers debug/restore privileges.

If you run agents as an admin account, this system still *raises the cost* over
the naive same-user design (the agent must now know to elevate and run a bypass),
but it does **not** close the hole against a determined misaligned agent. Do not
deploy it that way and believe the boundary holds.

## Steps that need YOU (admin + your offline key)

### 0. Create the dedicated AGENT user and strip its privileges  (admin shell)
This is the step that makes the whole boundary real (see THREAT MODEL above).
Run agents as THIS user, never as your admin account.
```
net user agent-user <a-strong-password> /add
# standard user only — do NOT add to Administrators or any debug/backup group.

# Explicitly deny SeDebugPrivilege (belt-and-suspenders). Run the helper:
powershell -ExecutionPolicy Bypass -File C:\code\kevin\vault\deny-debug-privilege.ps1 agent-user
```
Verify it is NOT an admin and has no debug right:
```
net localgroup Administrators            # agent-user must NOT appear
whoami /priv                              # (run AS agent-user) — no SeDebugPrivilege
```

### 1. Create the vault-svc user  (admin shell)
```
net user vault-svc <a-strong-password> /add
# keep it out of interactive logon groups; it only runs the service
```

### 2. Lay down Vault storage, ACL'd away from astmo  (admin)
```
mkdir C:\vault\data
copy C:\code\kevin\vault\vault.hcl C:\vault\vault.hcl
icacls C:\vault /inheritance:r /grant vault-svc:(OI)(CI)F Administrators:(OI)(CI)F
# note: astmo gets NOTHING — that's the point
```

### 3. Run Vault as a service under vault-svc  (admin)
```
sc create Vault binPath= "\"C:\...\vault.exe\" server -config=C:\vault\vault.hcl" obj= ".\vault-svc" password= "<pw>" start= auto
sc start Vault
```

### 4. Initialize — THIS is your offline master key  (you, once)
```
set VAULT_ADDR=http://127.0.0.1:8200
vault operator init -key-shares=5 -key-threshold=3
```
- **Write the 5 unseal keys + root token to OFFLINE media. Do NOT leave them on
  this box.** These are your master key.
- Unseal: `vault operator unseal` ×3 (feed 3 of the offline shares).

### 5. Migrate the age vault → Vault  (you, with the root token transiently)
```
set VAULT_TOKEN=<root token from step 4>
python C:\code\kevin\vault\migrate_and_policies.py
```
- This writes every secret into Vault KV + one read-only policy each, and emits
  `service_policy_map.json` for the broker.
- Verify, then delete the age files:
  `del C:\code\secrets\keystore\vault.age C:\code\secrets\keystore\identity.age-key`
- **Revoke the root token** when done: `vault token revoke <root>`. No standing
  root on the box.

### 6. Give the broker a long-lived, least-priv Vault token  (you)
Create an AppRole or a periodic token that can ONLY read `kv/data/cloud/*`, and
put it + an owner secret in the broker service's environment (as vault-svc, not
astmo). Then run `vault_broker.py` as vault-svc (a second service, or under the
voice service if that runs as vault-svc).

### 7. Wire cc.bat to mint a nonce per launch
`cc --secrets <spec>` must, as an **owner-side step**, call the broker's
`mint_nonce` with `BROKER_OWNER_SECRET` (held by vault-svc/you, NOT in astmo
space) and the resolved policy set, then pass the returned nonce to the session
so the MCP shim registers with it. **This nonce-mint is the one piece that must
run with the owner secret** — design it so an agent can't read that secret
(e.g. cc.bat shells out to a tiny vault-svc helper for the mint).

## What's already done (no action needed)
- `vault.hcl`, `vault_broker.py`, `vault_client.py`, `migrate_and_policies.py`
- MCP shim (`keystore_mcp.py`) prefers the Vault broker, falls back to age.
- Verified: broker fetches from real Vault, stages nameless file, value never in
  response; granted service served, non-granted denied; no-nonce/fake-nonce/
  self-mint/unregistered all denied.

## The honest open question (step 7)
The nonce-mint needs the owner secret. If cc.bat (running as astmo) holds it, an
agent reads it. The clean answer is a tiny **vault-svc-owned mint helper** that
cc.bat calls — so the owner secret never enters astmo space. That helper is the
last piece to build; flag me when you've done steps 1–6 and I'll build it to
match how you set up vault-svc.
