# Secrets and agent security

Status: BUILT, beta. The vault (`atn/_vendor/kevin/keystore.py`), the
allowance algebra (`atn/runtime/worker_host.py`), the names-only audit
trail (`atn/runtime/secret_audit.py`), and the leak tripwire
(`atn/_vendor/kevin/secret_alarm.py`) all ship in the daemon. The Secrets
tab in the app is the owner-facing surface over them.

This is the doc for the **owner**: the human who holds the wallet and
decides what their agents may reach.

## The honest threat model

Read this part first, because everything else only makes sense against it.

**An agent IS a model, and the model drives shell and Python tools. Any
value the agent's process can reach, the model can read.** The system does
not pretend otherwise, and neither should you.

So the vault is not a box that keeps a secret away from an agent you have
already authorized to use it. It is a mechanism for deciding *which* agents
can reach *which* secrets, keeping values off disk in plaintext, and keeping
them out of the conversation transcript.

What the vault **does** guarantee:

1. **Encrypted at rest.** Secrets live in an age-encrypted `vault.age`. A
   casual `cat` reveals ciphertext. Only the broker's identity key decrypts
   it: protect that key file and you protect the vault.
2. **Bounded blast radius.** An agent may only request services on its own
   allowlist. It cannot reach a secret belonging to another service, and a
   child agent can never hold more than its parent.
3. **No accidental context leak.** A requested secret is written to a *file*;
   the agent receives a path and a variable name, never the value in its
   response. The value does not pass through the model's context on the way
   in.

What it does **not** guarantee, stated plainly: a determined model can read
the exported file for a service it is *already authorized for*. That is
accepted, by design. The tripwire below is the backstop for values that
nevertheless end up in a transcript.

The practical consequence: **the allowance is the security boundary.** Grant
narrowly. A secret an agent never receives is the only secret it cannot
misuse.

## Adding a secret

The "Add secret" dialog takes three things.

**Service name**: the identifier, e.g. `GITHUB_TOKEN`. It doubles as a token
in allowance specs, so it may not contain spaces or commas, and `all` /
`none` are reserved keywords. Flat names are the grantable plane; dotted
names (`app.*`, `agent-key.*`) are daemon-internal and are never grantable to
an agent.

**Value**: write-only. Once stored, the app never displays it again. To
change it, rotate it; there is no "reveal".

**Authorized hosts** *(optional)*: the egress hostnames this secret is
allowed to travel to, e.g. `api.github.com`. Stored in a plaintext sidecar
(`secret_meta.json`) because hostnames are not secret and the daemon and UI
need them without decrypting the vault. Leave the field empty and the daemon
auto-prefills a suggestion.

An empty host list means *not host-bound*: value-in-hand mode. Host binding
is what lets the daemon confine a secret to the destination it was issued
for, rather than trusting that whatever holds it only talks to the right
place.

## The allowance: who gets what

Each agent carries a `secrets_allowance` spec: a comma-separated list of
service names, or the keywords `all` or `none`. Everything is **fail-closed**:
an unset, blank, or unparseable allowance resolves to deny-all, as does an
unknown agent or an unavailable keystore. The daemon-wide default root
allowance is `none`.

The rule that makes nesting safe:

> A parent's requested spec is only an **upper-bound wish**. The daemon
> computes the child's grant as the **intersection** of that wish with the
> parent's own allowance, using daemon-held state keyed by the authoritative
> pipe-bound parent id.

The parent never holds its own allowance as a token, never mints the child's
grant, and never sees the result. So a compromised or simply buggy parent can
only ever request something **narrower than or equal to** what it already
has. It can never widen it, never escalate sideways into a sibling's secrets. This
monotone clamp is the entire reason the fractal agent hierarchy can be
trusted with credentials at all, and it is why the computation must live in
the daemon rather than in the worker.

Grants are bound to a worker **process id** and torn down when that worker is
reaped.

## The audit trail

Every event touching the vault lands in an append-only `secret_access.jsonl`,
and is emitted live as a `SECRET_ACCESS` event the app displays:

| Action | Meaning |
|--------|---------|
| `granted` | a broker session was minted for a worker: the agent *can* now request these services |
| `staged` | the broker actually decrypted and staged one secret for the worker |
| `revoked` | the worker was reaped and its session torn down |
| `added` / `rotated` / `deleted` | owner mutations from the Secrets tab |

`granted` is capability, `staged` is use. The distinction matters when you
are reading the log after an incident: an agent may hold a grant it never
exercised.

**Rows carry secret NAMES only: never a value, and never a staged path.**
The path is omitted deliberately: it is a live scan token, and writing it
down would hand any reader of the log the tripwire's needle.

## The tripwire

The security monitor scans agent output for the literal text of known secret
values. A hit records a names-only alarm and raises it in the app. Values
shorter than 6 characters are skipped as a false-positive guard.

This is a **backstop, not a barrier**: it tells you a value has already
leaked into a transcript, after the fact. Treat an alarm as a rotation
trigger: rotate the secret, then find out which agent surfaced it and why.

## Operational guidance

- **Grant `none` and widen deliberately.** The default is deny-all; keep it
  that way for any agent that has no concrete need.
- **Prefer host-bound secrets.** Set authorized hosts whenever you know the
  destination.
- **Rotate on any alarm**, without waiting to determine severity.
- **Watch the `granted` rows, not just `staged`.** A grant that keeps
  appearing for an agent that never stages anything is a sign the allowance
  is wider than the job requires.
- **Protect `identity.age-key`.** Whoever can read it can decrypt the vault;
  file permissions are what stand between the two.

## Where things live

All under `~/.atn/keystore/` (override with `KEYSTORE_DIR`), outside any
repository:

| Path | Contents |
|------|----------|
| `identity.age-key` | the broker's age private key, which decrypts the vault |
| `vault.age` | age-encrypted `{ "SERVICE": "value" }` |
| `secret_meta.json` | plaintext sidecar: authorized egress hosts |
| `bundles.json` | named service bundles usable in allowance specs |
| `exports/` | decrypted single-secret files staged for workers |
| `sessions/` | per-session records of what was staged, for teardown and sweep |

## See also

- `docs/unified_agent_design.md`: the agent model the allowance hangs off
- `docs/cross_platform_isolation_design.md`: worker process isolation, the
  mechanism that makes a per-process grant meaningful
- `docs/agentic_loop.md`: how a worker actually requests and uses a staged
  secret
