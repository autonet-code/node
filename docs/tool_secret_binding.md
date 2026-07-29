# Tool–Secret Binding

**Status:** BUILT 2026-07-29, uncommitted. All four changes landed.
Tests: `tests/atn/test_tool_secret_binding.py` (23),
`tests/atn/test_tool_guard_hosts.py` (8+1 skip). Regression: 142 tests
across the tool/secrets surfaces green.

**Design note — shape B.** The spec below originally proposed
intersecting the tool's declared secrets against the agent's allowance
*at the broker's authorization step*. That is not expressible: the
broker session is keyed by kernel PID and authorizes from
`_SESSIONS[pid]["policies"]`, and **at the moment the broker authorizes a
request, no tool is executing** — the caller is the worker, i.e. the
agent's own LLM loop. There is no tool to intersect against.

What was built instead inverts who holds the secret. The daemon mints a
**separate, short-lived broker session bound to the tool subprocess's
PID**, scoped to `L_agent ∩ tool.capabilities.secrets`, and the tool
reads its own secret through it. The agent never receives a handle. That
is what makes the control structural rather than detective: the agent
cannot exfiltrate what it never holds.

This works because `register` already accepts a target PID distinct from
the caller (`vault_broker.py:381`), and both `mint_nonce` and `register`
are owner-gated on `BROKER_OWNER_SECRET`, which only the daemon holds.
**No broker change was needed.**

## The problem

Secret exposure in ATN is not a tool-*use* problem. It is a tool-
*authoring* problem, and the two subsystems that should meet there never
do.

Three facts, each verified in-tree:

1. **The value never reaches the model.** `secret_request_secret`
   returns `{var_name, path}` and nothing else
   (`atn/runtime/worker_loop.py:544-548`) — an opaque handle to a
   nameless staged file (`_stage_path`, `keystore.py:267-274`).
2. **The allowance clamp is real.** A child agent's secret set is
   `requested ∩ parent`, computed daemon-side and bound to a kernel PID
   (`atn/runtime/worker_host.py:206`, `:514-518`). A compromised parent
   can only narrow, never widen.
3. **Authored tools run bare.** `_exec_spec` (`atn/tool_store.py:554`):

   ```python
   if record.origin != "adopted":
       return [sys.executable, str(script)], None, None
   ```

   `env=None` → the subprocess inherits the **full daemon environment**.
   No `tool_guard.py`, no sandbox cwd, no `ATN_TOOL_POLICY`. Only
   *adopted* (foreign) code is contained.

So the `toolsmith` capability bundle (`atn/orchestrator/tools.py:3628`)
is a general code-execution primitive, and there is no join between a
tool and the secrets its caller holds. Grepping `tool_store.py` (1974
lines) for `secret|allowance|vault|keystore` yields **one** hit — a
comment at `:549`.

Consequence: an agent whose allowance is `{GITHUB_TOKEN, STRIPE_KEY}`
runs *every* tool it can call inside a process that may request *both*.
A tool cannot be scoped to a subset of its caller's secrets.

### What "scope" is today

There is no scope field on a secret. `authorized_hosts` is the nearest
thing and it is **inert**: written by `secrets_put`
(`atn/ws_server.py:3167-3173`) and `secrets_set_hosts` (`:3216`), read
by exactly two display paths (`:3055`, `:3194`). The keystore docstring
(`keystore.py:180-182`) describes a forwarding proxy that requires it —
that proxy does not exist in this repo. Nothing consults it at stage
time or at egress.

### Why the "two calls, same secret" heuristic can't run today

The audit log records one `staged` row per broker hand-off
(`atn/runtime/secret_audit.py:82-111`), with no call site and no use
count. A worker that reads the staged file 10,000 times produces
**exactly one row**. The heuristic needs data the log does not carry —
hence change (4).

---

## (1) Declare secrets on the tool; bind them to the tool's process ✅

**This is the whole fix.** New manifest field `capabilities.secrets:
[str]` — vault service names the tool needs. The binding is:

```
L_tool = L_agent  ∩  tool.capabilities.secrets
```

Same monotone-clamp shape as the fractal parent/child intersection, so
the invariant extends from `child ⊆ parent` to `tool ⊆ caller`.

Rules, all enforced:

- **Absent field → deny-all**, not allow-all. Every existing tool
  therefore binds nothing, which is why the default is safe.
- **Dotted names stripped** at declaration AND rejected by manifest
  validation — the daemon plane (`app.*`, `agent-key.*`) is unreachable
  from a manifest by any route, including a hand-written one.
- **Declaration is not a grant.** A manifest declaring `{STRIPE_KEY}`
  gets nothing if the caller's allowance lacks it.
- **The secret is gated; the execution is not.** If the binding fails
  (tripwire down, broker refuses), the tool still runs — it just gets no
  secret and fails on the missing credential in its own code. A security
  control that silently prevents execution is an availability bug.

Implementation:

| Piece | Where |
|---|---|
| Declaration parse + clamp | `atn/runtime/tool_secrets.py` (new) |
| Session lifecycle (mint/bind/release) | `ToolSecretSession`, same file |
| Exec wiring | `tool_store.py` `_call_pinned`, `_call_pinned_interactive` |
| Manifest schema | `nodes/common/world_model_substrate/tool_manifest.py` |
| Tool-side read API | `atn/tool_secret_api.py` (new) |

The tool reads its value via `atn.tool_secret_api.get_secret(...)`,
self-contained so it works under the scrubbed adopted-tool env. Names
(not values) are advertised to the subprocess as `ATN_TOOL_SECRETS`;
the advertisement is not authorization — the broker authorizes from the
PID session.

**Composition composes correctly.** Nested dep calls already run under
the *original* caller's authority, so each tool in a composite gets its
own clamp against the same `L_agent`. A composite cannot lend its
secrets to a dependency.

**Teardown is unconditional.** `release()` runs in a `finally` covering
the crash, non-zero-exit, and timeout paths — verified by test, because
a leaked session leaves staged plaintext readable after the tool exits.

## (2) Make `authorized_hosts` real ✅

`tool_guard.py` already intercepted the socket audit event — it just
discarded `args`. It now checks the destination against `policy.hosts`,
populated from the union of `authorized_hosts` for the secrets the tool
was bound. That field was written and displayed but **consulted by
nothing**; it is now load-bearing on the one path where it matters.

Arg shapes, verified live:

- `socket.getaddrinfo` → `args[0]` is the host
- `socket.connect` → `args[1][0]` is the host/IP as written

Matching is on **label boundaries**: `cdn.example.com` matches a listed
`example.com`; `notexample.com` and `example.com.evil.net` do not.

**Backward compatible:** empty/absent `hosts` = unrestricted (prior
behavior). `net: False` still wins — the allowlist only ever narrows.

**Honest limits, tested and documented in the module docstring.** A
raw-IP connect is checked against the same list, so DNS-side indirection
does not slip past — but a tool that resolves an allowed name and
connects to whatever address it gets, a proxy/CDN fronting other
origins, DNS-record exfiltration, and native-code bypass of the audit
hook are all still open. This raises the cost of a stealth call in
ordinary Python. **It is not an egress firewall** — consistent with the
module's own "HONESTY, NOT HERMETICS".

## (3) Run authored tools through the guard ✅

Authored tools now run under `tool_guard.py` by default, with one
deliberate concession: **env inheritance is preserved**. Scrubbing it
would break tools that read daemon config vars, for a benefit the audit
hook already delivers. Tightening env is a separate step, gated on a
survey of what actually reads what.

The policy default is **permissive when the manifest declares nothing**
(`net`/`fs`/`spawn` all true) — the guard's value here is the
destination check on a secret-bound tool, not a sudden deny-by-default
flip that would break every existing tool. An author who *does* declare
capabilities gets them enforced.

Escape hatch: `ATN_AUTHORED_TOOLS_BARE=1` restores the old bare exec.

Verified: five realistic authored-tool shapes (module-level stdin,
`__main__`-guarded, env reads, temp writes, stdlib net imports) all run
unchanged under the guard — `runpy` passes `run_name="__main__"`, so the
guarded form is transparent.

**Survey note:** this machine has 18 registered tools, all
connector-backed (attested), **zero pinned** — so the local breakage
surface for this change was empty. That is one machine, not a
population; the escape hatch exists because other daemons may differ.

## (4) Record the call site ✅

`SecretAuditLog.record` now takes a `tool` digest (16 hex chars),
carried on the `granted`/`revoked` rows of a tool-bound session and on
the `SECRET_ACCESS` event. The log can now answer **"which tool consumed
this secret"** — the call-site dimension the PID-only rows could not
provide.

Note what this does and does not give you. It records the *binding*,
not each read: a tool that reads its staged file repeatedly still
produces one pair of rows. The original "two calls with the same secret
is suspicious" heuristic is a **backstop** anyway — with (1) in place the
interesting case is structurally blocked, and the tripwire it would feed
only matches literal un-transformed values (base64 it and it is
invisible). It is a transcript-leak detector, not an exfiltration
detector (`docs/secrets.md:122-125`).

**Not done, still open:** `secret_access.jsonl` has no rotation bound
and `tail()` does a full `read_text()`. Unchanged by this work.

---

## Deploy note — the manifest schema is a network contract

`validate_manifest` gained `secrets` in its allowed-capability set. Tool
manifests are content-addressed and gossiped, so this is a **one-way
compatibility break**:

- A **new** daemon accepts old manifests unchanged (the field is
  optional, and absent means deny-all). Backward compatible.
- An **old** daemon rejects any published manifest carrying
  `capabilities.secrets` — "unknown capability 'secrets'". It will not
  adopt or vet that tool.

This does not fork the epoch close (the close reads usage receipts and
reviews, not capability keys), so it is **not a flag day** in the sense
the v4.1 deploy was. But a tool published with `secrets` before the
network has upgraded is invisible to un-upgraded peers. Practical
sequencing: ship the daemon build first, publish secret-declaring tools
after.

This one is a user call, not mine — the alternative is to accept the
field silently on old builds (a validator relaxation, which would need
to ship *first* and still leaves them unable to enforce the binding).

## What this does NOT change

The accepted residual is unchanged: a determined model can `cat` the
staged file for a service it is *already authorized for*
(`docs/secrets.md:38-41`). The point of (1) is to shrink "already
authorized for" from *the agent's whole allowance* to *this tool's
declared need* — and for a tool-bound secret, the agent is not
authorized at all.

Still true, still by design:

- `tool_guard.py`'s audit hook is bypassable by native code (ctypes,
  extension modules). The OS-level isolated runner is the wall.
- The security monitor matches literal values only.
- Vet status remains a review, not a gate.

---

## What already works — do not regress it

The tripwire is load-bearing, not observational, in two places:

- **Bind-time:** monitor down ⇒ pending grant popped and discarded, no
  broker session minted (`atn/runtime/__init__.py:511-530`).
- **Per-request:** value-push fails ⇒ broker unlinks the staged file and
  denies (`vault_broker.py:472-479`).

Plus: dotted-name exclusion (`vault_setup.py:52` + `broker_client.py:
273-285`), broker PID authentication (`vault_broker.py:444-448`), and
owner-only adoption approval (`tool_store.py:1270`).

## Accepted, by design

A determined model can `cat` the staged file for a service it is
*already authorized for* (`docs/secrets.md:38-41`). None of the above
changes that, and none tries to. The point of (1) is to shrink
"already authorized for" from *the agent's whole allowance* to *this
tool's declared need*.
