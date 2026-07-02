# Cross-Platform Isolation + Vault — Port Plan

**Status:** scoped, not started. The agent process-isolation path and the secret
vault are **Windows-only** today. In-process (single-process, all-agents) is the
only fully-working execution path off Windows, and it's the default everywhere
(gated by `ATN_WORKER_ISOLATION`, off by default).

This is the sprint to make isolation + vault OS-agnostic.

## Coupling inventory (what's anchored to Windows)

**Already OS-agnostic — no work:**
- Daemon↔worker RPC channel (`atn/runtime/worker_rpc.py`) — `multiprocessing.Connection`
  (`send_bytes`/`recv_bytes`); a socketpair on POSIX. The isolation *control plane*
  already crosses platforms.
- Secret store (`atn/_vendor/kevin/keystore.py`) — age/`pyrage` + stdlib file I/O.
- PID-reuse defense in the supervisor (`agent_supervisor._probe_identity`) —
  `psutil.create_time()`/`cmdline()`, cross-platform. (Caveat: `psutil` is a
  `[network]` extra, not core.)

**Windows-only — needs porting (the two hard parts):**
1. **Kill-tree containment** (`worker_manager.py`) — Win32 Job Objects
   (kill-on-close tree reap). POSIX today has only `start_new_session` (a
   grandchild can `setsid`-escape).
2. **Broker IPC + PID authentication** (`broker_client.py`, `vault_broker.py`) —
   Windows named pipes + `GetNamedPipeClientProcessId` (kernel-authenticated peer
   PID). Also `_identity(pid)` via `GetProcessTimes` (create-time reuse guard) and
   the value-push listener.

## Phases

### Phase 0 — foundations (small)
- Portable `_identity(pid)` in the broker: Linux `/proc/<pid>/stat` starttime,
  `psutil` elsewhere, replacing `GetProcessTimes`. Keeps the create-time PID-reuse
  guard.
- `psutil` → core deps (or a `/proc` fallback so isolation works off-Windows
  without pulling it).

### Phase 1 — POSIX process containment (`worker_manager.py`)
- Replace "Job-Object-or-nothing" with a `ProcessTree` abstraction. Windows keeps
  the Job Object; POSIX gets new-session/pgid (have it) + `PR_SET_PDEATHSIG`
  (Linux — orphaned grandchildren self-kill) + SIGTERM→SIGKILL to the pgid.
- Reap the bridge node-grandchild on POSIX (the `setsid`-escape case flagged in
  the P5 review).

### Phase 2 — POSIX broker IPC + PID auth (the core, ~half the sprint)
- Add an `AF_UNIX` transport alongside the named-pipe one, in both
  `broker_client.py` (client) and `vault_broker.py` (server).
- Peer PID+uid via **`SO_PEERCRED`** (Linux) / **`LOCAL_PEERCRED`/`getpeereid`**
  (macOS) — the exact kernel-authenticated analog of `GetNamedPipeClientProcessId`
  (and it yields uid too).
- Port the value-push listener (`_ValuePushListener`) to `AF_UNIX`. Socket dir
  owned by the broker uid, `0700`.

### Phase 3 — POSIX hardening + setup
- Broker runs as a **separate uid** (analog of the Windows `vault-svc` account);
  age-key `0600` owned by that uid so a same-daemon-uid agent can't read it.
- ptrace hardening: Yama `ptrace_scope` / `PR_SET_DUMPABLE(0)` — analog of
  "deny SeDebugPrivilege."
- `atn-vault-setup` POSIX branch: create the broker user, set perms, emit a
  systemd unit + a POSIX RUNBOOK.

### Phase 4 — fail-closed fallback + tests
- Make the in-process **fallback** (used when `ensure_worker` fails) fail-closed
  for a *granted* execution — today it silently downgrades out of isolation.
- POSIX integration tests for tree-kill and PID-auth (both deferred in the Windows
  build).

## The decision to settle before Phase 1
Tree-kill on POSIX is **weaker than a Job Object**. Process-group kill +
`PDEATHSIG` covers the common case, but a determined grandchild can
double-fork/`setsid` and escape. The only bulletproof Linux analog is **cgroup v2
`cgroup.kill`**, which needs cgroup delegation (running the daemon under a systemd
scope) — a real deployment dependency. **macOS has no equivalent** — best-effort
pgid kill only.

Resulting strength story: **full on Windows and Linux (with the cgroup/systemd
path), best-effort on macOS.** The cgroup-vs-plain-pgid choice changes whether the
Linux daemon has a systemd dependency — decide it first.

## Verification caveat
This work was scoped on a Windows box. The POSIX paths (`AF_UNIX` + `SO_PEERCRED`,
`PDEATHSIG`, cgroups) **cannot be verified on Windows** — the sprint needs a Linux
(and ideally macOS) test environment, not just fresh context.
