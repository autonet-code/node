# Cross-Platform Isolation + Vault: Port Plan

**Status:** IMPLEMENTED + Linux-verified (2026-07-02). All 4 phases landed and
run green end-to-end on the Ubuntu EC2 (`tests/test_posix_isolation.py`, 6/6).
macOS remains port-by-symmetry (`LOCAL_PEERCRED`/`getpeereid`), code-review-only.
In-process (single-process, all-agents) is still the default everywhere (gated by
`ATN_WORKER_ISOLATION`, off by default); isolation now works on POSIX when on.

This was the sprint to make isolation + vault OS-agnostic.

## What landed (per phase)
- **P0**: portable `_identity(pid)`: Linux `/proc/<pid>/stat` starttime (field
  22), Windows `GetProcessTimes`, psutil fallback for other POSIX; `psutil`
  moved to core deps. `pyrage` already core.
- **P1**: POSIX containment. Primary is a direct kill of the *tracked* worker
  PID (`_posix_signal_pid`), with a process-group sweep (`_posix_signal_group`,
  only when we own the pgid) as the descendant backstop. No cgroups/Job Object
  on POSIX. `setsid`-detach residual confirmed + accepted.
- **P2**: POSIX broker IPC. `AF_UNIX` transport in `broker_client.py` (client +
  value-push listener) and `vault_broker.py` (server), peer PID via
  `SO_PEERCRED` (== `GetNamedPipeClientProcessId`). Value-push socket uses a
  short runtime dir (`_short_unix_sock_path`) to stay under the sun_path limit.
- **P3**: hardening + setup. Broker calls `PR_SET_DUMPABLE(0)`;
  `atn-vault-setup` POSIX branch tightens keystore 0700 / age-key 0600, emits a
  systemd unit template + `RUNBOOK_POSIX.md` (separate `vault-svc` uid, Yama
  `ptrace_scope`).
- **P4**: fail-closed fallback. A *granted* execution whose worker can't spawn
  now finalizes FAILED instead of silently downgrading to un-isolated in-process
  (`_has_pending_grant` / `_fail_closed_no_isolation`). Committed POSIX
  integration tests formalize the EC2 verification.

## Coupling inventory (what's anchored to Windows)

**Already OS-agnostic, no work:**
- Daemon↔worker RPC channel (`atn/runtime/worker_rpc.py`): `multiprocessing.Connection`
  (`send_bytes`/`recv_bytes`); a socketpair on POSIX. The isolation *control plane*
  already crosses platforms.
- Secret store (`atn/_vendor/kevin/keystore.py`): age/`pyrage` + stdlib file I/O.
- PID-reuse defense in the supervisor (`agent_supervisor._probe_identity`):
  `psutil.create_time()`/`cmdline()`, cross-platform. (Caveat: `psutil` ships in
  core.)

**Windows-only, needs porting (the two hard parts):**
1. **Kill-tree containment** (`worker_manager.py`): Win32 Job Objects
   (kill-on-close tree reap). POSIX today has only `start_new_session` (a
   grandchild can `setsid`-escape).
2. **Broker IPC + PID authentication** (`broker_client.py`, `vault_broker.py`):
   Windows named pipes + `GetNamedPipeClientProcessId` (kernel-authenticated peer
   PID). Also `_identity(pid)` via `GetProcessTimes` (create-time reuse guard) and
   the value-push listener.

## Phases

### Phase 0: foundations (small)
- Portable `_identity(pid)` in the broker: Linux `/proc/<pid>/stat` starttime,
  `psutil` elsewhere, replacing `GetProcessTimes`. Keeps the create-time PID-reuse
  guard.
- `psutil` → core deps (or a `/proc` fallback so isolation works off-Windows
  without pulling it).

### Phase 1: POSIX process containment (`worker_manager.py`)
- **Approach: track every spawned PID in the supervisor and kill each directly**
  (`os.kill`), not by process group. We already record our own tree (`_children`);
  the bridge node-grandchild just needs adding to that tracking. It is portable,
  needing neither Job Objects nor cgroups. On Windows we keep the Job Object as a
  belt-and-suspenders backstop, but the per-PID kill is the primary, portable path.
- The only thing per-PID kill can't reach is a descendant we **never saw** (an
  agent that shells out and the grandchild double-forks/`setsid` to detach before
  we record its PID). That residual is **equal on every OS** (Job Objects catch it
  on Windows; nothing does on plain POSIX) and is accepted: it's inherent to
  tracking-based containment, not a Linux-specific gap.

### Phase 2: POSIX broker IPC + PID auth (the core, ~half the sprint)
- Add an `AF_UNIX` transport alongside the named-pipe one, in both
  `broker_client.py` (client) and `vault_broker.py` (server).
- Peer PID+uid via **`SO_PEERCRED`** (Linux) / **`LOCAL_PEERCRED`/`getpeereid`**
  (macOS): the exact kernel-authenticated analog of `GetNamedPipeClientProcessId`
  (and it yields uid too).
- Port the value-push listener (`_ValuePushListener`) to `AF_UNIX`. Socket dir
  owned by the broker uid, `0700`.

### Phase 3: POSIX hardening + setup
- Broker runs as a **separate uid** (analog of the Windows `vault-svc` account);
  age-key `0600` owned by that uid so a same-daemon-uid agent can't read it.
- ptrace hardening: Yama `ptrace_scope` / `PR_SET_DUMPABLE(0)`, the analog of
  "deny SeDebugPrivilege."
- `atn-vault-setup` POSIX branch: create the broker user, set perms, emit a
  systemd unit + a POSIX RUNBOOK.

### Phase 4: fail-closed fallback + tests
- Make the in-process **fallback** (used when `ensure_worker` fails) fail-closed
  for a *granted* execution: today it silently downgrades out of isolation.
- POSIX integration tests for tree-kill and PID-auth (both deferred in the Windows
  build).

## Tree-kill: resolved (no open decision)
Kill by **tracked PID**, not process group: the supervisor already knows every
process it spawned, and killing a known PID is trivial and portable on every OS.
So there is **no** cgroup/systemd dependency and **no** Windows/Linux/macOS
strength split for our own tree. The one residual (a descendant that detaches
before we record it) is identical across platforms and accepted. This removes
the only architecture decision that was blocking Phase 1.

## Verification
Scoped on a Windows box; the POSIX paths (`AF_UNIX` + `SO_PEERCRED`, tracked-PID
kill) can't run on Windows. **Linux verification target: the Ubuntu EC2** (ssh via
ZeroTier; see the `ec2_smoke_setup` memory). P1/P2/P4 get real end-to-end runs
there. **macOS** (`LOCAL_PEERCRED`/`getpeereid`) stays code-review-only unless a
Mac is available: verify Linux, port-by-symmetry for macOS, flag it as untested.
