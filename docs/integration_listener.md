# Integration listener (third WS socket)

Status: built (milestone M1, integration-listener half; guest tool authoring
added after). Code: `atn/ws_server.py` (`INTEGRATION_*`, `_handle_integration_*`),
`atn/ws_auth.py` (`IntegrationTokenStore`, `TokenBucket`, `local_peer_denied`),
`atn/integration_token_cli.py`, `atn/guest_sandbox.py`, `atn/guest_launcher.py`.
Tests: `tests/test_integration_listener.py`, `tests/atn/test_guest_tools.py`,
`tests/atn/test_guest_launcher_posix.py` (Linux, root).

## Why a third listener

The daemon already runs two sockets:

| Listener | Default | Who | Auth |
|---|---|---|---|
| local | `localhost:7700` | the owner on this box | none (privileged by construction) |
| remote | off (`remote_ws_host`) | owner wallet or an agent's own key | signed challenge |
| **integration** | off; `127.0.0.1:7710` | a guest harness (for example the Odysseus sidecar) | **bearer token bound to one agent** |

A guest harness must act as one agent without holding that agent's private key
(agent self-auth needs it) or the owner wallet. The integration listener gives
it a revocable, non-money credential scoped to exactly one agent, and a strict
allowlist of message types.

## Enabling it

Config (`autonet:` section of the daemon config):

```yaml
autonet:
  integration_ws_enabled: true
  integration_ws_host: 127.0.0.1   # Docker sidecar: 0.0.0.0 on an internal network only
  integration_ws_port: 7710
  integration_rate_per_sec: 5.0    # per token, shared across its connections
  integration_burst: 20
```

Env overrides (win over the config): `ATN_INTEGRATION_WS=1`,
`ATN_INTEGRATION_WS_HOST`, `ATN_INTEGRATION_WS_PORT`. A bind failure is logged
and does not stop the daemon. Never publish port 7710 to a host or public
network: the token is the only gate.

## Tokens

```
atn integration-token create --agent <agent_id> [--label <label>] [--force]
atn integration-token list [--json]
atn integration-token revoke <token_id | label>
```

All take `--data-dir <path>` (default: the configured daemon data dir).

- `create` prints the plaintext token once on stdout (`atn_it_<43 chars>`);
  provenance goes to stderr. Only `sha256(token)` is stored, in
  `<data_dir>/integration_tokens.json` (atomic write, mode 0600 on POSIX).
- A token is bound to exactly one agent id. Binding to an owner sentinel
  (`""`, `user`, `orchestrator`) is refused at mint, at load and at dispatch.
  `create` also refuses an agent id not found in the agents dir unless
  `--force`; the daemon refuses the token until that agent exists.
- `revoke` takes effect without a restart: the daemon re-reads the file when it
  changes, re-checks the token on every message, and closes the connection
  (code 4401) after replying `token_revoked`. A revoked token gets 401 on
  reconnect.
- `token_id` is the first 12 hex chars of the hash, safe to log and share.

## Handshake

The token goes in the WebSocket upgrade request:

```
GET / HTTP/1.1
Upgrade: websocket
Authorization: Bearer atn_it_...
```

The server checks it before upgrading:

- missing, malformed, unknown or revoked token, or bound agent not in the fleet: HTTP **401**
- token already has `integration_max_conns` (4) live connections: HTTP **429**

On success the first frame is:

```json
{"type": "integration_ready", "agent_id": "guest", "token_id": "6f068b184446",
 "allowed": ["adopt_tool", "attest_tools", "find_services", "list_tools",
             "network_status", "probe_tools", "publish_tool", "register_tool",
             "status", "tool_reviews", "use_tool"]}
```

There is no snapshot and no event stream on this listener.

## Wire protocol

Request (one JSON object per frame, max 2 MiB):

```json
{"type": "<message type>", "msg_id": "<client correlation id>", ...arguments}
```

Response:

```json
{"msg_id": "...", "ok": true,  "result": {...}}
{"msg_id": "...", "ok": false, "error": "human text", "code": "<code>"}
```

Error codes: `bad_request`, `token_revoked`, `rate_limited`, `owner_only`,
`not_allowed`, `wrong_agent`, `agent_unavailable`, `unknown_tool`,
`tool_error` (the tool itself returned an error).

### Allowlist

| type | arguments | notes |
|---|---|---|
| `status` | none | `{agent_id, agent_name, token_id, label, allowed, rate_limit}` |
| `list_tools` | `include_operations?` | forced to `category=registered`; each row adds `mine` (authored here by the bound agent), `origin`, `published` |
| `probe_tools` | `query`, `k?` (1..25) | library search; local fallback scoped to the bound agent |
| `use_tool` | `name`, `arguments` | **registered tools only**: resolves the name, refuses core/connector/pipeline tools and any connector-backed manifest (or composite reaching one); runs through `tool_store.call` and the `tool_guard` containment path, author-lineage checked against the bound agent |
| `attest_tools` | `judgments[]`, `context` | post-use reviews with optional per-axis scores, recorded as the bound agent |
| `register_tool` | `name`, `description`, `input_schema`, `code`, `capabilities?` (`net`/`fs`/`spawn` booleans only), `dependencies?` | **guest authoring**: pinned code only, stored with `origin="integration"` and always executed on the guest containment path (below). Connector/provider/endpoint backing, `env`, `secrets`, `provides`, and dependencies reaching a connector are refused. Needs the `toolsmith` bundle grant |
| `publish_tool` | `digest` | author-only (the bound agent's own tools, guest-authored ones included), and only if the agent's bundle grant includes `publish_tool` (`publishing` bundle) |
| `adopt_tool` | `digest`, `reason` | **proposes** adoption; approval (`approve_adoption`) stays owner-only |
| `find_services` | `query?`, `limit?` | public read: the on-chain services market (same rows as the agent `find_services` tool) |
| `tool_reviews` | `digest`, `limit?` | public read: drifted position, vetting, usage; local review rows only for a tool the bound agent can see (else `unknown_tool`); a digest the node doesn't hold gets close state only |
| `network_status` | none | public read: `{autonet, chain{chain_id, label, testnet, substrate_address}, p2p{running, peers, gossip}, epoch, epochs_closed, service_market}`; never the RPC URL (it can embed a provider key) |

The three public reads (`INTEGRATION_PUBLIC_READS`) are answered inline, not
through `execute_tool`: they read network state any peer can see, act as
nobody and move nothing, so they need no bundle grant. In particular a guest
does not need the `services` bundle (which also carries `pay_for_service`)
to browse the market. Paying for or requesting a service stays denied here;
the owner does it over the remote listener (below).

Before the node joins the network (no agent registered on chain yet and
`autonet.enabled` unset) it is local-only and reads no chain. `find_services`
and `network_status` then answer `ok: true` with
`{joined: false, status: "not_joined", message: "Not joined: register an
agent to join the network."}` (`network_status` keeps its other keys with
`chain: null`). Once joined, both payloads carry `joined: true`.

### Guest tool authoring

An ordinary authored tool runs with the daemon's environment and working
directory, so a guest that could register one and then `use_tool` it would
have code execution on the daemon host. Guest registrations are therefore a
separate kind of record:

- The listener sets `_origin="integration"` after stripping every `_` key
  from the request, so a guest cannot pick its own origin. The record keeps
  `origin="integration"` (persisted; survives reload).
- Every execution of such a record, by any caller (the guest, an in-daemon
  agent it was granted to, the owner, a vet replay), goes through
  `atn/guest_sandbox.py`, never the authored or adopted path. Composites keep
  the line protocol; each dependency runs on its own containment path and
  AS THE GUEST AUTHOR, not the caller: when the owner or another agent uses
  a guest composite, a dependency runs only if the guest agent may call it
  itself, with the guest agent's (empty) tool-secret binding. A guest tool
  may not depend on a tool that declares `secrets` or `env`.
- No tool secrets are bound, and `capabilities.env` is not accepted, so no
  daemon variable is ever passed through.
- Names are checked on the node: snake_case, 3-64 chars, not starting with
  `atn_`, `reg_`, `pipeline_`, `tool_`, `connector_` or `mcp_`, and not a
  name another author's tool or a core tool already uses (that would make
  lookup by name ambiguous for its owner).
- Publishing stamps the signed manifest with `authored_via: "integration"`,
  so a node that proposes adopting it shows the owner that a token-holding
  guest wrote the code, not the node's own agent. A guest record from
  before the stamp must be registered again before it can be published.

Containment, strongest first:

1. **Separate uid** (the Docker image, Linux). The image's entrypoint is a
   small root launcher (`atn/guest_launcher.py`) that starts the daemon as
   `atn` (10001) with one end of a private socketpair, and runs each guest
   tool as its own uid from a pool (`--guest-uid 10002 --guest-uid-count
   16`: uids 10002-10017, gid = uid, one per live run; when all are busy a
   new run is refused). No filesystem socket exists. A run gets a fresh 0700
   sandbox `/tmp/atn-guest-<run>`, an environment built from scratch by the
   launcher (nothing copied from the daemon), rlimits (CPU, address space
   512 MiB, file size 64 MiB, 256 fds, 64 processes for the uid), and a
   wall-clock timeout (`ATN_GUEST_TOOL_TIMEOUT_S`, default 30 s). On exit,
   timeout, kill or the daemon dropping the run, the launcher runs a reaper
   as that uid that does `kill(-1, SIGKILL)` until no process of the uid is
   left (no environment marker is trusted), then deletes every file the uid
   owns at the top of `/tmp` and `/dev/shm`. A uid is handed to a new run
   only after such a reap comes back clean. So runs cannot read each
   other's sandbox or `/proc` entries, signal each other, or share a
   process budget, and nothing outlives its run. A pool uid cannot read
   `/data/atn` (0700), `/home/atn` (0700) or the daemon's
   `/proc/<pid>/environ`, and cannot signal the daemon. `spawn` may be
   granted here (declared).
   The launcher needs `CAP_SETUID` and `CAP_SETGID` (compose:
   `cap_add: [SETUID, SETGID]`). It also sets `ATN_GUEST_REQUIRE_UID=1`:
   if the launcher is missing, guest tools are **refused**, not run weaker.
   Started without those caps it exits rather than run the daemon as root;
   started as `atn` (`--user atn`) the daemon runs directly with guest tools
   refused.
2. **Same-uid fallback** (dev boxes, Windows, pip installs: any daemon not
   started by the launcher). **Off by default**: without the launcher, guest
   tools are refused unless the operator sets `ATN_GUEST_ALLOW_SAME_UID=1`,
   and the boot log then warns. On this path the guest code runs as the
   daemon user and the only barrier is `tool_guard`, which native code
   (ctypes) bypasses: an integration-token holder can then read the data
   dir, keystore and `integration_tokens.json`. When opted in: fresh
   sandbox in the system temp dir (refused if that is inside the data dir),
   environment built from scratch, POSIX rlimits, its own session or process
   group, a kill of the group, the session and every descendant, `spawn`
   (including `os.fork`) forced off, and `tool_guard`'s `deny_paths` (data
   dir, `KEYSTORE_DIR`, `~/.atn`, `/proc`) even when `fs` is declared. Run
   PIDs are tracked for the owner-socket gate.

Both run the code under `tool_guard` with a deny-by-default policy from the
declared `net` / `fs` / `spawn`, plus `deny_loopback` (no connect, sendto or
resolve toward loopback, unspecified or link-local addresses, `localhost`,
or AF_UNIX paths). **These are declarations checked by a Python audit hook,
not walls**: native code (ctypes, an extension module) skips the hook, so a
guest tool that declares nothing can still open sockets, read any file its
uid can read and fork. The walls under uid isolation are the OS ones: the
separate uid (no `/data/atn`, no daemon env, no signals to the daemon or
other runs), the rlimits, and the owner socket's peer-credential refusal.

**The owner socket.** The local listener (`:7700`) pre-auths every
connection as the owner, so it now checks the peer before doing that
(`ws_auth.local_peer_denied`): a client socket owned by a guest pool uid
(`/proc/net/tcp`, uid isolation) or by a live tracked guest PID or its
descendant (`psutil`, fallback) is closed with 4403 and gets no snapshot.
The client socket is found by the full 4-tuple (its local address and port
equal the connection's peer, its remote address and port equal the
listener's); rows on the server port are skipped and more than one match
counts as unresolved. While uid isolation is configured, an unresolvable
peer is refused. With no guest uid configured and no guest run live, the
check is a no-op.

Each allowlisted call is dispatched as `execute_tool(type, args, caller_id=<bound agent>)`,
so the existing agent gates still apply on top: the H2 agent-callable check and
the agent's declared bundle grant (`resolve_tool_grant`). To narrow a guest
further, give its agent a restricted `tools:` bundle list.

### Clamp rules

- `caller_id` is always the bound agent. Nothing in the request can change it.
- If the request carries any identity key (`caller_id`, `agent_id`, `target`,
  `parent_id`, `id`, `author`, `author_id`, `caller`) with a value other than
  the bound agent, the request is refused with `wrong_agent` (not silently
  rewritten).
- Keys starting with `_` (for example a forged `_caller_id`) are stripped.
- Before dispatch the server asserts the bound id is a real agent and
  `is_owner_caller(id)` is false. `is_owner_caller` treats `None`, `""`,
  `"user"` (`OWNER_ID`) and `"orchestrator"` (`_LEGACY_ROOT_ID`) as the owner;
  none of these can be bound (a test asserts the two sentinel lists agree).

### Deny-list

`INTEGRATION_DENIED_MESSAGES` names every owner-only surface explicitly (an
import-time assert keeps it disjoint from the allowlist). Anything else not on
the allowlist is also refused (`not_allowed`). Denied groups:

- key custody and vault: `KEY_LOCAL_ONLY_MESSAGES` (`export_agent_key`,
  `register_agent_on_chain`, `set_owner_wallet`, `secrets_put/delete/import/config`, ...)
  and all `secrets_*`
- adoption and tool-grant decisions: `approve_adoption`, `reject_adoption`,
  `list_adoption_proposals`, `grant_tool`, `revoke_tool`, `set_tool_enabled`,
  `set_tool_published`, `vet_tool`
- ownership, wallets, chain: `rotate_owner`, `autonet_wallet_*`,
  `autonet_set_chain`, `autonet_start/stop`, registration signing and checks
- money: `pay_for_service`, `request_service`, `invoke_service`,
  `register_service`, `retire_service`, `service_request`
- budgets and sponsor inference: `set_budget`, `set_credit_budget`,
  `get_budget`, `update_sponsor_budget`, `create_sponsor_agent`, ...
- fleet shape: `create_agent`, `remove_agent`, `update_agent`, `clone_agent`,
  `merge_clone`, activate/deactivate/kill, `trigger_run`, `post_message`,
  `send_agent_message`, delegate and task approval messages
- providers, connectors, OAuth, daemon control, profile, model setters
- whole-fleet reads: `snapshot`, `get_snapshot`, `list_agents`

## Owner actions go through the remote listener

Money and owner decisions (approve adoption, register on chain, owner
binding, pay for / request a service) never ride an integration token. A
guest UI (the Odysseus Economy panel) sends them from the browser over the
existing remote listener, signed in with the owner wallet (the same
`auth_challenge` / `auth_response` handshake atn_web uses). For a container,
env overrides configure it:

| env | effect |
|---|---|
| `ATN_REMOTE_WS_HOST` | bind host for the remote listener (empty = disabled) |
| `ATN_REMOTE_WS_PORT` | its port (default `local_ws_port + 1`, 7701) |
| `ATN_OWNER_WALLET` | fills `owner_wallet` only when the config has none; never replaces a configured owner; ignored unless it is a 0x address |

Publish that port on the host's loopback only (`127.0.0.1:7701:7701`). The
remote listener has no TLS; behind HTTPS put it behind a wss proxy.

## Rate limiting

Token bucket per token (shared across that token's connections), default 5
requests/s with burst 20, checked before any other work. Over the limit:
`rate_limited`. Plus at most 4 concurrent connections per token (HTTP 429).

## Threat model

Assets: the owner's keys and vault, the owner wallet and ATN, fleet shape and
budgets, other agents' identities, the owner's connector credentials, and the
integrity of tool reviews.

| Threat | Mitigation |
|---|---|
| Token theft (leaked env, logs) | Token is not a money key: it reaches no wallet, transfer, signing, key export, budget or secrets surface. Blast radius is one agent's tool-substrate actions. Revoke with one CLI call; effective on the next message. Bind to loopback or an internal Docker network only. |
| Token store theft | Only sha256 hashes are stored; 256-bit random tokens make offline guessing infeasible. |
| Owner escalation via sentinel ids | Sentinels refused at mint, load and dispatch; `is_owner_caller` asserted false before every dispatch. |
| Acting as another agent | `caller_id` forced; identity keys naming others refused; `_`-prefixed keys stripped; publish is author-checked. |
| Escalation through `use_tool` (core tools such as `create_agent`, `pay_for_service`) | `use_tool` restricted to REGISTERED tools. |
| Reaching the owner's MCP connectors | Connector-backed manifests refused at register and use, including composites that depend on one. |
| Running guest code | Guest-registered tools carry `origin="integration"` and always run on the guest path: a per-run uid under the image's launcher (no read of `/data/atn`, the daemon env, or other runs; no signals to the daemon or other runs; every process of the uid killed after the run), rlimits, wall-clock kill, no secrets, no env pass-through. Without the launcher, refused unless `ATN_GUEST_ALLOW_SAME_UID=1`. `net`/`fs`/`spawn` declarations are checked by `tool_guard` only (a Python tripwire native code bypasses), not by the OS. |
| A guest composite using its caller's authority | Dependencies run as the guest author, only if that agent may call them; deps declaring `secrets` or `env` are refused at registration. |
| Misattributed provenance after publish | Published guest manifests carry a signed `authored_via: "integration"`, shown on adopters' approval cards. |
| A guest tool dialing the owner socket | `tool_guard` `deny_loopback` (Python only), and the local listener refuses a peer whose socket (matched by 4-tuple) belongs to a guest uid or a tracked guest PID (close 4403). |
| Self-approving adoption | `adopt_tool` only proposes; `approve_adoption` denied. |
| Flooding / review spam | Per-token rate limit and connection cap. Reviews are weighted by the agent's REP share at close, so a zero-REP guest moves nothing. |
| Proxy confusion | The listener is separate from the local socket; nothing on it is trusted by peer address. |

Known residuals (later milestones):

- Guest tool egress is not firewalled, for EVERY guest tool, declared or
  not, in both modes. Undeclared `net` is enforced only by the audit hook,
  so any guest tool can reach whatever the node can (the internet, the
  internal Docker network, other containers on it) through native code or
  a spawned child; `tool_guard` also only checks destinations as written.
  An OS egress filter needs `CAP_NET_ADMIN` or a separate network
  namespace (not done).
- Likewise undeclared `fs`: under uid isolation a guest tool can read any
  world-readable file in the container (not `/data/atn` or `/home/atn`).
- On the same-uid fallback (opt-in, `ATN_GUEST_ALLOW_SAME_UID=1`), the wall
  is `tool_guard` (an audit hook) plus the PID-based socket gate; native code
  (ctypes) can get past the hook and read what the daemon user can.
  Production nodes should run the image (uid isolation,
  `ATN_GUEST_REQUIRE_UID=1`). Windows has no rlimits on this path.
- On macOS `psutil` cannot map sockets to PIDs without root, so the owner
  socket refuses local connections while a fallback guest run is live.

- `attest_tools` usage events carry no per-event signature (`tool_store.attest_usage`);
  attribution rests on the daemon. (M7)
- Fee-bearing tools accrue to an off-chain ledger only; no on-chain charge
  happens on this path today.
- An agent with no declared bundles has no grant restriction, so the allowlist
  is the only narrowing. Give integration agents an explicit `tools:` list.
- No TLS on this socket; keep it on loopback or a private network.
