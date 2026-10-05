# Integration listener (third WS socket)

Status: built (milestone M1, integration-listener half). Code: `atn/ws_server.py`
(`INTEGRATION_*`, `_handle_integration_*`), `atn/ws_auth.py`
(`IntegrationTokenStore`, `TokenBucket`), `atn/integration_token_cli.py`.
Tests: `tests/test_integration_listener.py`.

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
 "allowed": ["adopt_tool", "attest_tools", "list_tools", "probe_tools",
             "publish_tool", "status", "use_tool"]}
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
| `list_tools` | `include_operations?` | forced to `category=registered` |
| `probe_tools` | `query`, `k?` (1..25) | library search; local fallback scoped to the bound agent |
| `use_tool` | `name`, `arguments` | **registered tools only**: resolves the name, refuses core/connector/pipeline tools and any connector-backed manifest (or composite reaching one); runs through `tool_store.call` and the `tool_guard` containment path, author-lineage checked against the bound agent |
| `attest_tools` | `judgments[]`, `context` | post-use reviews with optional per-axis scores, recorded as the bound agent |
| `publish_tool` | `digest` | author-only (the bound agent's own tools), and only if the agent's bundle grant includes `publish_tool` |
| `adopt_tool` | `digest`, `reason` | **proposes** adoption; approval (`approve_adoption`) stays owner-only |

`register_tool` is **denied** (`owner_only`). An authored (non-adopted) tool
runs with the daemon's full environment and working directory, so
`register_tool` followed by `use_tool` would give the token holder code
execution on the daemon host: the keystore, the vault, and the
pre-authenticated owner socket on `:7700`. Guests use tools; authoring stays
with agents running inside the daemon.

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
| Running guest code | Registered tools execute only via `tool_store.call` and `tool_guard` (same containment as any agent tool). Declared `capabilities` follow the existing agent rules. |
| Self-approving adoption | `adopt_tool` only proposes; `approve_adoption` denied. |
| Flooding / review spam | Per-token rate limit and connection cap. Reviews are weighted by the agent's REP share at close, so a zero-REP guest moves nothing. |
| Proxy confusion | The listener is separate from the local socket; nothing on it is trusted by peer address. |

Known residuals (later milestones):

- `attest_tools` usage events carry no per-event signature (`tool_store.attest_usage`);
  attribution rests on the daemon. (M7)
- Fee-bearing tools accrue to an off-chain ledger only; no on-chain charge
  happens on this path today.
- An agent with no declared bundles has no grant restriction, so the allowlist
  is the only narrowing. Give integration agents an explicit `tools:` list.
- No TLS on this socket; keep it on loopback or a private network.
