# ATN local runtime and agents: code-derived reference

All paths are relative to `D:\autonet_codeonly`. Every claim comes from reading the code. No documentation was consulted. The `docs/*.md` files and `atn/external_agents.md` that the code refers to are not in this copy.

---

## 1. Processes and entry points

### 1.1 Installed commands

| Command | Target | Source |
|---|---|---|
| `atn` | `atn.cli:main` | `pyproject.toml:137-138` |
| `atn-vault-setup` | `atn.vault_setup:main` | `pyproject.toml:139` |
| `python -m atn` | `atn/__main__.py` calls `atn.cli.main()` | `atn/__main__.py:1-4` |

### 1.2 `atn` dispatch (`atn/cli.py:985-1036`)

`main()` checks `sys.argv[1]`. Four offline subcommands run and exit without booting the daemon:

| Subcommand | Behavior | Source |
|---|---|---|
| `atn guide` | Prints `atn/external_agents.md` to stdout. The file is declared as package data (`pyproject.toml:148`) but is absent from this copy. | `atn/cli.py:987-992` |
| `atn send '<json>' [--url ws://...]` | Opens one WebSocket to the daemon (default `ws://127.0.0.1:7700`), adds a `msg_id` if missing, waits up to 600 s for the reply with the same `msg_id`, and prints it. Exit codes: 0 when `ok`, 1 when not ok, 2 for usage or bad JSON, 3 when no daemon is reachable. | `atn/cli.py:945-982` |
| `atn relay ...` | Calls `nodes.relay.main`, a networking-only node. Flags: `--port` (env `AUTONET_P2P_PORT`, 4001), `--host` (`AUTONET_P2P_HOST`, 0.0.0.0), `--announce`, `--bootstrap`, `--jurisdiction` (repeatable, env `AUTONET_RELAY_JURISDICTIONS`), `--data-dir` (`AUTONET_RELAY_DIR`, `~/.autonet-relay`), `--transport tcp|ws` (`AUTONET_P2P_TRANSPORT`), `--no-upnp`, `--advertise-interval`. | `atn/cli.py:995-1000`, `nodes/relay.py:64-90` |
| `atn integration-token create --agent <id> [--label L] [--force]`, `revoke <id-or-label>`, `list [--json]`, global `--data-dir` | Edits `<data_dir>/integration_tokens.json` offline. `create` refuses owner sentinels and, unless `--force`, refuses agents not found on disk. The plaintext token is printed once on stdout. | `atn/integration_token_cli.py:50-112` |

Any other argv boots the daemon:
1. `apply_staged_update_if_any()` runs first (`atn/cli.py:1009-1014`).
2. Then `asyncio.run(run_cli())` (`atn/cli.py:1016`).
3. After it returns: if `restart` was requested, the process re-execs with `os.execv` (`atn/cli.py:1020-1026`). Otherwise it exits with `os._exit(0)` (`atn/cli.py:1036`).

The only daemon flag is `--headless` (`atn/cli.py:780`). Headless mode also switches on automatically when stdin is not a TTY.

### 1.3 Other process entry points

| Entry | What it is | Source |
|---|---|---|
| `python -m atn.ws_server` | Standalone daemon. Local listener only. Stdin accepts `r`/`restart` (full re-init) and `q`/`quit`. | `atn/ws_server.py:6266-6430` |
| `python -m atn.mcp_server` | stdio MCP proxy to a running daemon (section 4). | `atn/mcp_server.py:222-228` |
| `python atn/substrate_mcp.py` | Second stdio MCP server with tools `substrate_capsule`, `substrate_query` and `substrate_status`. Writes capsule JSONL under `ATN_DATA_DIR/agents/<id>/conversations`. | `atn/substrate_mcp.py:1-130, 279-330`, `substrate-mcp.json` |
| `python -m atn.guest_launcher` | Root launcher with `--daemon-uid`, `--daemon-gid`, `--guest-uid`, `--guest-uid-count`, `--guest-gid` (ignored), `--guest-root`, then `-- cmd`. Internal modes: `--stage <fd>` and `--reap <run_id>`. | `atn/guest_launcher.py:419-488` |
| `python -m atn.agent_worker` | Per-agent worker process spawned by the daemon over an inherited pipe. Its environment is scrubbed of any variable whose name contains KEY, SECRET, TOKEN, AUTH and similar markers. | `atn/agent_worker.py:1-40`, `atn/runtime/worker_manager.py:192-213, 286-295` |
| `atn-vault-setup` | Creates the age identity and the service-to-policy map, tightens permissions, and prints the steps for running the broker under a separate uid. | `atn/vault_setup.py:72-98` |

### 1.4 Daemon boot order (`run_cli`, `atn/cli.py:613-811`)

1. `load_config()` (`:617`).
2. Singleton lock: `LockManager` on `<data_dir>/daemon.lock`. The process exits if another daemon holds it (`:620-633`, `atn/lock_manager.py:24`).
3. Warm-import `numpy` on the main thread (`:643-646`).
4. Build `EventBus`, then `Runtime(...)` (`:648-649`). The constructor (`atn/runtime/__init__.py:70-369`) does the following:
   - resolves the boot jurisdiction key;
   - creates the `InputArbiter`;
   - creates the stores: inbox, output, execution log, credentials, profile, credit budget, sponsor bindings, metering;
   - creates `ConnectorManager` from the bundled specs plus config connectors;
   - creates the group chat and delegate registry;
   - creates the step executors, `AgentRegistry`, `ProviderManager` (with `setup_providers`) and `ExecutionEngine`;
   - creates the broker client, logging SECURITY errors if `BROKER_OWNER_SECRET` is not scrubbable or `VAULT_TOKEN` is present (`:230-250`);
   - creates and starts `SecurityMonitor`, then the secret audit log and the broker value-push;
   - creates `AgentSupervisor`, `ExecutionControl` and `SessionManager`;
   - creates the scheduler, `AutonetBridge`, `TraceLogger`, `ToolStore` (`data_dir/tools`), `ServiceStore` and `ToolRegistry` (`:370-470`).
5. Subscribe the console printer to all events. Print the banner, including provider key hints (`:652-666`).
6. `await runtime.start()` (`atn/runtime/__init__.py:730-884`) does the following:
   - starts the connector idle reaper and the channel settler;
   - recovers crashed executions and reaps orphan workers;
   - starts the worker monitor if `worker_isolation.enabled`;
   - cleans up delegates and runs provider auto-detection;
   - starts the bridge-updater and model-catalog loops (unless `ATN_DISABLE_PROVIDER_AUTODETECT=1`);
   - warms the embedder and starts the scheduler;
   - starts the robot service and robot rail, voice and chat if each is enabled;
   - starts autonet only if `autonet.enabled`;
   - loads the constitution, attaches the trace logger, starts the Firestore agent-directory publisher if `public_ws_endpoint` is set, and emits `RUNTIME_STARTED`.
7. `_load_agents()`: loads YAML from `agents_dir`, runs the legacy routing pin migration, unregisters removed agents, registers each agent with `legacy=True`, heals dangling parents, and auto-activates agents that have `schedule` or `heartbeat` (`atn/cli.py:464-540`).
8. `seed_default_fleet()` seeds agent `kevin` once per install (`atn/cli.py:675-681`, `atn/fleet_seed.py:140-210`).
9. `maybe_autostart_autonet_for_registered_agents()` starts autonet if any agent has `registered_on_chain` (`atn/cli.py:686-691`, `atn/runtime/__init__.py:886-911`).
10. `WebSocketBridge.start()`: local listener, then remote, then integration. Then `runtime.register_surface(ws_bridge)`. On `OSError` with an unconfigured port, the CLI tries `_try_reclaim_port` once (`atn/cli.py:695-754`).
11. If `auto_update.enabled` and this is not a dev environment, start `_update_poll_loop` (`atn/cli.py:760-778`).
12. Headless mode waits on `runtime._shutdown_event`. Interactive mode prints help and runs `_input_loop` (`atn/cli.py:780-795`).
13. Shutdown: cancel the update poll, stop the bridge, `runtime.stop()`, release the lock (`atn/cli.py:798-811`).

### 1.5 Interactive console commands (`atn/cli.py:137-433`)

| Command | Effect |
|---|---|
| `status` | Table of agents and running executions |
| `run <id>` | `runtime.trigger_run` |
| `msg <id> <text>` | `send_agent_message(..., surface=None)`. A trusted internal caller, never mic-gated (`:259-281`). |
| `kill <exec_id or agent_id>` | Tries the execution first, then the agent |
| `killall` | Kills all executions |
| `activate <id>`, `deactivate <id>` | Changes agent status |
| `agents` | Lists agents, including worker PIDs from `supervisor._workers` |
| `reload` | Calls `_load_agents` again (see section 7, finding 1) |
| `reconcile` | `runtime.reconcile_chain_registrations()` |
| `history <id>`, `logs`, `clear <id>`, `clearall` | Execution-log operations |
| `usage` | Shows `_rate_limits` from active bridge providers |
| `tools` | Lists pending adoption proposals |
| `approve <digest> [--core-replacement]`, `reject <digest>` | Resolves an adoption (`:401-421`) |
| `restart` | Sets `_restart_requested` and exits the loop |
| `help`, `quit`/`q`/`exit` | |

### 1.6 Auto-update

- Polling only stages a newer wheel. It runs every `max(60, check_interval_secs)` seconds and emits `UPDATE_STATUS` when the state changes (`atn/cli.py:844-942`).
- Staged updates are applied at the next boot, before the heavy imports (`atn/update_boot.py:145-232`).
- `apply_staged_update_if_any` reads `<ATN_DATA_DIR or ~/.atn>/staged_update/pending.json` (`atn/update_boot.py:33-41, 170`), runs `pip install --upgrade`, sets `ATN_UPDATE_APPLIED=1` and re-execs.
- Both paths skip a dev or editable install (`dev_environment_reason`).

---

## 2. Network listeners

### 2.1 Summary

| Listener | Bind | Port | Enabled when | Auth | Source |
|---|---|---|---|---|---|
| Local (privileged) | `"localhost"` (hardcoded in the CLI) | `autonet.local_ws_port`, else 7700 | Always | None. Every connection is pre-authed as the full-fleet owner, unless the peer is a guest tool process. | `atn/cli.py:701-716`, `atn/ws_server.py:58, 467-473, 627-664` |
| Remote | `ATN_REMOTE_WS_HOST` or `autonet.remote_ws_host` | `ATN_REMOTE_WS_PORT` or `autonet.remote_ws_port` (config default 7701); the bridge falls back to `port+1` only when given 0 | Host is non-empty | Wallet signature challenge: owner, or agent self-auth | `atn/cli.py:585-610`, `atn/ws_server.py:409, 484-496`, `atn/config.py:211-212` |
| Integration | `ATN_INTEGRATION_WS_HOST`, else `autonet.integration_ws_host`, else 127.0.0.1 | `ATN_INTEGRATION_WS_PORT`, else `autonet.integration_ws_port`, else 7710 | `ATN_INTEGRATION_WS` truthy, or `autonet.integration_ws_enabled` | `Authorization: Bearer atn_it_...` token bound to one agent | `atn/cli.py:564-582`, `atn/ws_server.py:503-527` |

On both owner listeners, each connection gets an outbound event queue of 2000. When the queue is full, events for that client are dropped and counted (`atn/ws_server.py:682-683, 6225-6237`).

### 2.2 Local listener

- **Peer check.** `ws_auth.local_peer_denied` runs before any session is created. If guest uids are configured (`ATN_GUEST_UID`), Linux resolves the client socket's uid from `/proc/net/tcp{,6}` by the full 4-tuple and refuses guest uids. An unresolvable uid is refused (fail closed). While same-uid guest runs are live, the client PID (via psutil) is refused if it is a tracked guest PID or a descendant of one. A refusal closes the socket with code 4403 (`atn/ws_auth.py:187-240`, `atn/ws_server.py:635-652`).
- **Session state.** `local=True, authed=True, owner=True, scope_ids=None` (`atn/ws_server.py:658-663`). A full snapshot is sent immediately (`:690`).
- **Scope of trust.** Only guest tools are excluded. Any other local process gets owner rights, including agent shell tools, authored tools and the MCP proxy.

### 2.3 Remote listener: challenge and response

1. On connect the server sends `{"type":"auth_challenge", nonce, challenge, conn_id}` (`atn/ws_server.py:752-770`).
2. The challenge text is built by `build_challenge_text` (`atn/ws_auth.py:272-301`). It binds `daemon_id` (persisted in `<data_dir>/daemon_id`, `:251-269`), `chain_id`, `owner_wallet`, `conn_id`, the nonce and the issue time.
3. The client replies with `auth_response {signature, root?}`. `_handle_auth_response` (`atn/ws_server.py:795-875`) checks the following:
   - at most 5 failures per connection (`:793`);
   - challenge TTL of 120 s (`atn/ws_auth.py:52`);
   - the challenge text is re-derived from server-held state and the nonce is consumed;
   - `recover_signer` uses `eth_account` and fails closed if it is missing (`atn/ws_auth.py:304-322`).
4. If the signer equals `owner_wallet`, the session is the owner. It may name a `root` (agent id or a unique 0x) to scope itself (`:842-849, 877-897`).
5. Otherwise the signer must be a known agent address (`registry.get_agent_id_by_address`). The session becomes that agent, scoped to its subtree (`:850-862`).
6. On success the server sends the scoped snapshot (`:705-711`).

`auth_status` reports `{authed, owner, root, requires_auth}` (`:1475-1480`).

**Snapshot and event scoping.** A scoped session's snapshot drops `providers`, `budget`, `user`, `connectors` and `autonet` (`atn/ws_server.py:1369-1377`). Events are delivered only to authed sessions. A scoped session receives only events whose origin agent is in its live subtree (`:6215-6224, 6239-6249`).

### 2.4 Message gates on the owner listeners (`_handle_message`, `atn/ws_server.py:1446-4246`)

The gates run in this order:

1. **Key custody (Gate 1).** `KEY_LOCAL_ONLY_MESSAGES` is refused unless `session.local` (`:1466-1470`). The set (`:92-120`) is: `export_agent_key`, `export_operator_key`, `autonet_publish_standards`, `autonet_claim_reward`, `register_agent_on_chain`, `secrets_put`, `secrets_delete`, `secrets_import`, `secrets_config`, `set_owner_wallet`.
2. **Handshake.** `auth_response` and `auth_status` are always allowed (`:1473-1480`).
3. **Payment-authorized stranger.** An unauthenticated `service_request` goes to `_handle_service_request`, which validates the on-chain payment (`:135-137, 1487-1489, 4774-4868`).
4. **Gate 2.** Any other message from an unauthenticated session gets `"unauthenticated"` (`:1492-1493`).
5. **Gate 2b.** A scoped session may reach only the inline handlers in `SCOPED_INLINE_ALLOWED` (`:195-212`). Those in `SCOPED_REQUIRES_AGENT` (`:216-222`) must name an `agent_id` in the subtree. Every other inline type is refused as owner-only (`:1430-1444`). The set of inline types is extracted by regex from `_handle_message`'s own source, up to the `# --- Gate 3` marker (`:1406-1428`).
6. **Inline handlers.** Several re-check `owner and scope_ids is None`: `operator_status`, `register_operator`, `sweep_to_owner`, `submit_delegation` (`:3298-3344`), `owner_binding_status` and `rotate_owner` (`:3823-3833`), secrets (`:2030-2035`), adoption (`:2152-2158`), run opt-ins and bonds (`:2164-2169`), and review batches (`:2278-2283`).
7. **Gate 3 and fall-through.** `_authorize_tool_call` refuses `OWNER_ONLY_TOOLS` (`:174-183`) and any `agent_id`/`target`/`parent_id`/`id` outside the subtree (`:1383-1403`). The call then goes to `execute_tool` with `caller_id = msg.caller_id or ""` for full-fleet sessions, or `root_agent_id` for scoped sessions (`:4223-4246`). A `post_message` from a client is tagged `source="user"` (`:4220-4221`).

**Inline message types**, grouped (line numbers from `atn/ws_server.py`):
- **Session and input:** `snapshot`, `input_request`, `input_status`, `daemon_restart` (`:1501-1543`).
- **OAuth and providers:** `oauth_start`, `oauth_status`, `provider_list`, `provider_configure`, `provider_bridge_update`, `provider_rate_limits`, `provider_refresh_usage`, `provider_remove`, `custom_provider_add`, `custom_provider_remove` (`:1546-1717`).
- **Agents, delegates and context:** `get_delegates`, `delegate_message` (arbiter-gated), `send_agent_message` (arbiter-gated, `force` to take the mic), `agent_processes_*`, `interrupt_delegate`, `session_stats`, `get_delegate_output`, `session_context`, `context_breakdown`, `agent_system_prompt`, `tool_surface`, `clone_agent`, `merge_clone` (`:1720-1956`).
- **Tools:** `list_registered_tools`, `grant_tool`, `revoke_tool`, `tool_earnings`, `owner_claim_status`, `set_tool_published`, the adoption family, the run opt-in and bond family, `list_distros`, `core_replacement_evidence`, `review_batches`, `tool_security_status`, `probe_tools`, `set_tool_enabled` (`:1962-2404`).
- **Services:** `register_service`, `list_services`, `market_services`, `service_topics`, `provider_record`, `retire_service`, `service_request`, `invoke_service` (`:2405-2785`).
- **Voice:** `voice_*` (`:2789-2896`).
- **Onboarding, profile and budget:** `get_onboarding_status`, `skip_onboarding`, `get_profile`, `get_budget`, `set_budget` (`:2897-2950`).
- **Autonet:** `autonet_*`, `claude_sessions`, `claude_session_import`, `robot_*`, `sanctions` (`:2951-3152`).
- **Sponsors and ownership:** sponsor family, `set_owner_wallet`, operator family (`:3153-3344`).
- **Chain registration and substrate reads:** `register_agent_on_chain`, `check_agent_registration`, `confirm_agent_registration`, `rpb_*`, `epoch_*`, `economy_graph`, `tool_reviews`, `fleet_voice`, `owner_binding_status`, `rotate_owner`, `local_models`, `set_local_model`, `update_status` (`:3347-4100`).
- **Agent edits and files:** `reset_agent_conversation`, `set_agent_model`, `remove_agent`, `attachment_upload` (15 MiB cap, stored as `<data_dir>/attachments/<sha16>.<ext>`) (`:4101-4175`).
- **Keys:** `export_operator_key` (local plus full-fleet owner), `export_agent_key` (local) (`:4181-4211`).
- **Secrets:** `secrets_*` (`:4262-4637`).

### 2.5 Integration listener (`atn/ws_server.py:903-1241`, `atn/ws_auth.py:393-645`)

**Tokens**
- Format: `atn_it_` followed by `token_urlsafe(32)`. Only the sha256 is stored. The token id is the first 12 hex characters of the hash (`atn/ws_auth.py:550-567`).
- Store: `<data_dir>/integration_tokens.json`, written atomically with mode 0600. It is re-read when its mtime or size changes. A corrupt file means no token is valid (`:475-546`).
- A token may not be bound to `""` or `"user"` (`:403, 411-419`).

**Upgrade checks** (`process_request`): 401 for a missing, invalid or revoked token or an unknown bound agent. 429 when the token already has 4 live connections (`integration_max_conns`, not configurable) (`atn/ws_server.py:398, 930-942`). The first frame is `integration_ready {agent_id, token_id, allowed}` (`:1052-1057`).

**Per-message checks, in order** (`:1080-1200`):
1. The token is still active (a revoke closes the socket with code 4401).
2. Per-token rate limit, a token bucket of 5/s with burst 20 from `integration_rate_per_sec` / `integration_burst`.
3. `INTEGRATION_DENIED_MESSAGES` (`:287-336`) answers `owner_only`.
4. Anything outside `INTEGRATION_ALLOWED_MESSAGES` answers `not_allowed`. The allowed set is `status`, `list_tools`, `probe_tools`, `use_tool`, `attest_tools`, `publish_tool`, `adopt_tool`, `register_tool`, `find_services`, `tool_reviews`, `network_status` (`:239-255`). An import-time assert keeps the two sets disjoint (`:337-339`).
5. The bound agent must still exist and not be owner-like.
6. Any of `caller_id`, `agent_id`, `target`, `parent_id`, `id`, `author`, `author_id`, `caller` naming another identity is refused (`:345-346`).
7. Protocol keys, identity keys and `_`-prefixed keys are stripped.
8. `list_tools` is forced to `category=registered`.
9. `use_tool` must target a REGISTERED tool that does not reach a connector.
10. `register_tool` is validated by `_integration_register_args` (`:954-1026`):
    - no `connector_id`, `provider`, `entrypoint` or `endpoint`; no `publish`;
    - `code` required; snake_case name of 3-64 characters without reserved prefixes;
    - the name must not be taken by another author or by a non-registered tool;
    - capabilities limited to `net`, `fs`, `spawn` (booleans);
    - dependencies must be visible to the agent and connector-free;
    - then `_origin="integration"` is set.
11. `execute_tool(..., caller_id=<bound agent>)`.

`find_services`, `tool_reviews` and `network_status` are answered inline as public reads (`:1202-1241`). `network_status` carries no RPC URL and no wallet (`:1293-1359`).

---

## 3. The agent model

### 3.1 Definition and modes

- `AgentDefinition` (`atn/models.py:164-330`) has a mode of `PIPELINE` (steps of type `script`, `cognitive`, `message`, `pull`, `collect`) or `COGNITIVE` (`atn/models.py:29-39`).
- Other fields cover: provider (a string or a fallback chain), `cognitive_model`, tools/bundles, budgets, `secrets_allowance`, spawn limits, `parent_id`, heartbeat, `identity`, `wallet`, sponsor fields, `service_provider` (a marketplace binding, validated by `normalize_service_binding`, `:333-367`), `claude_session` and `loadout`.
- Status values are `registered`, `active`, `running`, `stopped`, `error`, `completed`, `budget_paused` (`atn/models.py:42-49`).

### 3.2 Native vs external agents; providers

**Provider ids.** `KNOWN_PROVIDER_IDS` lists `claude_max`, `codex_max`, `anthropic`, `gemini`, `openai`, `deepseek`, `ollama`, `rpb`, `service`, `substrate`, `jev` (`atn/provider_identity.py:21-24`). Construction (`atn/runtime/provider_manager.py:493-575`):
- `claude_max` is `BridgeProvider` (Claude Code bridge).
- `codex_max` is `CodexBridgeProvider`.
- `anthropic` is `AnthropicProvider`.
- `gemini`, `openai`, `deepseek` and custom providers are `OpenAICompatibleProvider`.
- `ollama` is `OllamaProvider`.
- `rpb` is `RPBNetworkProvider`.
- `service`, `substrate` and `jev` use dedicated builders.
- `jev` cannot call tools, so an agent on it must have `tools=[]` (`atn/provider_identity.py:32, 96-108`).

**Routing.** Every cognitive agent must name both a provider and a model. The exceptions are a service binding, the `service` provider and the `external` provider (`atn/provider_identity.py:57-93`).

**External agents.** These have `provider == "external"` (`atn/provider_identity.py:41-46`).
- They are created through `create_agent` with `provider: "external"` (`atn/agent_tools.py:2175-2177`).
- `prompt`, `steps` and `schedule` are refused. Default tools are `unified_tools`, `toolsmith`, `publishing`, `adoption`, `identity`. `parent_id` is the caller, and `notify_parent=False` (`atn/agent_tools.py:2117-2158`).
- The engine never runs them: `trigger_run` returns None (`atn/runtime/execution_engine.py:128-131`).
- `use_tool` results for them carry `review_due` instead of an injected review turn (`atn/agent_tools.py:2872-2891`).

**Imported Claude Code sessions.** `claude_session_import` creates a `claude_max` agent bound to an on-disk session (`atn/ws_server.py:6057-6115`).

### 3.3 Tool-call permissions for agents

- `is_owner_caller` is true for `None`, `""` and `"user"` (`atn/agent_tools.py:47-52`).
- For any other caller, `execute_tool` allows only schema-backed tools (`_AGENT_CALLABLE_TOOLS`, the "H2" gate) and the agent's bundle grant (`atn/agent_tools.py:5387, 5496-5516`). Bundles are listed in `_TOOL_CATEGORIES` (`atn/agent_tools.py:4255-4321`).
- An agent with an empty tool list, `atn_full` or `atn_progressive` is unrestricted (`atn/agent_tools.py:4344-4377`).

### 3.4 Identity, keys and the vault

**Keypair and lineage**
- `generate_keypair` uses `eth_account`. Without it, it falls back to a sha256 pseudo-address (`atn/agent_identity.py:10-26`).
- `public_key` is set to the address (`:68`).
- Lineage is `sha256(parent_lineage : prompt[:256] : public_key)`, rooted at the owner wallet for parentless agents (`:34-46, 49-80`).

**Persistence**
- The identity file is `agents_dir/<id>/identity.json`, written without the key (`atn/runtime/agent_registry.py:171-228`).
- The private key is stored in the vault as `agent-key.<id>` (`:192-209`).
- On load, a legacy `private_key` field is migrated into the vault. A missing key, or a key whose derived address does not match the file, causes regeneration (`:237-335`).
- `unregister_agent` deletes the vault key (`:784-792`).

**Keystore**
- Vendored age keystore at `KEYSTORE_DIR`, default `~/.atn/keystore`. Files: `identity.age-key`, `vault.age`, `bundles.json`, `secret_meta.json`, `exports/`, `sessions/` (`atn/_vendor/kevin/keystore.py:43-56`).
- Dotted names are daemon-plane and never grantable to agents (`atn/vault_setup.py:40-52`). They are `agent-key.*`, `app.*` (CredentialStore, `atn/credentials.py:43`), `owner-key.local` and `operator-key.local`.

### 3.5 Owner wallet, owner binding, operator key, households

**Owner wallet (`autonet.owner_wallet`)**
- Sources: config.yaml; the WS `set_owner_wallet` (local only, persisted, mirrored live, `atn/ws_server.py:3204-3227`); or `ATN_OWNER_WALLET`, which only fills an unset value and only in memory (`atn/cli.py:601-609`).
- It authenticates the remote owner (`atn/ws_server.py:839`).
- It roots lineage for parentless agents (`atn/runtime/agent_registry.py:721-735`).
- It is the consensus author for tools by owners and unregistered agents (`atn/tool_store.py:2734-2758`).
- It is required before `set_my_sponsor` (`atn/ws_server.py:3236-3239`).
- It is the fallback recipient for `sweep_to_owner` (`:5932, 5966-5967`).
- It is the claimant reported by `owner_claim_status` (`:2049-2102`).

**Owner binding.** Registration is always `registerAgentBound(peerId, owner, parent, sig)`, where `sig` is an EIP-712 `OwnerBinding{agent,parent,nonce}` signature over domain `AutonetSubstrate/1` (`atn/on_chain.py:739-860`). The signature comes from one of three places:
- the owner wallet (frontend `owner_sig`);
- `self_owned`, which uses the daemon's local owner key `owner-key.local`, created on first use (`atn/ws_server.py:5775-5786`, `atn/runtime/agent_registry.py:866-886`);
- otherwise the parent agent's key, in which case the child inherits the parent's on-chain owner (`atn/agent_tools.py:4720-4749`).

`rotate_owner` submits `rotateOwner` signed by each agent's own key with the new owner's signature (`atn/ws_server.py:3912-3980`).

**Operator key.** `operator-key.local`, created on first use (`atn/runtime/agent_registry.py:888-908`), unless `autonet.private_key` is configured (`atn/ws_server.py:5811-5815`). It is registered as a bound agent through `register_operator` (`:5994-6046`). It receives the owner's REP delegation through `submit_delegation` (`delegateBySig`, `:5873-5910`), funds gas for sweeps (`:5976-5981`), and becomes the autonet node's signing key (`atn/autonet_service.py:947-955`).

**Households.** There is no local household data structure. The code uses "household" to mean the agents bound to one on-chain owner. `_resolve_household` always returns `""` (`atn/agent_tools.py:3350-3355`).

### 3.6 Registering agents on chain

**WS `register_agent_on_chain`** (local only, `atn/ws_server.py:3347-3397, 5651-5809`):
1. Calls `ensure_network_config()`, which joins the network by resolving the registry.
2. Requires `substrate_address` and `rpc_url`.
3. Requires a daemon-held agent key.
4. Pre-flights the gas balance, returning `needs_funding` if it is zero.
5. Gets the libp2p peer id, starting autonet just in time and polling for about 30 s.
6. Binds through `owner_sig`, `self_owned` or the parent signature.
7. On success, sets `registered_on_chain` and the tx, persists the identity, returns the private key to the local client, and emits `AGENT_REGISTERED`.

**Agent tool `register_on_chain`** (bundle `identity`) uses the same flow but always the parent signature (`atn/agent_tools.py:4752-4811`).

**Other paths**
- `check_agent_registration` syncs the local flag with the chain (`atn/ws_server.py:6144-6190`).
- `confirm_agent_registration` sets the flag without any chain check (`:3414-3426`).
- `reconcile` (CLI) re-verifies all agents (`atn/cli.py:222-247`).

**Joining the network.** `resolve_network_registry` fills unset fields from, in order: `ATN_REGISTRY_URL`, then `ATN_JURISDICTION_REGISTRY` or the saved `<data_dir>/jurisdiction.json`, then the packaged `atn/registry.json`. Fields set explicitly in config.yaml are never overwritten. It sets `network_joined` (`atn/config.py:939-1045`). The jurisdiction can only start if `substrate`, `rep_token`, `tool_bond`, `dao` and `timelock` all resolved, the epoch rules are version `"2"`, and the charter text is valid (`atn/config.py:664-712`, `atn/autonet_service.py:284-322`).

---

## 4. MCP server (`atn/mcp_server.py`)

- It is a stdio MCP server that proxies each tool call over a fresh WebSocket. It never boots a runtime (`:1-33, 85-127`). The call timeout is 300 s.
- **Owner mode** (no token): connects to `ATN_MCP_URL` or `ws://localhost:7700`, which is the local, pre-authed owner listener. It lists every core tool definition (`:154-158`). With `ATN_MCP_AGENT=<id>`, every call except `create_agent`, `get_agent` and `list_agents` gets `caller_id=<id>` (`:59-60, 97-98`).
- **Guest mode** (`ATN_MCP_TOKEN` set): connects to `ATN_MCP_URL` or `ws://127.0.0.1:7710` with the bearer header (`:53-55, 123-127`). It lists only the `allowed` set from `integration_ready`, plus generic schemas for `status`, `network_status` and `tool_reviews` (`:130-169`).
- Both modes add a local `atn_guide` tool that returns `external_agents.md` (`:62, 80-82, 176-189`). They also set server `instructions` that tell the agent to call `atn_guide`, search with `probe_tools`, run with `use_tool`, review with `attest_tools` and publish with a novelty claim (`:63-72`).

---

## 5. Configuration

### 5.1 Loading

- The file path is fixed at `~/.atn/config.yaml`. There is no CLI flag or env override (`atn/config.py:1339-1340`).
- `~/.atn/.env` is loaded first and never overrides variables already set (`:1073-1107`).
- `${VAR}` references are interpolated in the autonet, connectors and providers sections (`:1052-1066`).
- A legacy `orchestrator:` section is rewritten to `defaults:` (`:1283-1331`).
- A missing file gives defaults (`:1352-1355`).

### 5.2 Keys

| Section | Keys (default) | Source |
|---|---|---|
| top | `data_dir` (`~/.atn`), `agents_dir` (`./agents` if it exists, else `<data_dir>/agents`), `connector_idle_timeout_s` (900), `defaults:{provider,model}` (legacy, migration only) | `atn/config.py:34-53, 1368-1391, 1470-1475` |
| `providers.<id>` | `api_key`, `default_model`, `base_url`, `models`, `dollar_limit`, `dollar_period` (none/hourly/daily/weekly/monthly); other keys go to `extra` | `:1494-1518` |
| `connectors.<id>` | `mode` (local/npx/uvx), `package`, `entry`, `command`, `args`, `env`, `env_required` | `:1478-1491` |
| `voice` | `enabled`, `backend`, `ptt_keys`, `mute_key`, `voice_volume`, `tools_volume`, `effects_volume`, `narrate_tools`, `announcements`, `output_device`, `input_device`, `kokoro_model_dir`, `piper_module_dir` | `:1394-1412` |
| `chat` | `enabled`, `platform` (discord), `token_env` (`DISCORD_BOT_TOKEN`), `channel_id`, `bound_agent`, `policy` (open/operator/credit), `operator_ids`, `credit_window_secs`, `credit_default_limit`, `root_label`, `excluded_agents` | `:1414-1428` |
| `autonet` (merged over `rpb`, over `blockchain.{rpc_url,chain_id,contracts.AutonetDAO}`) | `enabled`, `config_path`, `rpc_url`, `chain_id`, `gas_symbol`, `gas_decimals`, `private_key`, `wallet_address`, `owner_wallet`, `local_ws_port`, `remote_ws_host`, `remote_ws_port` (7701), `integration_ws_enabled`, `integration_ws_host`, `integration_ws_port`, `integration_rate_per_sec`, `integration_burst`, `public_ws_endpoint`, `firestore_project`, `dao_address`, `jurisdiction_id`, `rpb_contract_address`, `substrate_address`, `rep_token_address`, `charter_hash`, `charter_uri`, `charter_text`, `epoch_rules`, `bootstrap_peers`, `containment_deny_harm`, `containment_trust_harm`, `containment_trust_mass_bps`, `deploy_block`, `anchor_election_address`, `rep_claim_address`, `dao_registry_address`, `tool_bond_address`, `registry_address`, `service_registry_address`, `payment_channel_address`, `token_address`, `economy_address`, `timelock_address`, `min_alignment_threshold`, `generate_keypairs`, `sponsor_inference`, `sponsor_provider`, `sponsor_model`, `sponsor_address`, `ws_input_policy` | `:1185-1280` |
| `trace_logging` | `enabled`, `trace_dir`, `include_user_data`, `min_turns` | `:1433-1440` |
| `auto_update` | `enabled` (true), `check_interval_secs` (86400), `source` (pypi), `pypi_index_url`, `package_name` | `:402-406, 1443-1452` |
| `worker_isolation` | `enabled` (true), `memory_cap_mb` | `:1119-1134` |
| `secrets` | `default_root_allowance` (`none`) | `:1162-1171` |
| `robot` | `enabled`, `agent_id`, `bundle_dir`, `bag_dir`, `required_packs`, `heartbeat_interval_s` | `:1150-1159` |
| `robot_poll` | `enabled`, `period_s`, `pollers`, `percentile`, `deadline_s`, `commit_gap_s`, `retention_rounds` | `:1137-1147` |

The WS writes back to config.yaml through `save_owner_wallet_to_config`, `save_sponsor_address_to_config`, `save_sponsor_mode_to_config`, `save_secrets_config_to_yaml`, provider save/remove and connector save/remove (`atn/config.py:1531-1824`).

### 5.3 Environment variables read by the runtime

| Variable | Effect | Source |
|---|---|---|
| `ATN_WORKER_ISOLATION` | Overrides `worker_isolation.enabled` | `atn/config.py:1127-1129` |
| `ATN_AUTO_UPDATE` | 0/1 overrides `auto_update.enabled` | `atn/config.py:1174-1182` |
| `ATN_REGISTRY_URL` | Registry document fetched at join | `atn/config.py:575-577` |
| `ATN_JURISDICTION_REGISTRY`, `ATN_JURISDICTION_RPC` | Launched-jurisdiction DAO Registry and its RPC | `atn/config.py:861-904` |
| `ATN_OWNER_WALLET` | Fills an unset owner wallet | `atn/cli.py:602-609` |
| `ATN_REMOTE_WS_HOST`, `ATN_REMOTE_WS_PORT` | Remote listener | `atn/cli.py:594-600` |
| `ATN_INTEGRATION_WS`, `ATN_INTEGRATION_WS_HOST`, `ATN_INTEGRATION_WS_PORT` | Integration listener | `atn/cli.py:568-579` |
| `ATN_MCP_TOKEN`, `ATN_MCP_URL`, `ATN_MCP_AGENT` | MCP proxy | `atn/mcp_server.py:53-59` |
| `ATN_DAEMON_WS`, `ATN_DATA_DIR`, `ATN_SUBSTRATE_AGENT`, `CLAUDE_NAME` | substrate_mcp | `atn/substrate_mcp.py:60-78` |
| `ATN_DATA_DIR` | Also used by update_boot and the guest deny paths | `atn/update_boot.py:33-41`, `atn/guest_sandbox.py:147` |
| `ATN_GUEST_REQUIRE_UID`, `ATN_GUEST_ALLOW_SAME_UID`, `ATN_GUEST_LAUNCHER_FD`, `ATN_GUEST_UID` | Guest containment mode | `atn/guest_sandbox.py:92-118` |
| `ATN_GUEST_TOOL_TIMEOUT_S` (30), `_CPU_S`, `_MEM_MB` (512), `_FSIZE_MB` (64), `_NPROC` (64) | Guest limits | `atn/guest_sandbox.py:79-89` |
| `ATN_AUTHORED_TOOLS_BARE=1` | Run authored tools without the guard | `atn/tool_store.py:1017` |
| `ATN_TOOL_POLICY` | Guard policy, set by the daemon and popped by the guard | `atn/tool_guard.py:85` |
| `ATN_TOOL_SECRETS` | Bound secret names advertised to a tool | `atn/tool_secret_api.py:57` |
| `VAULT_BROKER_SOCK` | Tool-side broker socket (POSIX) | `atn/tool_secret_api.py:90` |
| `ATN_VAULT_BROKER_SOCK`, `ATN_VAULT_POLICY_MAP`, `KEYSTORE_DIR`, `XDG_RUNTIME_DIR` | Daemon broker client and keystore | `atn/runtime/broker_client.py:48-96`, `atn/_vendor/kevin/keystore.py:43` |
| `BROKER_OWNER_SECRET` | Daemon-to-broker owner secret | `atn/runtime/broker_client.py:519` |
| `VAULT_TOKEN` | Only checked for absence (logged as an error if set) | `atn/runtime/__init__.py:246` |
| `ATN_TOOL_BOND_ADDRESS`, `ATN_TOOL_BOND_DEV_OVERRIDE` | ToolBond gate | `atn/runtime/tool_bond.py:36-37` |
| `ATN_CORE_SWAP_DISABLED` | Core-swap kill switch | `atn/runtime/shell_provider.py:75` |
| `ATN_DISABLE_PROVIDER_AUTODETECT`, `ATN_DISABLE_REVIEW_DRIVER` | Disable autodetect, bridge loops and the review driver | `atn/runtime/__init__.py:786`, `atn/review_driver.py:73-75` |
| `ATN_USAGE_MIN_INTERVAL`, `ATN_BRIDGE_IDLE_CEILING`, `ATN_BRIDGE_FIRST_EVENT_TIMEOUT`, `ATN_BRIDGE_KILL_GRACE` | Bridge tuning | `atn/providers/bridge.py:245-270` |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude subscription token | `atn/runtime/provider_manager.py:1717` |
| `CLAUDE_CONFIG_DIR` | Claude session discovery | `atn/claude_sessions.py:26` |
| `NPM_CONFIG_REGISTRY` | Bridge updater | `atn/providers/bridge_updater.py:172` |
| `ATN_ROBOT_SPONSOR_PROBE=assume` | Robot heartbeat | `atn/robot/service.py:107` |
| `AUTONET_CONFIG` | autonet.yaml path when saving | `atn/autonet_service.py:2369` |
| `ELEVENLABS_API_KEY`, `GOOGLE_CLIENT_SECRET`/`GOOGLE_CLIENT_ID`, `DORG_API_BASE`, `HOSTNAME`, `<chat.token_env>` | Voice, OAuth, connectors, lock info, chat | `atn/voice_service.py:389`, `atn/oauth.py:32`, `atn/runtime/__init__.py:1129` |
| `ATN_IS_WORKER` | Set inside workers, read by the worker loop | `atn/agent_worker.py:448`, `atn/runtime/worker_loop.py:539` |
| `ATN_UPDATE_APPLIED` | Re-exec guard for update_boot | `atn/update_boot.py:30` |

Host discovery also reads many third-party variable names (for example `GITHUB_TOKEN`, `OPENAI_API_KEY`, `STRIPE_SECRET_KEY`) to offer them for import into the vault (`atn/runtime/host_discovery/probes.py`).

---

## 6. Tool store and execution security

### 6.1 Records and storage

- `ToolStore(runtime, data_dir/tools)` keeps `registry.jsonl`, `receipts.jsonl`, `attestations.jsonl`, `adoption_proposals.json`, `blobs/`, `code_cache/`, `sandbox/` and `run_optins.json` (`atn/tool_store.py:208-274`, `atn/tool_security.py:127-137`).
- A `ToolRecord` has the manifest digest, `grants`, `enabled`, `published`/`published_in`, `jurisdiction`, `local_author`, `origin` (`authored`/`adopted`/`integration`), `core_replacement_ack` and `last_evidence_digest` (`atn/tool_store.py:89-174`).
- `author` is the consensus 0x identity. `author_id` is the local agent id (`:141-149`).

### 6.2 Registration

**Agent and owner path: `register_tool`** (bundle `toolsmith`, `atn/agent_tools.py:2894-3009`)
- Refuses core, shell and web names and the prefixes `atn_`, `reg_`, `pipeline_`, `tool_`, `connector_`, `mcp_`. Refuses `endpoint`.
- An agent cannot set `publish`; the owner can register and publish in one step.
- The author is the caller (`"user"` for the owner).

**`ToolStore.register`** (`atn/tool_store.py:280-524`)
- Guest-origin rules: code required, no connector/provider/entrypoint, capabilities limited to net/fs/spawn, no dependencies on connector tools or tools that declare secrets or env.
- Normalizes `capabilities.secrets` and `capabilities.provides`.
- Code limit is 512 KiB. Trust class is `pinned` with code, otherwise `attested`.
- Dependencies must exist locally and be enabled.
- The consensus author is resolved through `_consensus_identity`. Registration is idempotent on content.
- The manifest is signed with the author agent's key (`:2760-2784`).

### 6.3 Visibility and calls

- `allowed()`: the owner always; otherwise the tool must be enabled and the caller must be the author, a grantee, or an ancestor of the author (`atn/tool_store.py:740-756`).
- The check is applied at call time by `ToolRegistry._call_registered_tool` (`atn/tool_registry.py:465-488`).
- `grant_tool`/`revoke_tool` are WS-only (`atn/ws_server.py:2002-2024`).

### 6.4 Publishing (`atn/tool_security.py:243-349`)

- Not allowed for adopted tools.
- Needs a novelty claim, the successor-by-same-author rule, a ToolBond stake signed by the author agent (registered on chain, manifest author equal to its address), then `set_published`.
- Owner-authored tools are refused.
- `ATN_TOOL_BOND_DEV_OVERRIDE=1` skips the stake.

### 6.5 Adoption

- `adopt_tool` only proposes. It requires a pinned manifest and refuses sanctioned authors for agent proposers (`atn/tool_store.py:1987-2088`).
- `approve_adoption` (WS or CLI, owner, not available to scoped sessions) does the following (`:2098-2192`):
  - re-fetches the manifest and code;
  - refuses a signature that does not recover to `author` or `author_pubkey`;
  - refuses tools the network has denied;
  - refuses `busted` tools (taken from the stored proposal);
  - requires `core_replacement_ack` for tools whose `provides` claims core names;
  - installs the record as `origin="adopted"`, scoped to the proposer.

### 6.6 Run gate and containment tiers

- `ToolStore.call` is the single dispatch point (`atn/tool_store.py:878-948`).
- Network tools (`origin=="adopted"`, `atn/tool_security.py:57-61`) must be `isBonded` on ToolBond. This fails closed (`:140-157`).
- They then get a tier from `nodes/common/containment.py:38-55`:
  - `denied`: harm at or above `deny_harm` (50) with REP mass at or above `trust_mass_bps` (10 bps). Never runs.
  - `declared`: harm below `trust_harm` (10) with quorum. Gets the manifest capabilities.
  - `strict`: everything else. No capabilities.
- Thresholds come from `autonet.containment_*` (`atn/tool_security.py:183-193`).
- Every network-tool run is traced. Guard refusals are returned in `containment.denied` and raise a security alarm (`atn/tool_store.py:923-947, 1148-1163`).

### 6.7 Execution by origin (`_exec_spec`, `_call_pinned`, `_call_guest`; `atn/tool_store.py:974-1318`)

| Origin | Process | Environment | cwd | Guard policy |
|---|---|---|---|---|
| authored | `python tool_guard.py script` (or bare with `ATN_AUTHORED_TOOLS_BARE=1`) | Full daemon env inherited, plus `ATN_TOOL_SECRETS` and `ATN_TOOL_POLICY` | Inherited | Declared net/fs/spawn; if none declared, all allowed; `hosts` from secrets' authorized hosts (`:1016-1045`) |
| adopted | `python tool_guard.py script` | Scrubbed: PATH, TEMP/TMP, PYTHONIOENCODING, SystemRoot, plus declared `env` names (`:1058-1070`) | `tools/sandbox/<digest16>` | Tier caps; `deny_paths` = keystore plus data dir children outside tools (`:1047-1095, 1128-1146`) |
| integration (guest) | `spawn_guest` through the launcher (uid) or same-uid fallback | Built from scratch; no secrets, no env pass-through (`atn/guest_sandbox.py:210-227`, `atn/guest_launcher.py:94-108`) | Fresh temp sandbox | Declared net/fs; spawn only under uid isolation; `deny_loopback`; `deny_paths` (`atn/guest_sandbox.py:159-170`) |
| connector-backed | MCP call through `ConnectorManager` | n/a | n/a | n/a (`atn/tool_store.py:965-966`) |

- The timeout is 120 s for authored and adopted tools. For guests it is the guest timeout (+2 s under the launcher) (`atn/tool_store.py:55, 1272-1279`).
- Composites with declared dependencies use a line-framed call rail with a maximum depth of 4. Nested calls run under the original caller's authority (`:1320-1360`).

### 6.8 `tool_guard.py`

- An audit hook enforces `net` (socket events), `fs` (path events outside cwd, the Python prefix and the script dir) and `spawn` (subprocess/exec/fork) (`atn/tool_guard.py:265-289`).
- It also enforces the `hosts` allowlist on `getaddrinfo`/`connect`, `deny_loopback` and `deny_paths` (`:118-202`).
- An optional trace records file, net and spawn events plus `denied` records (`:204-302`).
- Its own docstring calls it a tripwire that native code can bypass (`:30-40, 60-66`).

### 6.9 Guest containment (`atn/guest_sandbox.py`, `atn/guest_launcher.py`)

Mode selection (`atn/guest_sandbox.py:121-142`):
- `uid` when `ATN_GUEST_LAUNCHER_FD` is set on POSIX;
- otherwise `refused`, unless `ATN_GUEST_ALLOW_SAME_UID=1` and not `ATN_GUEST_REQUIRE_UID=1`, in which case `same-uid`.

**uid path**
- The daemon sends `run` plus four fds over SCM_RIGHTS, then JSON on the ctl socket (`atn/guest_sandbox.py:358-421`).
- The launcher takes a uid from the pool (verified clean by a reap), stages the run as that uid with rlimits and a wall clock, reaps all processes of that uid before and after, and releases the uid (`atn/guest_launcher.py:251-416`).
- The launcher never runs the daemon as root (`:443-462`).

**same-uid path**
- Same-uid runs get rlimits (no NPROC), their own session or process group, tree kill, and PID tracking used by the local-listener gate (`atn/guest_sandbox.py:177-207, 230-318, 521-551`).
- The boot banner warns when the mode is `same-uid` or `refused` (`atn/cli.py:734-738`).

### 6.10 Tool secrets

- A pinned tool that declares `capabilities.secrets` gets its own broker session bound to the tool's PID. The binding is clamped to the caller's allowance (`atn/tool_store.py:1097-1126, 1221-1252`).
- The tool reads values through `atn.tool_secret_api` (named pipe on Windows, AF_UNIX on POSIX) (`atn/tool_secret_api.py:1-100`).

---

## 7. Inconsistencies and dead code

1. **CLI `reload` destroys existing agents.** For an agent already registered, `_load_agents` calls `runtime.unregister_agent(id, promote_children=False)` (`atn/cli.py:501-505`). That call:
   - rmtrees `data_dir/agents/<id>` (`atn/runtime/__init__.py:1265-1267`);
   - deletes `agents_dir/<id>` including `agent.yaml` and `identity.json` (`:1275-1277`, `atn/loader.py:526-543`);
   - deletes the vault key `agent-key.<id>` (`atn/runtime/agent_registry.py:786-792`).

   The re-registration then generates a new keypair (`:708-738`) and does not re-save `agent.yaml` unless budget keys moved (`atn/runtime/__init__.py:1206-1219`). The on-chain identity is lost and the agent disappears on the next boot. The comment ("Reload, not removal", `atn/cli.py:502-503`) says the opposite was intended.
2. **Owner wallet and the remote listener.** The config comment says the owner wallet must be preconfigured for the remote listener to start (`atn/config.py:195-197`), and the `start()` docstring says the same (`atn/ws_server.py:452-453, 475-477`). The code starts the remote listener whenever a host is set and allows agent self-auth without an owner wallet (`:478-489`).
3. **Remote port default.** The config comment says the remote port defaults to `local_ws_port + 1` (`atn/config.py:203-204`). The loader defaults `remote_ws_port` to 7701 (`:1266`), so the `port+1` fallback (`atn/ws_server.py:409`) is reached only with an explicit 0. The `atn send` default URL (`atn/cli.py:952`) and the MCP default URL (`atn/mcp_server.py:54-55`) also ignore `local_ws_port` and the integration port config.
4. **`secrets_config` read access.** Comments say an empty-payload `secrets_config` is a remotely reachable read-only GET (`atn/ws_server.py:154-157`). The whole type is in `KEY_LOCAL_ONLY_MESSAGES` (`:114`), so remote GETs are refused.
5. **`ws_input_policy` has no effect.** Both branches return `AllowAll` (`atn/runtime/__init__.py:1167-1176`), while the single-writer arbiter always gates WS `send_agent_message` and `delegate_message` (`atn/runtime/session_manager.py:123-127`, `atn/input_arbiter.py:214-230`). The config comment says `"allow"` lets every client drive (`atn/config.py:320-325`).
6. **`_handle_register_on_chain` docstring.** It describes a root-agent mode with a frontend `private_key` or unsigned call data (`atn/ws_server.py:5662-5672`). The parameters `is_root`, `private_key` and `sponsor_address` are never used in the body (`:5673-5809`). Registration is always daemon-signed.
7. **`confirm_agent_registration` sets the flag without proof.** It sets `registered_on_chain=True` with no chain check (`atn/ws_server.py:3414-3426`). The flag then drives autonet auto-start (`atn/runtime/__init__.py:900-910`).
8. **Integrity check results are discarded.**
   - `config.load_config` calls `_cache.warm_left()` on every load and discards the result (`atn/config.py:1521-1522`).
   - `autonet_service.start` calls `_cache.validate` and ignores the return value (`atn/autonet_service.py:923-933`).
   - The scheduler stores `_freshness_ok`, which nothing reads (`atn/runtime/scheduler.py:72, 179-183`), and it passes `registry_address` where `autonet_service` passes `dao_registry_address`. The config seed sets `registry_address` to the ServiceRegistry (`atn/config.py:646-647`).
   - `blockchain._session_nonce` (from `warm_right`) is never read (`nodes/common/blockchain.py:84-89`).
   - `_cache.py` lists itself as core-protected (`:19`) but `_CORE_FILES` omits it and `_iter_sources` skips it (`:46-54, 85`). Its constants are byte-encoded "to avoid literal grep matches" (`:65-68`).
9. **`ATN_DATA_DIR` disagreement.** `update_boot` claims to mirror `atn.config` by honoring `ATN_DATA_DIR` (`atn/update_boot.py:33-37`), but `atn.config` never reads it. Meanwhile the updater stages into `config.data_dir` (`atn/cli.py:837`). A custom `data_dir` in config.yaml means staged updates are never applied. `substrate_mcp` and `guest_sandbox` also use `ATN_DATA_DIR` or `~/.atn` rather than the config.
10. **(Fixed 2026-10-28: authored tools now get the adopted-tool scrubbed env and deny paths.)** **Authored tools inherit the daemon environment and filesystem.** Authored tools get `env = dict(os.environ)` with no deny paths (`atn/tool_store.py:1031-1045`). That environment can hold `BROKER_OWNER_SECRET` and provider keys, which the worker scrub (`atn/runtime/worker_manager.py:192-213`) and the startup invariant (`atn/runtime/__init__.py:232-245`) exist to keep away from lower-trust code. The keystore (`identity.age-key`, `vault.age`) is also readable unless `fs` is declared false. The `tool_guard` docstring says it is "for ADOPTED pinned tools" (`atn/tool_guard.py:1`), but it also wraps authored tools.
11. **(Fixed 2026-10-28: the local listener refuses tracked tool, shell-command and worker PIDs and their descendants, `atn/contained_pids.py`.)** **The local listener trusts more than the comments suggest.** It pre-auths any loopback process except guest-tool uids and PIDs (`atn/ws_auth.py:187-230`). Agent shell tools, workers and authored tools can therefore connect to :7700 as owner and call `export_agent_key` or `approve_adoption`, or act as any agent through `caller_id` (`atn/ws_server.py:4236-4237`). `ATN_MCP_AGENT` and the external-agent tool defaults are advisory on that listener for the same reason.
12. **Vault hardening conflicts with daemon reads.** `vault_setup` advises chowning the keystore to a separate `vault-svc` uid so agents cannot read the age key (`atn/vault_setup.py:143-154`), and `keystore.get_secret` says "Broker-side only" (`atn/_vendor/kevin/keystore.py:139-145`). Yet the daemon itself reads agent, owner and operator keys and `app.*` credentials straight from the keystore (`atn/runtime/agent_registry.py:296, 878, 900`, `atn/credentials.py:160`). Following the hardening steps would make every agent identity fail to load and regenerate (`atn/runtime/agent_registry.py:303-306`). The keystore docstring also still describes `C:\code\secrets\keystore` and an `acl.json` (`:24-29`).
13. **Broker socket paths differ.** On POSIX the tool-side default is `VAULT_BROKER_SOCK` or `/tmp/vault-broker.sock` (`atn/tool_secret_api.py:90`). The broker and daemon use `ATN_VAULT_BROKER_SOCK` or `<KEYSTORE_DIR>/vault-broker.sock` (`atn/_vendor/kevin/vault/vault_broker.py:93-95`, `atn/runtime/broker_client.py:48-51`). Adopted tools get a scrubbed env, so the default mismatch applies to them. `broker_client.py:85` refers to `atn vault setup`; the command is `atn-vault-setup`.
14. **(Fixed 2026-10-28: recovery to `author_pubkey` counts only when the chain binds it to `author`; approval re-reads `busted`.)** **Manifest signature check.** `_verify_manifest_sig` accepts a signature that recovers to either `author` or `author_pubkey`, and both values sit inside the payload being verified (`atn/tool_store.py:2205-2234`). A signer can claim any `author` as long as it stamps its own address as `author_pubkey`. The approval comment calls this check a re-attribution guard (`:2121-2139`). The `busted` check reads the stored proposal, not re-fetched state (`:2151-2155`), despite the surrounding "re-verify" comment.
15. **`use_tool` reaches MCP connector tools without a caller check** (`atn/tool_registry.py:196-197, 255`). An agent granted only `unified_tools` can call `mcp_<cid>_<op>` without the `connectors` bundle. The integration listener blocks this explicitly (`atn/ws_server.py:1155-1177`); the agent path does not.
16. **`register_on_chain` (agent tool) does not check the caller against `agent_id`** (`atn/agent_tools.py:4752-4811`). Any agent with `identity` can register any local agent whose parent is registered.
17. **Standalone `python -m atn.ws_server` diverges from the CLI** (`atn/ws_server.py:6294-6345`):
    - no remote or integration listener, fleet seeding or auto-update;
    - `register_surface` is not called, so `runtime.ws_bridge` stays None and the WS policy is not wired;
    - agents register without `legacy=True` and without a try/except, so a single agent lacking a budget can abort boot (compare `atn/cli.py:499-510`).
18. **`_try_reclaim_port` is effectively dead.** It kills a process whose cmdline contains `atn.mcp_server` and that listens on 7700 (`atn/cli.py:543-561`). Both MCP servers are stdio-only (`atn/mcp_server.py:213-219`, `atn/substrate_mcp.py:375`) and never listen.
19. **Auto-update default and duplicated check.** The CLI comment calls auto-update "opt-in via config" (`atn/cli.py:756`), but `AutoUpdateConfig.enabled` defaults to True (`atn/config.py:372, 402`). The dev-environment check runs twice (`atn/cli.py:762-771` and `:864-875`).
20. **External agents and `run`.** `trigger_run` returns None for external agents (`atn/runtime/execution_engine.py:128-131`), which the CLI `run` command reports as "At concurrency limit" (`atn/cli.py:249-255`).
21. **Stale names in the guest launcher.** Its docstring names `GuestLauncherClient in atn/guest_sandbox.py` (`atn/guest_launcher.py:55`); the class is `_LauncherProcess` (`atn/guest_sandbox.py:340`). `--guest-gid` is accepted and ignored (`atn/guest_launcher.py:429-430`).
22. **Run opt-ins no longer gate anything.** `RunOptIns` is kept but "no longer changes what runs" (`atn/tool_security.py:20-21, 74-79`). Only the UI uses it.
23. **`_resolve_household` always returns `""`** (`atn/agent_tools.py:3350-3355`), so `author_household` is never set (`:3344-3346`).
24. **`VoiceConfig` fields not loadable.** `tools_voice`, `tools_backend`, `piper_mandatory_for_alarm` and `local_overlay` exist on the dataclass but are never read from config.yaml (`atn/config.py:115-131` vs `:1396-1412`). The comment marks `local_overlay` as reserved.
25. **`stop_chat` ordering bug.** It sets `self.chat = None` before reading `self.chat.client`, so the client close never runs (`atn/runtime/__init__.py:1187-1200`).
26. **Raw secrets over plain `ws://`.** The stated reason for local-only key messages is that a raw secret would cross plain `ws://` behind a proxy (`atn/ws_server.py:60-66, 102-105`). `provider_configure` and `custom_provider_add`, which carry raw API keys, are not in that set (`:1605-1707`).
27. **Lineage helpers disagree.** `verify_lineage` hashes `charter_text` with `agent_id=""` (`atn/agent_identity.py:83-91`), while generation hashes the system prompt (`:70-71`). `compute_lineage_hash` ignores its `agent_id` argument (`:34-46`). `public_key` holds the address, not a public key (`:68`).
28. **Missing documentation files.** `atn guide`, `atn_guide` and `vault_setup` point to `atn/external_agents.md` and `atn/_vendor/kevin/vault/RUNBOOK*.md` (`atn/cli.py:990`, `atn/mcp_server.py:82`, `atn/vault_setup.py:126, 154`), which are not in this copy. `atn guide` and `atn_guide` would raise `FileNotFoundError` here.
