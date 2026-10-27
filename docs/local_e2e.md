# Local end-to-end: tool economy composes

`scripts/local_e2e_tool_economy.py` is the tool-substrate session's
capstone. It proves the whole tool-economy architecture (daemon,
substrate consensus, and chain) composes on a single local deployment,
with each hop verified against the next.

## Run

```bash
python scripts/local_e2e_tool_economy.py
```

Prerequisites (already true in this repo): `npm install` done
(`node_modules/` present, incl. Hardhat), contracts compiled
(`npx hardhat compile`), and the Python deps installed
(`web3`, `eth-account`, the `atn` + `nodes` packages importable).
Windows and POSIX both supported; the script manages the Hardhat node
subprocess and kills it (taskkill /T on Windows) on teardown.

The script prints a stage-by-stage scoreboard and exits nonzero if any
stage FAILs. A stage may legitimately SKIP (with a printed reason)
rather than sink time: currently only the ServiceMarket channel leg is
allowed to SKIP (it catches and notes any error rather than blocking).

## What it proves

Seven stages, each verified:

1. **Chain up**: launches `npx hardhat node` (localhost:8545), deploys
   `Substrate.sol` + `ServiceMarket.sol` (ServiceRegistry + PaymentChannel,
   both taking the substrate address) via a tiny inline `hardhat run`
   deployer. Settlement is ATN-only, so no MockERC20 is deployed.
   Hardhat's funded `account[0]` is the author's human OWNER wallet,
   `account[1]` the caller's.
2. **Fleet**: two real ATN `Runtime`s in tmp dirs (`autonet.enabled=False`;
   we drive the substrate pieces directly). Runtime A hosts `author-1`
   (owner = account[0]); it also hosts `caller-1` (owner = account[1] on
   chain) so tool invocation stays in one daemon (tools are daemon-local)
   while the two DISTINCT on-chain owners the damper needs still exist.
   Runtime B stands up its own agent to prove the two-runtime fleet.
   Verifies `author-1`'s parentless lineage is **owner-rooted** (hashes
   from the owner wallet).
3. **Chain identity**: owner-bound registration on chain. Reads each
   agent's key from `runtime.registry.get_agent_key`, funds the agent
   address with gas from its owner, signs the owner-binding EIP-712, and
   calls the registration method from the agent's own key. Verifies
   `getAgentOwner` returns the two distinct owners on chain.
4. **Tool economy (daemon)**: `author-1` registers a pinned echo tool
   (`register_tool`), then `publish_tool`s it (the production two-step
   publish gate); `caller-1` invokes it via `tool_registry.call_tool` and
   then runs `attest_tools` (the cognitive attestation, the only
   mint-counting usage). Two validators then `vet_tool` it: they first
   read the pinned code back, then attest a pass verdict with a report.
   Under v4.1 (vet gate retired) a vet is an INSPECTION REVIEW: it drifts
   position but mints nothing and gates nothing. A `WorldService` is wired
   to the tool_store's `manifest_sink`/`event_sink` exactly like
   `atn/autonet_service.py` does; the tool_used receipts are captured off
   the event sink and the registration sprout is read from the
   WorldService epoch buffer.
5. **Consensus**: the REAL federated close. Registration sprouts (author
   signing key) and receipts (caller signing key) become canonical
   `EventBatch` chains, driven through `FederatedCloseDriver.run()` with a
   mocked gossip (the `tests/test_tool_mint` TestDriverCarryOver pattern).
   The `agent_owner_map` is **read back from chain** (`getAgentOwner` for
   both agents), keyed by the same 0x addresses the events now carry.
   Verifies the mint credits `author-1`, `attesters == 1`, and, under
   v4.1, that the **vet gate is removed**: the tool mints from first
   attested use, the author keeps 100% (no validator royalty), and the
   inspection-review vets earn nothing. Also a **control case**: setting
   the caller's owner equal to the author's in the map drops the mint to
   zero, proving the chain-fed same-owner exclusion.
6. **Settlement**: Substrate's governor (hardhat #9 here; the DAO
   timelock in production) anchors the epoch (`anchor` via
   `EpochAnchorer`; a non-governor anchor is checked to revert), and the
   household's mint is claimed for owner A by another key (`claimFor` via
   `AuthoritativeChainSubmitter`, mint-merkle proof against the anchored
   root; the ATN lands on owner A, the submitter receives nothing). Money only (Decision 2026-07-10):
   Substrate has no reputation surface: the mint is a 2-field
   `(agent, amount)` leaf. Verifies the cumulative tool-pool earnings
   ledger (`agentMintTotal`) and the ATN `balanceOf` both moved on chain
   by the scaled `agent_mint[author]` amount (REP is claimed DAO-side,
   not minted here). Then registers one Service on `ServiceRegistry` and
   runs one `PaymentChannel` round-trip in ATN (client opens a deposit →
   signs one EIP-712 voucher → provider closes → provider balance +ask NET
   of the 2.5% service fee, routed through `payForService`; remainder
   refunds fee-free after the challenge window).
7. **Teardown**: kills the Hardhat node, removes tmp dirs, prints the
   scoreboard.

## Expected output (sketch)

```
  [PASS] Stage 1: chain up (hardhat node + deploy)
  [PASS] Stage 2: fleet (two ATN runtimes, owner-rooted lineage)
           author_lineage_root=0xf39Fd6e5…   author_addr=0x…
  [PASS] Stage 3: chain identity (owner-bound registration)
           author_owner_onchain=0xf39Fd6e5…  caller_owner_onchain=0x7099797…
           distinct_owners=True
  [PASS] Stage 4: tool economy (register + invoke + attest + vet)
           registration_events=1  attested_receipts=1  vets=2
  [PASS] Stage 5: consensus (federated close + owner-map exclusion)
           tool_mint_raw=4.15888308  agent_mint_raw=8.15888308
           vet_gate=removed (v4.1); author keeps 100%
           control_same_owner_mint=0 (excluded)
  [PASS] Stage 6: settlement (anchor + recordTraining + channel)
           earnings_delta=8158883  atn_delta=8158883
           service_id=1  channel_provider_delta=975  channel_client_refund=2000
  [PASS] Stage 7: teardown
  7 PASS / 0 FAIL / 0 SKIP
```

## Seams this e2e revealed

These are the composition frictions found while wiring the loop
end-to-end. None block the loop (the script works around each), but each
is a real observation about where the pieces don't yet click cleanly.

1. **Registration sprout carries `author_post=False` → zero-standing tool
   mint.** `WorldService.submit_tool_manifest` emits the registration
   `SubClaimSprouted` with `author_post=False`. In the default
   **ledger-pricing** federated close (no equilibration), the manifest
   claim node then has zero `net_score` standing, so
   `mint = standing × usage_term = 0`: a freshly-registered, genuinely-
   attested pinned tool mints NOTHING. The unit fixture
   (`tests/test_tool_mint._registration_event`) hand-sets
   `author_post=True` to get standing 1, which is why the tests pass while
   the live daemon path would mint zero until standing accrues organically.
   The script sets `author_post=True` on the buffered registration events
   (documented inline) to exercise the mint. **Open question for the
   owner:** should tool registration stamp an author post (giving the
   manifest immediate unit standing), or is a tool author supposed to earn
   standing only through subsequent review activity? This is the single
   most load-bearing seam: the whole "tools mint to their author" claim
   hinges on it. (v4.1 note: usage mints from first attested use
   regardless, so this seam is now about ledger-pricing standing math, not
   the mint gate. The vet gate it once interacted with is retired.)

2. **Tool registration events don't reach the `event_sink`.** Only
   `tool_used` receipts flow through `tool_store.event_sink`; the manifest
   registration goes through `manifest_sink → submit_tool_manifest`, whose
   sprout lands in the WorldService's epoch buffer, not the sink. So a
   consumer that only taps `event_sink` (the natural place to collect
   "everything for the epoch batch") is missing the registration and gets
   an empty mint. The script reads registration events out of
   `ws._epoch_events` instead. A single unified "all substrate events for
   this epoch" tap would remove this footgun.

3. **String agent-id vs 0x address at the mint→chain boundary.** The tool
   events carry the daemon-local agent id (`author-1`, `caller-1`) in
   `author`/`tool_author`/`author_agent`. `federated_epoch_close` keys
   `agent_mint` by that string. But `mint_merkle` only admits keys that
   parse as 0x addresses (the contract can only verify `msg.sender`), so a
   string-keyed mint has **no on-chain claim path**: it silently drops at
   the merkle-leaf stage. Production bridges this with
   `world_substrate_feed.ResolvedAgentIdentity` (local id → 0x). The script
   replicates that remap explicitly before building the canonical batches.
   The seam: nothing in the tool-mint path itself enforces or documents
   that the author field must be an address by the time it reaches the
   anchor; it's an implicit contract between the feed resolver and the
   merkle builder.

4. **On-chain amount is `agent_mint[author]`, not `tool_mint[digest]`.**
   The tool-author mint is merged into the broader `agent_mint` map
   (through the violator-pays gate) BEFORE anchoring, so the amount that
   hits the chain (`8.15…`) is larger than the standalone tool_mint entry
   (`4.15…`). Verifying the chain delta against `tool_mint[digest]` fails;
   the correct source is `agent_mint[author]`. Obvious in hindsight, but a
   trap for anyone reasoning "the tool minted X, so X should appear on
   chain."

5. **Hardhat node's `chainId` is 1337, not 31337.** `npx hardhat node`
   serves the in-process `hardhat` network (`hardhat.config.js`:
   `chainId: 1337`), while the daemon's chain config defaults to `31337`
   (the eth_tester default). Signing txs for 31337 gets rejected with
   "incompatible EIP-155 transaction, signed for another chain." The
   script pins `CHAIN_ID = 1337`. Worth noting for anyone pointing the
   real daemon at a local `hardhat node`.

6. **Hardhat node stdout must be drained.** `npx hardhat node` logs every
   RPC call; if its stdout PIPE buffer fills, the process blocks and the
   deploy side sees an undici `HeadersTimeoutError`. The script sends the
   node's stdout/stderr to DEVNULL. (A subtle subprocess-management trap,
   not an architecture seam, but it cost a run.)

7. **Owner-binding EIP-712 is under active rename.** During this work the
   contract's owner-binding struct was renamed Sponsorship → OwnerBinding
   (`registerAgentSponsored` → `registerAgentBound`, `sponsorshipHash` →
   `bindingHash`, `sponsorshipNonce` → `bindingNonce`, event
   `AgentSponsored` → `OwnerBound`) and gained a mandatory `nonce` in the
   typed struct. The script is written to survive either shape: it
   discovers the method/view/event names by ABI introspection and, rather
   than reconstruct the typed-data struct itself, asks the **contract's
   own `bindingHash`/`sponsorshipHash` view** for the exact digest and
   signs that raw 32-byte digest with the owner key. This sidesteps all
   struct-field/typehash drift: whatever the contract hashes is what gets
   signed. Any future field addition to the binding struct needs no script
   change as long as a hash view is exposed.
```

# Local end-to-end: the venture loop (scrapped)

`scripts/local_e2e_venture_loop.py` exercised the VentureVault, its verifier
trials and the evidence CON rail. The venture vault was scrapped on
2026-10-26 (no real use case: capital in ATN is not anyone's bottleneck, and
the jurisdiction's trustless projects already fund work), so the script, the
contract and the trial rail are gone. See `docs/tool_substrate.md`,
"Decision (2026-10-26): the venture vault is scrapped".

# Local end-to-end: cross-daemon inference, nothing mocked

`scripts/local_e2e_cross_daemon_inference.py` proves the marketplace
inference rail (`docs/services_market.md`, "Decision (2026-07-26)") with
**no mocked seam at all**. Its sibling
`scripts/local_e2e_service_provider.py` proves the *economics* with the
model faked (a `CannedProvider` on the selling side, two in-process
Runtimes); this one answers the question that leaves open: does the thing
the child paid for actually think?

Three things are real here that are stubbed everywhere else:

- **Two real daemon processes.** Two `python -m atn` subprocesses, each
  with its own isolated home, its own `config.yaml`, its own WS port.
  Every frame crosses a real socket between two real OS processes.
- **A real model.** The provider daemon serves off its own provider stack
  (ollama). The assertion is not "non-empty" but "answers a
  *distinguishable* prompt": the script asks for one specific word and
  requires it in the reply, which a canned responder cannot satisfy.
- **A real agent turn.** The child is driven through
  `send_agent_message` (the surface a human's app uses), not by calling
  `ServiceProvider.send` directly.

## Run

```bash
python scripts/local_e2e_cross_daemon_inference.py     # 9 stages
```

Prerequisite: a reachable ollama with any small model installed
(`ollama pull qwen3:4b`). The script starts `ollama serve` itself if the
binary is present and nothing is listening, and refuses to run against a
canned model, because a fake model would defeat its entire purpose.

## Parameterized for a remote re-run

Every endpoint is an env var defaulting to a fully local run, so the same
script re-runs with the consumer on another machine:

| Env var | Default | What it names |
|---|---|---|
| `E2E_RPC_URL` | `http://127.0.0.1:8545` | chain RPC both daemons dial |
| `E2E_PROVIDER_WS` | `ws://127.0.0.1:7710` | the endpoint published **on chain**, what the consumer resolves and dials |
| `E2E_PROVIDER_HOST` / `E2E_CONSUMER_HOST` | `127.0.0.1` | bind hosts for the two listeners |
| `E2E_PROVIDER_PORT` / `E2E_CONSUMER_PORT` | `7710` / `7720` | the two daemons' WS ports |
| `E2E_HARDHAT_HOST` | `0.0.0.0` | hardhat bind (wide by default, so an off-box consumer can reach the chain) |
| `E2E_OLLAMA_URL` | `http://127.0.0.1:11434` | the provider daemon's model backend |
| `E2E_MODEL` | first installed | served model id |
| `E2E_SKIP_PROVIDER` | unset | `1` => do not launch the provider; assume one already serves at `E2E_PROVIDER_WS` |
| `E2E_PROVIDER_HOLD` | unset | `1` => run the **provider side only** (stages 1, 2, 3, 5a, 6), print the consumer's exports, and hold until released |
| `E2E_HOLD_SENTINEL` | `<provider home>/e2e_provider_release` | a path whose appearance releases a hold (Ctrl+C also works) |
| `E2E_PROVIDER_REMOTE_PORT` | `E2E_PROVIDER_PORT + 1` | the provider's **remote** (auth-required) listener, the only port an off-box buyer can reach |

`E2E_PROVIDER_WS` is deliberately separate from `E2E_PROVIDER_HOST`:
behind NAT or ZeroTier the reachable address is not the bind address, and
it is the *reachable* one that must go on chain.

The consumer half additionally reads what the hold side prints:
`E2E_SUBSTRATE`, `E2E_SERVICE_REGISTRY`, `E2E_SPEC_DIGEST`,
`E2E_PROVIDER_ADDR`, `E2E_MODEL`.

## The two-box run (verified 2026-07-26)

Ran green across two machines on a ZeroTier LAN: provider (hardhat +
ollama `qwen3.5:4b` + provider daemon) on Windows, consumer daemon +
child agent on an Ubuntu 24.04 EC2 box. **7 PASS / 0 FAIL / 2 SKIP**, the
two SKIPs being the stages that belong to the other box.

On the **provider** box:

```bash
E2E_PROVIDER_HOLD=1 E2E_PROVIDER_HOST=0.0.0.0 \
E2E_PROVIDER_WS=ws://<reachable-ip>:7711 E2E_HARDHAT_HOST=0.0.0.0 \
  python scripts/local_e2e_cross_daemon_inference.py
```

It runs stages 1/2/3/5a/6, prints a block of `export` lines (also written
to `<provider home>/e2e_exports.json`), and waits. On the **consumer**
box, paste those and run the same script: it dials the provider's chain,
resolves the listing and the wss endpoint *from chain*, boots its own
daemon, and drives the paid turn. Release the provider with Ctrl+C or by
touching the sentinel.

Each side runs only what it owns. The consumer never starts a chain (it
would have neither the contracts nor the provider's registrations); the
provider-local `requests.jsonl` / `served_requests.json` assertions run
only where those files exist, and cross-machine the replay refusal goes
over the real wire through `service_client.request_service` instead,
which is the stronger proof of the same gate.

### Reaching the provider off-box

The privileged local listener is **loopback-only and not configurable**
(`atn/cli.py` pins `host="localhost"`) by design: it is pre-authed as
owner and exports keys. Off-box reachability is the **remote** listener
(`autonet.remote_ws_host` / `remote_ws_port`), which is auth-required, so
the script enables it automatically when `E2E_PROVIDER_HOST` is not
loopback and publishes *that* port on chain.

A paying counterparty on another machine can never pass that handshake:
it is not the daemon's owner and holds no key in the daemon's fleet. So
`service_request` is dispatched **pre-auth** on the remote listener
(`PAYMENT_AUTHORIZED_MESSAGES` in `atn/ws_server.py`): the on-chain
payment is the credential, which is the doctrine
`docs/services_market.md` §3 already states ("authenticates the channel,
validates vouchers"). It is a strictly stronger credential than a session
login: it costs ATN per call, is checked against the receipt (recipient +
amount), and is consumed against a replay set. Nothing else on the daemon
is reachable from such a session.

## What it proves

1. **Chain up**: hardhat + Substrate + ServiceRegistry, bound wide.
2. **Real model up**: ollama reachable, model installed and warmed.
3. **Provider daemon**: real subprocess, isolated home, port 7710.
4. **Consumer daemon**: second real subprocess, port 7720. It has
   **no provider stack of its own**: the child thinks on purchased
   cognition or not at all, so a silent fallback to a local model cannot
   let the test pass while proving nothing.
5. **Agents on chain**: provider agent (the payment recipient), consumer
   parent + child; the child's wallet funded through the production mint
   rail (anchor an epoch, child records its own mint).
6. **Service listed**: inference-backed spec over the real WS handler,
   registered in `ServiceRegistry`, wss endpoint published on chain.
7. **The real turn**: the child cannot bind itself; the parent binds it;
   `send_agent_message` drives one turn; a real model answers the probe.
8. **Settlement**: `ServicePayment` with child payer / provider
   recipient / correct amount, provider credited net of the 2.5% fee,
   owner wallet untouched, the provider daemon's own `requests.jsonl`
   recorded the served item, the gate verified on chain rather than
   degrading open, and a replayed `request_id` is refused.
9. **Teardown**: both daemons, the chain, and any ollama the script
   started, on every path.

## Seams this e2e revealed

1. **The local WS port was hardcoded, and a second daemon KILLED the
   first.** `atn/cli.py` passed the `DEFAULT_PORT` (7700) literal, and on
   collision called `_try_reclaim_port`, which kills whatever holds the
   port. A second daemon on one machine therefore murdered the running
   one. Fixed: `autonet.local_ws_port` (0/unset => 7700), and an
   EXPLICITLY configured port is never reclaimed: a configured collision
   is operator error or a live sibling, not the stale-MCP-server case the
   reclaim exists for.

2. **The service payment gate blocked the event loop.** `on_chain.py`'s
   methods are `async def` but synchronous inside (web3 HTTP), and
   `_validate_service_payment` awaited `verify_service_payment` directly.
   That occupies the loop for the whole receipt fetch, so the buyer's
   connection (and every other client of the provider daemon) starves on
   the websocket keepalive and gets dropped mid-request, surfacing as an
   opaque transport error for what is really a server-side stall. Fixed by
   routing it (and the voucher path's `verify_voucher`) through
   `_offload`, the same treatment `invoke_service` and
   `owner_binding_status` already carry. Only observable with a REAL
   counterparty on a real socket: the in-process sibling script calls the
   handler directly and never has a keepalive to starve.

3. **`create_agent`'s `prompt` is a work instruction, not an identity.**
   It activates the agent and immediately `trigger_run`s it. Passing one
   to each agent made the *provider's* own agent start thinking on
   creation, queueing ahead of the buyer's paid work item on the single
   local model and failing the purchase. Identity goes in
   `system_prompt`, which registers the agent idle.

4. **A fresh agent is `REGISTERED`, not `ACTIVE`, and
   `send_agent_message` only TRIGGERS a run for an ACTIVE agent**:
   otherwise it queues to the inbox and returns no `execution_id`, and the
   turn silently never happens. The script calls `activate_agent` first
   and asserts it got an `execution_id`, so this failure mode can never
   again masquerade as a timeout.

5. **`Path.home()` redirection needs more than `USERPROFILE`.** On
   Windows `ntpath.expanduser` resolves `USERPROFILE`, then
   `HOMEDRIVE`+`HOMEPATH`, then `HOME`; all four are typically set, so
   the script sets `USERPROFILE`/`HOME` and *clears* `HOMEDRIVE`/`HOMEPATH`
   or the real home leaks back.

6. **The child daemon must not launch from the repo root.**
   `_default_agents_dir` returns a relative `agents/` if one exists in the
   CWD, *before* consulting `data_dir`, so an "isolated" daemon started
   from the checkout happily shares `autonet/agents`. The script launches
   from the temp home and puts the repo on `PYTHONPATH`.

7. **The daemon shuts down on stdin EOF.** `_input_loop` breaks on a
   closed stdin, so a subprocess given `DEVNULL` exits seconds after boot.
   It needs a pipe that stays open.

8. **The port LISTENS before the runtime finishes booting** (~10s to
   listen, heavy numpy/world_model imports after). An early connect is
   accepted and then hangs, so the client retries the whole handshake and
   proves liveness with a real `snapshot` round trip rather than trusting
   the TCP accept.

9. **`ask.token` is GONE** (2026-07-26). The vestigial field (required
   by `validate_ask` long after ATN-only settlement was ratified,
   2026-07-10) was removed: an ask is `{amount, unit}`, ATN-denominated
   by construction. Both e2e scripts stopped sending the Substrate
   address. Removal is tolerant: `normalize_ask` strips a `token` an old
   caller still passes, and persisted specs carrying one still load.

10. **A reasoning/"thinker" model fails the probe for the wrong reason.**
    It emits hundreds of tokens of deliberation before the one word asked
    for, which the declared token cap then truncates. The script prefers
    an instruction-following chat model when auto-selecting; `E2E_MODEL`
    overrides.

11. **The replay guard was prefix- and case-sensitive, and one payment
    bought TWO real inferences.** Found only by the two-box run
    (2026-07-26), and the most serious defect either e2e has surfaced.
    `ServiceStore.has_served_request` / `mark_request_served` keyed the
    seen set on the **raw string**, and the two sides of the wire
    genuinely spell one bytes32 differently:
    `service_client.new_request_id()` emits `0x<64 hex>`, while the
    on-chain `ServicePayment.requestId` decodes to bare hex (case varying
    by web3 version). The payment *verifier* normalizes to bytes32
    correctly, so both spellings verified against the same receipt. Only
    the replay set disagreed. A replayer flipping the `0x` (or the case)
    got a second free work item per payment, indefinitely.

    The provider's own log recorded it plainly: two `ok:true` rows for one
    `ServicePayment`, ids `8ed3f2…` and `0x8ed3f2…`. Fixed with
    `ServiceStore.normalize_request_id` (lowercase, `0x`-stripped) applied
    on read, on write, **and on load**, because a set persisted by a pre-fix
    daemon may hold both spellings of one id and must still refuse both
    after the upgrade. Regression test:
    `tests/atn/test_service_store.py::TestServiceRequestDispatch::test_replay_guard_is_prefix_and_case_insensitive`.

    Why the single-box runs missed it: on one machine the buyer's request
    id round-trips as the same string, so the two spellings never met. The
    two-box run reads the id back off the *chain event* to build the replay
    probe, which is what produced the other spelling, and what a real
    attacker would do.

12. **`anchor` hash-chains, so the second box's anchor must read the
    head.** `fund_agent_atn` passed zero `prevEpochRoot`/`prevAnchorHash`,
    which is correct only for the very first anchor. In a split run the
    provider box anchors first, so the consumer's own funding anchor
    reverted with `PrevAnchorHashMismatch`. It now reads
    `latestEpochRoot`/`latestAnchorHash` and chains onto them, and the two
    sides use distinct epoch ids.

13. **A stale pip-installed `atn` will shadow the synced source.** On the
    remote box the venv had `autonet-computer` 0.2.7 installed; with a
    mis-expanded `PYTHONPATH` the daemon silently booted the *ancient*
    package (which ignored `local_ws_port` and tried to seize 7700, the
    live daemon's port). The script always puts the repo first on
    `PYTHONPATH`, but on a box with a pip-installed daemon, verify with
    `python -c "import atn; print(atn.__file__)"` before trusting a run.

# Local end-to-end: the tool security rail (staked reviews)

`scripts/local_e2e_tool_security.py` proves staked reviews
(docs/multi_jurisdiction.md decision 10, trustless-contracts ToolBond
448023a) against real contracts on a local hardhat node, driving the daemon
code paths.

## Run

```bash
python scripts/local_e2e_tool_security.py
```

Needs `node_modules` with hardhat + `@openzeppelin/contracts` at or above
the repo root, and a trustless-contracts checkout (env
`TRUSTLESS_CONTRACTS_DIR`, default `C:/code/dao/trustless-contracts`). The
jurisdiction sources it needs (ToolBond, ProposalLifecycle, RepToken,
Registry and their interfaces) are COPIED into a scratch hardhat project at
`build/e2e_tool_security/` (gitignored) next to this repo's `Substrate.sol`,
compiled with solc 0.8.26 and served on port 8557
(`E2E_TOOL_SECURITY_PORT`). The checkout is never written. HOME,
USERPROFILE and KEYSTORE_DIR point at a temp dir for the whole run.

## What it proves

1. **Chain up**: a Registry holding the default charter text, RepToken
   (deployer is admin, a hardhat account is the timelock), Substrate
   (deployer as governor and vaultMinter for test ATN), ToolBond (minStake
   1000 ATN, reveal window 300s, degrade 10%, author slash 100%, reviewer
   share 40%, harm weight 2x, appeal window at its 2-day floor), and
   `setSlasher(ToolBond)`.
2. **Fleet**: two real `Runtime`s. Daemon A hosts `author-1` and `rev-1`;
   daemon B hosts `rev-2` and fetches manifests and code from A through
   `tool_store.blob_fetcher`. REP 100/300/200, ATN 5000/500/500.
3. **Publish**: `register_tool` + `publish_tool` -> `publish_with_stake`
   -> `ToolBondClient.register_and_bond`, for a clean tool and a malware
   tool.
4. **Genesis anchor**: the governor anchors epoch 0 so REP weighs from
   epoch 1 (an epoch's REP snapshot is its start).
5. **Review step**: `attest_tools` (rev-1 on the clean tool, owner-granted)
   and `vet_tool` (rev-1 on the malware tool with harm 1 and a 100 ATN
   stake; rev-2 on both from daemon B with scores and small stakes) queue
   entries into each reviewer's epoch batch.
6. **Commit and reveal**: `ReviewDriver.tick` commits nothing until the
   epoch's local close fires, then commits each batch (commitment + total
   stake); the governor anchors epoch 1 and the next tick reveals both.
7. **Finalize**: after the reveal window the clean round settles at once
   (harm 0) and the malware round waits out the appeal window. Settled
   harm 0.68 (the REP-weighted mean); the author's bond is slashed by 68%
   and their REP by 68%; stakes, winnings (paid through `payForService`)
   and REP move by accuracy; REP is conserved; every batch ends `settled`.
8. **Ratings**: `ToolBondClient.rating` reads toolRating/toolHarm; WS
   `tool_security_status` and `review_batches` show them.
9. **Close**: pinned to anchor 1 the review snapshot is empty; after anchor
   2 `read_voice_state` returns both ratings and both rounds, and
   `federated_epoch_close` with three attested uses of the clean tool and a
   10 ATN pool gives the reviewers 40% split by stake x accuracy and the
   author 60%; `tool_positions` is the on-chain rating.
10. **Teardown**: kills the node, removes the temp dirs.

Not covered: anchoring through a real DAO proposal (the governor is an EOA
here), DAO overrule (`daoOverrule` is timelock-only), forfeit of a missed
reveal, unbond/withdraw.

# Local end-to-end: the robot MVP

`scripts/local_e2e_robot.py` runs the robot side (docs/robots.md) against a
simulated TurtleBot3 in headless Gazebo, in Docker (`robot/sim/`). The
daemon's robot service runs in-process against a temp home.

## Run

```bash
python scripts/local_e2e_robot.py          # 10 stages; builds the image on first run
E2E_ROBOT_KEEP=1 python scripts/local_e2e_robot.py   # leave the container up to inspect
```

## What it proves

1. The sim is up: rosbridge reachable, `/scan` and `/odom` flowing.
2. The daemon exports the required pack and the filter verifies and loads it.
3. Driving at an obstacle over the speed cap is slowed, then stopped short of it.
4. Reversing away is allowed.
5. A pack edited in place is refused, and nothing moves.
6. Re-exporting restores motion.
7. A stale daemon heartbeat stops the robot.
8. An unreachable sponsor stops the robot.
9. A challenge answer verifies; a tampered chunk, a wrong key and a late answer fail.
10. The agent's ros-mcp connector pushing forward is clipped and never crosses
    the stop distance.

Limits are listed in docs/robots.md ("Known limits of the MVP").
