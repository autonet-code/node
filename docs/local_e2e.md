# Local end-to-end: the household economy loop

`scripts/local_e2e_tool_economy.py` is the economy proof-of-life: one turn
of the loop in `economy_design.md`, end to end on a local hardhat node,
through the daemon's own code. Three households take part. Each is an owner
wallet and a daemon whose operator key is registered on Substrate bound to
that wallet, and each daemon's external agent acts on chain as its operator
(Decision 2026-11-08). Household A authors a tool, B and C review it, C sells
a service and B buys it.

## Run

```bash
python scripts/local_e2e_tool_economy.py      # 9 stages
```

Needs `node_modules` with hardhat + `@openzeppelin/contracts` at or above
the repo root, and a trustless-contracts checkout (env
`TRUSTLESS_CONTRACTS_DIR`, default `C:/code/dao/trustless-contracts`). As in
the tool-security script below, the jurisdiction sources it needs
(ToolBond, RepToken, Registry, AutonetRepClaim and their interfaces) are
COPIED into a scratch hardhat project at `build/e2e_tool_economy/`
(gitignored), with this repo's `Substrate.sol` and `ServiceMarket.sol` under
`contracts/core/`, compiled with solc 0.8.26 and served on port 8558
(`E2E_TOOL_ECONOMY_PORT`). The daemon code reads the ABIs of that compile
(`ATN_ARTIFACTS_DIR`). The checkout is never written. HOME, USERPROFILE and
KEYSTORE_DIR point at a temp dir for the whole run. The script prints a
stage-by-stage scoreboard and exits nonzero if a stage fails; a failed stage
stops the run, and teardown still runs. About 20 seconds after the first
compile.

## What it proves

1. **Chain up**: a Registry owned by the timelock (an EOA here) holding the
   default charter, the treasury ToolBond pays; RepToken; Substrate (fee
   2.5%, half burned; the deployer is the vault minter; operator A is the
   governor, so it anchors directly, as on a dev network); ToolBond (minStake
   100 ATN, reveal window 300 s, reviewer share 40%) as RepToken slasher;
   AutonetRepClaim as RepToken minter; ServiceRegistry and PaymentChannel
   (challenge window 60 s).
2. **Households**: three `Runtime`s, each with its owner wallet and a
   configured operator key (one process has one vault, which holds one
   operator key). `create_agent` with provider `external` makes each
   daemon's agent: no keypair, its address is the operator's. The owner
   signs the OwnerBinding and the operator registers on Substrate
   (`OnChainService.register_agent_bound`); the owner wallets never hold
   gas. Genesis REP goes to the owner wallets (100, 300 and 200, minted by
   RepToken's admin as the tool-security run seeds it), the operators buy
   ATN from the vault, and the genesis anchor lands. Each daemon weighs its
   operator by its owner's REP at the epoch's snapshot, as ToolBond does.
3. **Publish**: A's agent runs `register_tool` + `publish_tool` with a
   novelty claim. The manifest's author and `Substrate.toolAuthor` are
   operator A, the 100 ATN bond leaves its balance, daemon B's gate sees
   the bond, and the daemon announces the manifest once
   (`tool_store.manifest_sink`, which the autonet bridge turns into the
   `tool_registered` event).
4. **Reviews**: B adopts the tool (manifest and code fetched from A's blob
   store), is reminded to review it, uses it and reviews it with
   `attest_tools`. C checks the code out (`adopt_tool` with a path), uses it
   and reviews it. Each review carries the four fields and lands in the
   operator's epoch batch; one missing a field is refused. Each daemon's
   `ReviewDriver` commits at the close with the operator key (weight = the
   owner's REP at the snapshot, no ATN moves), reveals after the anchor
   (`Reviewed` carries the fields), then finalizes and claims after the
   reveal window (REP conserved, the rating a REP-weighted mean). The
   reviewing households hold 83% of REP and are 2 of the 3 REP-holding
   households: over the conclusion gate.
5. **Fees**: C's owner sells a local tool as a service (WS
   `register_service`, ServiceRegistry entry under operator C, endpoint
   published on chain, the daemon's WS listener up). B's agent pays with
   `pay_for_service` and sends the request with the proof
   (`request_service`); C serves it. Operator C gets the 400 ATN net of the
   10 ATN fee: 5 ATN to the treasury, 5 ATN burned. That burn is the next
   close's pool (the pool lags the burns by one epoch).
6. **Close, anchor, ATN claims**: `read_voice_state` pinned to the anchor
   after the burn feeds `federated_epoch_close` with the tool's
   registration event. The pool is the 5 ATN burned; the tool concludes and
   is paid once, its share the whole pool: 40% to B and C by accuracy, the
   author side 60% x V x C x S x N to A, keyed by owner wallet, the rest
   carried over. Each daemon's `ChainSubmissionDriver` takes the close: A's
   anchors it (`agentMintRoot` over the mint and concluded leaves), B's and
   C's find it anchored. Each operator claims its owner's share (`claimFor`:
   the owner wallet receives it, `agentMintTotal` the same, the operator
   only pays gas) and shows ToolBond the conclusion (`conclude` from the
   leaf: no malware, the bond stays).
7. **REP**: in the same driver step each operator claims its household's
   REP through AutonetRepClaim: `RepClaimed` to the owner wallet at exactly
   `agentMintTotal` x 1e12 (1 ATN earned is 1 REP), the watermark at the
   earnings, nothing claimable after, a second claim reverts, and the REP
   votes for its household.
8. **Payment channel**: B's agent buys C's service twice with
   `request_service` and no direct payment. One channel opens with a 1200
   ATN deposit, one voucher per request, and C keeps the best one (800 ATN).
   C's settlement sweep (`settle_channels_once`) closes the idle channel
   with it (C paid net of the fee). B's sweep sees the provider's close and,
   after the challenge window, withdraws the 400 ATN remainder, fee-free.
9. **Teardown**: the WS listener, the daemons and the hardhat node this run
   started; the temp home is removed.

Not covered: the anchors before the mint's are plain governor anchors with
placeholder roots (the AnchorElection path runs in `scripts/s0_local_run.py`); the
DAO's actions, malware verdicts and bounties (the tool-security script
below); libp2p gossip and blob transport (one process: the daemons read
each other's blob stores, and one in-memory resolver stands in for the
anchor blobs).

## Notes

- The chain clock runs ahead of the wall clock here: `evm_increaseTime`
  passes the reveal window, and hardhat stamps each block at least a second
  after the last. The buyer's refund sweep compares the channel's
  `closesAt` with the wall clock, so stage 8 hands `sweep_refunds` the
  chain's time. On a live chain the two agree.
- The hardhat node serves chain id 1337 (the scratch config), and its
  output goes to DEVNULL: a full pipe blocks the node.

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
   listen, heavy numpy imports after). An early connect is
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

# Local end-to-end: reviews, the review mint and malware verdicts

`scripts/local_e2e_tool_security.py` proves the reviews redesign and the
malware verdicts (Decisions 2026-11-09 and 2026-11-10,
`docs/tool_substrate.md`; trustless-contracts ToolBond 8dcc1be) against real
contracts on a local hardhat node, driving the daemon code paths.

## Run

```bash
python scripts/local_e2e_tool_security.py
```

Needs `node_modules` with hardhat + `@openzeppelin/contracts` at or above
the repo root, and a trustless-contracts checkout (env
`TRUSTLESS_CONTRACTS_DIR`, default `C:/code/dao/trustless-contracts`). The
jurisdiction sources it needs (ToolBond, RepToken, Registry and their
interfaces) are COPIED into a scratch hardhat project at `build/e2e_tool_security/`
(gitignored) next to this repo's `Substrate.sol`, compiled with solc 0.8.26
and served on port 8557 (`E2E_TOOL_SECURITY_PORT`). The checkout is never
written. HOME, USERPROFILE and KEYSTORE_DIR point at a temp dir for the
whole run. About two minutes after the compile.

## What it proves

1. **Chain up**: a Registry holding the default charter (the four value
   axes) and owned by the timelock (it is the treasury), RepToken,
   Substrate (deployer as governor and vaultMinter), ToolBond (minStake 100
   ATN, reveal window 300 s, degrade 10%, author slash 100%, reviewer share
   40%, unbond delay 120 s, no appeal period; the timelock is an EOA so the
   run plays the DAO's proposals), and `setSlasher(ToolBond)`.
2. **Fleet**: two real `Runtime`s, every agent its own household. Daemon A
   hosts `author-1` and `rev-1`, daemon B `rev-2` and `rev-3`; they fetch
   manifests, code and evidence from each other through
   `tool_store.blob_fetcher`. REP 100/300/200 at genesis.
3. **Publish**: `register_tool` + `publish_tool` bond three tools (one
   clean, two that pipe a remote script into a shell); a 10 ATN bounty on
   the clean one.
4. **Genesis anchor**: rev-3 gets 50 REP after it, and the daemon reads 0
   for rev-3 at epoch 1's snapshot (as ToolBond weighs a commit).
5. **Review step**: `attest_tools` and `vet_tool` queue reviews with the
   four fields and charter axes, no stake; a missing field and a malware
   flag without evidence are refused; rev-1 flags both bad tools with
   evidence; rev-3 is refused, and ToolBond would revert its commit
   NoWeight.
6. **Commit and reveal**: commit weight = REP at the snapshot (300, 200), no
   ATN moves; the reveal emits the four fields and the evidence hash, whose
   blob daemon B fetches (`fetch_evidence`).
7. **Finalize and claim**: accuracy over the four fields and four axes
   (0.91875 on the clean tool), REP conserved, the bounty paid 6:4 to the
   households, ratings as REP-weighted means; rev-1's malware rounds are
   claimed like the others (a slashed stake goes to the treasury).
8. **Withdraw guard**: unbond + withdraw of a tool 60% of its REP calls
   malware reverts MalwareSuspected; a top-up cancels the unbond.
9. **Close**: `read_voice_state` pinned to anchor 2 feeds
   `federated_epoch_close`: all three conclude ({clean 0, bad 6000, 6000}
   bps), each a third of the 5 ATN pool by its REP; the clean author side is
   60% x V x C x S (0.327746) x N; the bad tools' author sides are unpaid,
   recorded in ATN (`malware_unpaid`: 0.074535 ATN for one, nothing for the
   one whose correctness mean is below 0) and carried over, and both are
   pruned. The blob and the agentMintRoot
   (mint and concluded leaves) are anchored and every claimant claims
   through `AuthoritativeChainSubmitter`.
10. **Verdicts**: `VerdictSubmitter` concludes all three from the anchored
    leaves, which settles both malware verdicts at once: 60% of each stake
    to the treasury, the author household's REP x 100% x 0.6 burned per
    tool, nothing to the flaggers; the author withdraws what the slash left
    of one; every batch settles.
11. **The DAO's word**: only the timelock designates or clears
    (NotTimelock), and clear needs a verdict (WrongStage); it designates
    the clean tool (the whole stake to the treasury, no REP burn) and clears
    one bad tool (ToolBond returns the REP it burned); the treasury pays
    that tool's author the slashed stake and the recorded unpaid amount
    (`Registry.transferERC20`). The next close, pinned after them, prunes
    the designated tool, un-prunes the cleared one, mints nothing (a tool
    is paid once) and conserves the pools (minted + carried = both pools).
12. **Owner surfaces**: WS `tool_security_status`, `tool_round_reviews`
    (both households' reviews of a tool, whose evidence hash daemon B
    fetches) and `review_batches`.

Not covered: anchoring and the DAO's actions through a real DAO proposal
(the governor and the timelock are EOAs here), forfeit of a missed reveal,
REP moved between reviewers of different accuracy (two reviewers per round
are symmetric), daemon-side pruning of blobs (unit tests cover it).

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
