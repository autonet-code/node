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

# Local end-to-end: services (requests, credits, reviews)

`scripts/local_e2e_service_provider.py` proves services as Decision
2026-11-12 (`docs/services_market.md`) has them, on a real local chain with
two daemons in one process (a seller and a buyer, each a real Runtime; the
seller's WebSocket server is real and the buyer reaches it over its
on-chain wss endpoint):

1. chain up;
2. the seller's operator and a buyer agent registered and funded;
3. the seller sells one of its tools as a request service: a signed spec
   and price list, registered with no registry price; the buyer's catalog
   takes both;
4. a one-off request paid directly (`payForService`); its replay refused;
5. the buyer opens credits; vouchers pay ahead by the voucher size; a
   request the tool fails costs nothing;
6. a new price list with no transaction and no new spec;
7. the seller settles without closing (`ChannelPaid`, the fee burns), and
   the credits keep serving;
8. a top-up, then requests until the credits are spent;
9. the buyer closes; the seller settles what it served inside the window
   without moving the buyer's deadline; the remainder comes back;
10. a review: held by a close pinned before any payment, counted by one
    pinned after, shown in search;
11. teardown.

Sessions need libp2p between daemons: `tests/test_service_session.py` runs
them on real py-libp2p hosts, and the mesh NAT harness
(`scripts/mesh_nat_harness.py`) between NAT'd daemons.

```bash
python scripts/local_e2e_service_provider.py     # 11 stages
```

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
