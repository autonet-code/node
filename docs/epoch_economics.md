# Epoch Economics: Fees-Only Emission & Candle Close

Status (current, beta on testnet): the ATN epoch pool is BURNED SERVICE
FEES ONLY, with no base emission. Zero service volume in a window mints zero
ATN that epoch; conservation holds by construction (Σ minted == Σ burned).
The close computes MONEY ONLY: REP is not minted here, it is claimed
DAO-side (RepToken) on ratified ATN earnings, 1:1. This is "fees-only
emission + REP-from-earnings" (`docs/tool_substrate.md` Decision
2026-07-10, BUILT, sim-validated in
`experiments/econ_attest/sim/results/summary_fees_only.md`), live on the
Etherlink shadownet as of v0.7.0. The fee rail and per-window burned-fee
summing (`read_voice_state` in `nodes/common/voice_state.py`) are the
mechanism. Candle close is IMPLEMENTED for the local daemon path
(2026-06-11) with a chain-derived seed; the federated candle cutoff is
still pending (see below).

> **HISTORICAL NOTE.** The dated `Decision (2026-07-08)` section below
> describes the earlier "fee-recycled emission" model, where the pool was
> `BASE_EMISSION_PER_EPOCH + recycled(N)` (a small clock-based floor plus
> burned fees). That base floor was deleted 2026-07-10: `BASE_EMISSION`
> and the emission-rate decision are gone, `pool(N) = recycled(N)`. The
> "zero floor deadlocks at zero supply" argument in that section no longer
> binds: ATN now enters by PURCHASE (`mintFromVault`, 2026-07-08), so the
> vault, not emission, primes the pump. The window-summing and the
> conservation/wash-proof reasoning carry over unchanged. Reputation
> stopped minting with usage entirely and became a DAO-side pull claim on
> ATN earnings (1:1). The dated sections are preserved as the decision
> ledger; the live description above governs.

## Decision (2026-07-08): fee-recycled emission

Ratified in discussion; replaces "pick an ATN-per-time-unit constant"
as the emission model. The pool at each federated close is

    pool(N) = BASE_EMISSION_PER_EPOCH + recycled(N)
    recycled(N) = Σ ServiceFee.burned in the snapshot anchor's window

so emission tracks real economic activity in the services economy
instead of a clock, with a small clock-based floor.

**The fee rail (Substrate.sol, implemented).** Every `payForService`
takes `SERVICE_FEE_BPS` (250 = 2.5%, PROVISIONAL) of the gross:
`FEE_TREASURY_BPS` (5000 = half, PROVISIONAL) of the fee transfers to
the immutable `treasury` address (set at construction, no admin key
to repoint; the DAO's native revenue rail), and the remainder BURNS
(supply decreases, checkpointed). The recipient receives the net.
`ServiceFee(payer, recipient, amount, burned, toTreasury)` is emitted
per collection.

**The recycling (close side, implemented).** `read_voice_state`
(nodes/common/voice_state.py) sums `ServiceFee.burned` over the
window between the previous two anchors (every fee lands in exactly
one window) and returns `emission_pool = base + recycled`; the
federated-close driver passes it to `federated_epoch_close`, where
`apply_emission_pool` normalizes mints pro-rata. Burn-and-remint is
recycling implemented on the existing mint rail: no new payout
machinery, no change to `recordTrainingForEpoch`.

**Why this shape (the scrutiny that produced it):**

- *Wash-proof by conservation.* Volume-linked PRINTING invites wash
  trading (fake volume is free: money in a circle). Fee-funding
  inverts it: pumping volume pays real fees into a pool you only ever
  share pro-rata: every wash cycle is strictly value-losing.
  (`test_self_payment_still_pays_the_fee` pins the miniature case.)
- *The floor is the faucet, not a salary.* Mint is ATN's only supply
  path; a zero floor would deadlock the economy at zero supply forever
  (no ATN → no service payments → no fees → no pool). BASE
  (`BASE_EMISSION_PER_EPOCH` = 100.0, PROVISIONAL) primes the pump and
  damps the bust side of pro-cyclicality: it pays the same in a slump,
  when building the commons should be cheapest.
- *Doctrine closure.* Paid service demand is the gap map (absorption
  frontier, docs/tool_substrate.md); this makes the commons' funding
  proportional to the size of the map: the market finances its own
  commoditization, mechanically.

Epoch 1 (no anchor yet): no agreed fee window exists, so the pool is
the floor alone. Tests: `tests/test_voice_snapshot.py`
(`TestFeeRecycledEmission`, one-window conservation),
`tests/test_phase7_2_pay_for_service.py` (fee split / burn /
treasury), `tests/test_epoch_emission.py` (the normalizer).

PROVISIONAL parameters awaiting blessing: `SERVICE_FEE_BPS`,
`FEE_TREASURY_BPS` (on-chain constants: changing them is a
redeploy), `BASE_EMISSION_PER_EPOCH` (close-side constant:
consensus-relevant, flag-day to change).

## Fixed emission: the normalizer (implemented)

(This section describes the pool-normalizer MECHANISM
(`apply_emission_pool`) that both the fee-recycled and the current
fees-only decision ride on. Under fees-only (Decision 2026-07-10) the
federated close's pool is `recycled(N)` alone: the burned service fees
in the window, read from the fee rail via the driver's voice-state
refresh (`read_voice_state`), with NO base floor. The
`BASE_EMISSION_PER_EPOCH` term and the `pool = rate × duration` formula
below are RETIRED for the federated path. The `epoch_emission_rate`
config knob still exists and still gates the LOCAL close's clock-based
projection, but the network's authoritative pool is fees-only.)

Problem: `mint(node) = max(0, score_change) × survival` is uncapped:
every claim prints new tokens, supply scales with activity, and junk
dilutes all holders invisibly. Nobody has a concentrated incentive to
debate junk down: my spam doesn't change your payout.

Change: at epoch close, mints are normalized to shares of a fixed pool

    agent_mint[i] = pool × raw_i / Σ raw
    pool          = epoch_emission_rate × epoch_duration_seconds

- `reconcile.apply_emission_pool(result, pool)`: the normalizer.
  Applied AFTER the mint gate, so mint suppressed by debate is
  redistributed to surviving contributors: winning a CON debate
  literally increases the winner's share.
- Volume-triggered or skipped epochs stay fine: emission is anchored
  to TIME, not epoch count. Fast-spinning epochs carry small pots;
  long quiet epochs accrue big ones (countercyclical contribution
  incentive). Spam cannot raise aggregate emission, only contest
  shares within a window, where its victims are present and equipped
  to fight back.
- Config: `epoch_emission_rate` (tokens/second) in autonet.yaml.
  None/unset = legacy uncapped behavior. CONSENSUS-RELEVANT: every
  daemon in a network must flip together.
- Federation caveat (wired but caller-enforced): the local close may
  use the local clock (projection-level, authoritative=False). The
  federated close must derive duration from CANONICAL timestamps
  (anchored block times of the previous close vs this one), never a
  daemon-local clock. `federated_epoch_close(emission_pool=...)`
  takes the pool as an input for exactly this reason.

Tests: `tests/test_epoch_emission.py`.

## Candle close (designed)

Problem: any predictable close time invites last-second junk dumping
into a known settlement window (same pathology as last-second DAO
voting).

Rejected variant, secret close time via commit-reveal: someone holds
the secret (insider snipe), and multi-party XOR reveals have
last-revealer bias (RANDAO problem) with nothing to slash yet.

Chosen design, retroactive cutoff (candle auction, as in Polkadot
parachain auctions):

1. Every epoch runs the full `T_min + window`: no secret exists
   while it's open, so nothing can leak.
2. At `T_max`, randomness `R` (revealed only after the window ends)
   selects `T_cut ∈ [T_min, T_max]`.
3. The close computes over `events where ts ≤ T_cut`, a pure filter
   on the canonical order, identical at every daemon.
4. Events after `T_cut` are NOT dropped: they roll into the next
   epoch (atomic "not served = not paid" holds; they're paid next
   window). Pool for the epoch = rate × (T_cut − T_open); the
   remainder of the window's emission accrues to the next epoch.

Randomness source for `R`:
- Dev/shadownet: `H(prevAnchorHash ‖ epoch_id ‖ blockhash@T_max)`:
  free, verifiable, every daemon computes it. Caveat: an L2 blockhash
  trusts the sequencer.
- Mainnet upgrade path: VRF, or commit-reveal with reveal-or-slash
  once staking exists. Decide before mainnet; the cut function is
  source-agnostic (takes 32 bytes).

Same mechanism applies to DAO proposal voting (votes after the
retroactively drawn cutoff don't count).

Implementation state (2026-06-11):
- Local daemon path DONE: `EpochScheduler` candle mode
  (`candle_min_seconds`/`candle_window_seconds`; `_draw_candle_cut`),
  `WorldService.close_epoch(cutoff_ts=...)` partitions the buffered
  events at the cut, rolls post-cut events into the next epoch's
  buffer, and runs the emission clock to the cutoff so the window
  remainder accrues forward. Local seed: hash of (prev epoch id,
  prev close ts, current epoch id), which is fine single-daemon.
- Chain-derived seed DONE: `nodes/common/candle_seed.py`
  (`ChainCandleSeed`): `sha256(latestAnchorHash@B ‖ hash(B))` where
  `B` = first block with `timestamp ≥ T_max`. Anchor hash is read AS
  OF that block so late-sampling daemons agree even if a new anchor
  lands meanwhile. Wired into `EpochScheduler` via
  `candle_seed_source` (nodes/service.py builds it from blockchain
  config); falls back to the local seed when chain is unreachable.
- PENDING for federation/mainnet: VRF (or commit-reveal w/ slash)
  instead of blockhash; chain-time epoch opens so daemons agree on
  `T_max` exactly (today it's synced wall clocks); and the exact
  cutoff in the federated close (truncate the canonical batch
  sequence at T_cut before replay, so post-cut batches join the next
  epoch's canonical set).

## Merkle mint proofs (implemented 2026-06-11; 2-field leaf as of 2026-07-10)

Mint amounts are no longer self-reported. `submitAnchor` commits
`agentMintRoot`, a merkle root over the epoch's `(agent address,
scaled amount)` map (sorted-pair hashing, double-hashed leaves,
OZ-compatible), and `recordTrainingForEpoch(amount, epochIdHash,
proof)` verifies the caller's leaf against it. Anchors with no
claimable entries carry the zero root; the contract rejects all
nonzero claims against those (`MintRootMissing`).

The leaf is MONEY-ONLY: `keccak256(abi.encode(agent, amount))`, 2
fields. (The v4.1 build briefly used a 3-field `(agent, amount,
repAmount)` leaf; the fees-only decision 2026-07-10 reverted it, since
REP is now a DAO-side function of ratified earnings and nothing about
voice is committed on the close path. See `nodes/common/mint_merkle.py`
and `Substrate.recordTrainingForEpoch`.)

Python side: `nodes/common/mint_merkle.py` (tree/proof/scaling:
both the anchorer and the agent submitter MUST route the float→int
truncation through `scale_mint`, or the leaf is unprovable). Only
address-parsing mint-map keys enter the tree. Production already
satisfies this: the substrate feed resolves every local agent id to
its 0x address at event-authoring time (`_resolve_author` via the
identity resolver wired in `atn/autonet_service.py`), and skips
agents with no on-chain identity. The non-address filter is defense
in depth for test fixtures and legacy logs, not a production gap.

Deployed to the Etherlink shadownet (v0.7.0, 2026-07-10): the merkle
proofs and the money-only leaf are live on the current Substrate
(address of record in `registry.json`, `jurisdictions.autonet.contracts.
substrate`). A consensus-relevant change to the leaf or close output is a
flag-day redeploy: it wipes registrations and anchors, so it is bundled
with a deliberate reset (`scripts/deploy_substrate.js` +
publish_network_config.py + indexer restart per the shadownet runbook).

## Startup replay cost (known debt, separate from economics)

`WorldPersistence.try_restore` replays the full event log through
per-batch equilibration; profiling (py-spy, 2026-06-10) shows the time
goes to pure-Python distance math in the vendored engine
(`coordinate_frame._euclid_sq` via `novelty.measure`). A 3MB log takes
1h+. Constraints on fixes:
- The engine is vendored (`world_model/VENDORED.md`), so fix upstream
  (c:\code\world-model), then re-vendor.
- Replay is consensus-relevant: float summation ORDER matters for the
  bit-identical federated close. Naive numpy vectorization reorders
  sums, so verify bit-equality on a recorded log before adopting.
- Alternatives: exact-value memoization of pairwise distances
  (bit-safe by construction), and a verified world-snapshot
  round-trip (content-addressed node ids may have removed the
  ephemeral-UUID blocker noted in world_persistence.py's docstring;
  verify with serialize→deserialize→score-compare on a real log).
