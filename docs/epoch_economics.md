# Epoch Economics — Fixed Emission & Candle Close

Status: fixed emission IMPLEMENTED (2026-06-10, config-gated); candle
close IMPLEMENTED for the local daemon path (2026-06-11) — federated
candle + chain-derived seed still pending (see below). Local dev config:
100 ATN/day, 2-day epochs + 1-day candle window.

## Fixed emission (implemented)

Problem: `mint(node) = max(0, score_change) × survival` is uncapped —
every claim prints new tokens, supply scales with activity, and junk
dilutes all holders invisibly. Nobody has a concentrated incentive to
debate junk down: my spam doesn't change your payout.

Change: at epoch close, mints are normalized to shares of a fixed pool

    agent_mint[i] = pool × raw_i / Σ raw
    pool          = epoch_emission_rate × epoch_duration_seconds

- `reconcile.apply_emission_pool(result, pool)` — the normalizer.
  Applied AFTER the mint gate, so mint suppressed by debate is
  redistributed to surviving contributors: winning a CON debate
  literally increases the winner's share.
- Volume-triggered or skipped epochs stay fine: emission is anchored
  to TIME, not epoch count. Fast-spinning epochs carry small pots;
  long quiet epochs accrue big ones (countercyclical contribution
  incentive). Spam cannot raise aggregate emission, only contest
  shares within a window — where its victims are present and equipped
  to fight back.
- Config: `epoch_emission_rate` (tokens/second) in autonet.yaml.
  None/unset = legacy uncapped behavior. CONSENSUS-RELEVANT: every
  daemon in a network must flip together.
- Federation caveat (wired but caller-enforced): the local close may
  use the local clock (projection-level, authoritative=False). The
  federated close must derive duration from CANONICAL timestamps —
  anchored block times of the previous close vs this one — never a
  daemon-local clock. `federated_epoch_close(emission_pool=...)`
  takes the pool as an input for exactly this reason.

Tests: `tests/test_epoch_emission.py`.

## Candle close (designed)

Problem: any predictable close time invites last-second junk dumping
into a known settlement window (same pathology as last-second DAO
voting).

Rejected variant — secret close time via commit-reveal: someone holds
the secret (insider snipe), and multi-party XOR reveals have
last-revealer bias (RANDAO problem) with nothing to slash yet.

Chosen design — retroactive cutoff (candle auction, as in Polkadot
parachain auctions):

1. Every epoch runs the full `T_min + window` — no secret exists
   while it's open, so nothing can leak.
2. At `T_max`, randomness `R` (revealed only after the window ends)
   selects `T_cut ∈ [T_min, T_max]`.
3. The close computes over `events where ts ≤ T_cut` — a pure filter
   on the canonical order, identical at every daemon.
4. Events after `T_cut` are NOT dropped — they roll into the next
   epoch (atomic "not served = not paid" holds; they're paid next
   window). Pool for the epoch = rate × (T_cut − T_open); the
   remainder of the window's emission accrues to the next epoch.

Randomness source for `R`:
- Dev/shadownet: `H(prevAnchorHash ‖ epoch_id ‖ blockhash@T_max)` —
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
  prev close ts, current epoch id) — fine single-daemon.
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
  sequence at T_cut before replay — post-cut batches join the next
  epoch's canonical set).

## Pre-mainnet contract change (flagged 2026-06-11)

`recordTrainingForEpoch(amount, ...)` currently accepts a self-reported
amount — the contract checks anchoring + idempotency, not that the
amount matches the anchored agent_mint blob. Any registered agent can
over-report into an anchored epoch. Fix before mainnet: make
`epoch_root` (or a dedicated field in the anchor) a merkle root over
sorted `(agent, amount)` pairs and require a merkle proof in
`recordTrainingForEpoch`. Cheap on-chain (one keccak path), turns the
anchor into an enforced commitment instead of an honor-system record.

## Startup replay cost (known debt, separate from economics)

`WorldPersistence.try_restore` replays the full event log through
per-batch equilibration; profiling (py-spy, 2026-06-10) shows the time
goes to pure-Python distance math in the vendored engine
(`coordinate_frame._euclid_sq` via `novelty.measure`). A 3MB log takes
1h+. Constraints on fixes:
- The engine is vendored (`world_model/VENDORED.md`) — fix upstream
  (c:\code\world-model), then re-vendor.
- Replay is consensus-relevant: float summation ORDER matters for the
  bit-identical federated close. Naive numpy vectorization reorders
  sums — verify bit-equality on a recorded log before adopting.
- Alternatives: exact-value memoization of pairwise distances
  (bit-safe by construction), and a verified world-snapshot
  round-trip (content-addressed node ids may have removed the
  ephemeral-UUID blocker noted in world_persistence.py's docstring —
  verify with serialize→deserialize→score-compare on a real log).
