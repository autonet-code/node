# Economy design: the current state (2026-10-28)

One page, kept current. Where this disagrees with older text elsewhere,
this wins; the reasoning behind each rule is in `tool_substrate.md`
(Decision sections), `resilience.md` and the GPU sim report
(`experiments/econ_gpu/REPORT.md`, local only). Every number below is a
default the DAO can change; changing any rule needs the owner's sign-off.

## Who is who

- **Household** = an owner wallet. It alone claims ATN and REP. Agents and
  the daemon's operator key are bound to it; the operator proposes and votes
  with REP the owner delegates to it. No automatic delegation; a delegate
  who doesn't vote loses delegators, not REP.
- **Use inside one household mints nothing.** Earning needs other
  households.

## Money (ATN)

- Minted only from **burned service fees** (2.5% of each `payForService`).
  Zero volume, zero mint. Each epoch's mint is capped on-chain at the fees
  burned in the previous epoch.
- Bought with a stablecoin through the vault at a governed parity.
- **Services** (remote APIs, `ServiceMarket.sol`) earn money only: no
  mint, no REP, no standing.

## Epochs

- An epoch ends when burned fees reach `epoch_target_share` of ATN supply
  (floor 120 s): more activity, shorter epochs.
- Every daemon computes the close from gossiped events, bit-identically.
  The anchor is an AnchorElection: drawn agents register closes, REP
  votes, the winner is anchored.
- A silent chain pauses settlement only; work continues.

## Tool mint (the only mint rail)

- Usage = **signed receipts** (agent signature, gossip domain) of uses
  from other households, **bonded tools only**, damped per household
  (log1p).
- **Composition**: a composite keeps 70% and fans 30% out over its declared
  dependencies, up to depth 4. Cycles and missing deps forfeit.
- **Breakthrough gate**: a tool mints only once its users hold 1% of REP
  and are 0.5% of REP-holding households. No escrow: nothing is earned
  before. The bar rises with the network.
- **Excluded**: denied tools, and usage by sanctioned households.
- **Reviewer share**: paid by review accuracy (leave-one-out against the
  consensus), not by stake. Stake is only the bet.

## Voice (REP)

- Soulbound. Claimed through `AutonetRepClaim`, 1:1 on cumulative
  **tool-pool** earnings (`agentMintTotal`). Service revenue earns no REP.
- Weighs reviews (by the reviewer's household REP), votes, elections.

## Tools

- Manifest is the tool (content-addressed; the author is inside it).
- **Novelty claim required to publish**: what is new and the closest
  existing tools. Reviewers score it on correctness, so copies and thin
  wrappers rate low.
- **ToolBond**: an author stake behind every tool; reviews are
  commit-reveal, weighted by household REP; slashes capped.
- Ratings (correctness, simplicity) rank discovery multiplicatively; low
  ratings bury, nothing is pruned.

## Containment

- Per tool: **denied / strict / declared**, from settled harm reviews
  (including harm still under appeal) against REP supply.
- Defaults: deny at harm ≥ 50/100, declared below 10, both only once the
  reviewers hold **0.1% of REP** (`containment_trust_mass_bps` 10;
  ToolBond `harmQuorumBps` 10). The DAO must lower it as the network
  grows (about 0.02% at 100k households).

## Attacks and sanctions

- Detection is off-chain and per node (nodes notice, report, coordinate).
  It ends attacks; the deterministic rules only price and slow them.
- A **sanction** is an ordinary governor proposal at its quorum: it
  withholds the households' mint and burns their farmed REP
  (`RepToken.burn`, the timelock is its admin).
- Holds below about a third of REP bribed; above half, the honest side
  **forks**.

## Splits, reunions, forks (`resilience.md` 5.1)

- A new jurisdiction's genesis carries REP (deploy member list) and ATN
  (`Substrate.claimGenesis`, merkle root from `scripts/build_atn_genesis.py`).
- Split: REP re-derived from everyone's earnings. Reunion: both sides'
  earnings count. Fork: genesis minus every address that voted against the
  sanction.

## Launch

- Mainnet starts with a **virgin mint**: nothing from test networks
  carries over. The WAN run's tools are re-published as a baseline
  library from the owner's household.
