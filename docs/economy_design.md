# Economy design: the current state (2026-10-28)

One page, kept current. Where this disagrees with older text elsewhere,
this wins; the reasoning behind each rule is in `tool_substrate.md`
(Decision sections), `resilience.md` and the GPU sim report
(`C:\code\autonet-sims\REPORT.md`, local only). Every number below is a
default the DAO can change; changing any rule needs the owner's sign-off.

## Who is who

- **Household** = an owner wallet. Its ATN and REP are claimed to it (the
  daemon's operator key submits the claims; `claimFor` is permissionless,
  and an agent bound after a close's owner snapshot is paid at its own
  address). Agents and the daemon's operator key are bound to it; the
  operator proposes and votes with REP the owner delegates to it. No automatic delegation; a delegate
  who doesn't vote loses delegators, not REP.
- **Use inside one household mints nothing.** Earning needs other
  households.

## Money (ATN)

- Each jurisdiction's Substrate is its own ATN token, so wallets can tell
  them apart: named `<DAO name> ATN`, symbol `<first 8 characters of the
  REP symbol>ATN` (REP gives REPATN), set at setup. 6 decimals: 1 ATN is
  1e6 base units.
- The tool pool is funded only by **burned service fees**: each
  `payForService` takes a 2.5% fee, half to the DAO treasury and half
  burned, and an epoch's mint is capped on-chain at the fees burned between
  the two previous cutoffs (the mint lags the burns by one epoch). Zero
  volume, zero mint. The pool is shared in proportion to each tool's usage
  score, floored to 1e-6 ATN (the dust is never minted). Besides the pool,
  ATN is minted only by the vault (bought, below) and, for forks and
  splits, the genesis claim.
- Bought with a stablecoin through the vault at a governed parity.
- **Services** (remote APIs, `ServiceMarket.sol`) earn money only: no
  mint, no REP, no standing.

## Epochs

- An epoch ends when burned fees reach `epoch_target_share` of ATN supply,
  between a floor (the DAO's voting delay + voting period + timelock delay)
  and a ceiling (`epoch_ceiling_votes`, 10, voting periods): more activity,
  shorter epochs. Unset (0, unless the deployment sets it), every epoch
  lasts the floor. Daemons close `epoch_grace` (120 s) after a cutoff.
- Every daemon with a registered agent or operator computes the close from
  gossiped events, bit-identically. A daemon without one closes nothing: it
  follows the anchored state.
  The anchor is an AnchorElection: drawn agents register closes, REP
  votes, the winner is anchored.
- A silent chain pauses settlement only; work continues.

## Tool mint (the only mint rail)

- Usage = **signed receipts** (agent signature, gossip domain) of uses
  from other households, **bonded, pinned (code) tools only**, damped per
  household (log1p) and weighted by the household's REP share: a household
  without REP adds nothing. Uses on the author's own daemon don't count, and
  the owner wallet has no agent key, so its own uses sign no receipts.
- **Composition**: a composite keeps 70% and fans 30% out over its declared
  dependencies, up to depth 4. Cycles and missing deps forfeit.
- **Breakthrough gate**: a tool mints only once its users hold 1% of REP
  and are 0.5% of REP-holding households. No escrow: nothing is earned
  before. The bar rises with the network.
- **Excluded**: denied tools, and usage by sanctioned households.
- **Reviewer share** (40% of each tool's mint by default, ToolBond
  `reviewerShareBps`): paid by review accuracy (leave-one-out against the
  consensus) times novelty (less-reviewed tools pay more), not by stake;
  a review needs a stake above zero to be paid at all (any amount), and a
  household with no REP can't commit without one. Reviews written before
  the household registers are held by its daemon and submitted to the open
  round once it does. The ATN stake is only the bet: each review loses
  stake x (1 - accuracy) to the round's pool, paid out by stake x accuracy;
  an unrevealed batch loses its whole stake to the treasury. REP moves the
  same way (see Voice).

## Voice (REP)

- Soulbound. Claimed through `AutonetRepClaim`, 1:1 on cumulative
  **tool-pool** earnings (`agentMintTotal`). Service revenue earns no REP.
- Weighs reviews (by the reviewer's household REP), votes, elections.
- **Moves on reviews** (ToolBond): each review risks `degradeBps` (10%) of the
  household REP it weighed, losing that x (1 - accuracy), reassigned to the
  round's reviews by REP x accuracy; none is created. An unrevealed batch
  loses the full amount per review. A tool settled harmful costs its author's
  household H x `authorSlashBps` (100%) of its REP. So REP is earned only from
  the tool pool, but can be won or lost by reviewing.

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
  withholds the households' mint (not redistributed), weighs their reviews
  at 0 and burns their farmed REP (`RepToken.burn`, the timelock is its
  admin).
- Holds below about a third of REP bribed; above half, the honest side
  **forks**.

## Splits, reunions, forks (`resilience.md` 5.1)

- A new jurisdiction's genesis carries REP (deploy member list) and ATN
  (`Substrate.claimGenesis`, merkle root from `scripts/build_atn_genesis.py`).
- These are procedures, not yet code beyond the genesis builder
  (`scripts/build_atn_genesis.py` copies current ATN and REP balances minus
  an exclusion list). Split: REP re-derived from everyone's earnings.
  Reunion: both sides' earnings count. Fork: genesis minus every address
  that voted against the sanction.

## Launch

- Mainnet starts with a **virgin mint**: nothing from test networks
  carries over. The WAN run's tools are re-published as a baseline
  library from the owner's household.
