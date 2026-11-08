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
- **External agents act as the operator** (Decision 2026-11-08): every
  external session's tools, bonds, stakes, reviews and payments come from
  the operator's address and balance. So a household reviews as one batch
  per epoch (no second review of a tool that ToolBond would weigh at zero),
  and the owner funds only the operator: ATN for stakes and bonds, the
  native token for gas. Native agents keep their own addresses and spend
  only their own balances.
- **Reward pools** (decided 2026-11-06, being built): a passive income pool
  on REP balance and a delegate pool paid only to delegates whose weight
  received from others reaches a DAO-set share of all REP (default 1%), each
  with a DAO-set budget. A delegate's pay is budget x received weight / total
  REP, so the budget is a ceiling and the unclaimed rest stays in the treasury. Claiming both
  through a second wallet is possible; each jurisdiction's code of conduct
  and REP burns handle it.
- **A household never earns from its own tools' reviews.** Earning needs
  other households.

## Money (ATN)

- Each jurisdiction's Substrate is its own ATN token, so wallets can tell
  them apart: named `<DAO name> ATN`, symbol `<first 8 characters of the
  REP symbol>ATN` (REP gives REPATN), set at setup. 6 decimals: 1 ATN is
  1e6 base units.
- The tool pool is funded only by **burned service fees**: each
  `payForService` takes a fee (setup default 2.5%, the DAO retunes it
  between 0.5% and 10%), split between the DAO treasury and a burn (setup
  default half each; the DAO retunes the treasury share up to 90%, so at
  least a tenth always burns). Each anchor adds the fees burned between
  the two previous cutoffs to the pool (the mint lags the burns by one
  epoch), and claims are capped on-chain at the cumulative pool
  (`Substrate.mintPoolTotal`), so whatever a close leaves unminted carries
  over to the next epoch's pool. Zero volume, zero mint. Amounts are
  floored to 1e-6 ATN; the dust carries over too. Besides the pool,
  ATN is minted only by the vault (bought, below) and, for forks and
  splits, the genesis claim.
- **Bought, never redeemed.** The vault (`AutonetVault`, DAO side) sells
  newly minted ATN for the tokens in the jurisdiction's value index
  (Registry `jurisdiction.parity.<token>`, a stablecoin by default) at a
  DAO-set parity, uncapped, and sends the payment straight to the treasury.
  A parity edit is an ordinary proposal, not a constitutional one: the
  DAO may need to adjust it often, and turnout is the DAO's job.
  There is no redeem path. So the parity is a ceiling on ATN's price, not a
  peg: nobody pays more on a market than the vault asks, and nothing holds
  the price up from below.
- **ATN is money; REP is the scarce stake.** Supply follows demand at the
  ceiling, which keeps prices of services stable in ATN and gives no reason
  to hold ATN for appreciation. Scarcity, and the stake in the network, is
  REP: soulbound, earned only from the tool pool, and the only voice.
- **Cash-out is the open market, outside Autonet.** Earners (service
  providers, tool authors, reviewers) sell ATN on exchanges or DEX pools
  that anyone can open and anyone can provide liquidity to; nothing in
  Autonet trades ATN for cash. Below the ceiling the market price is ATN's
  value and service prices adjust to it. Keeping trading outside the
  framework, with no redemption and no fixed value promised, keeps the
  legal exposure low. Pools are expected to form wherever there is volume;
  a small jurisdiction's thin pool costs its providers a wider spread.
- **Why providers sell here anyway.** Selling a service through Autonet
  costs less than elsewhere: a fee (2.5% by default; the DAO sets it
  between 0.5% and 10%) with atomic payment and no
  chargebacks (card processing is about 3%), no platform cut (freelance
  marketplaces take 10 to 20%, app stores 15 to 30%), and buyers who find
  the service through the probe without a sales process. That margin is
  what absorbs the cash-out spread; where a pool is so thin its spread
  exceeds it, providers would rather sell elsewhere.
- **Services** (`ServiceMarket.sol`) are whatever needs an advantage that
  can't be copied: hardware (bare metal, GPUs, LLM inference), a human
  credential (a lawyer who passed the bar), data or access, physical
  presence (robots). Anything that is only know-how ends up a tool. They
  earn money only: no mint, no REP, no standing; trust in them is
  behavioral.
- **OPEN:** a legal opinion (EU, MiCA) on the vault sale and on any
  redemption from treasury reserves, which would turn the ceiling into a
  peg; the launch parity; and a monetary model of ATN earned against ATN
  bought, and of the pool depth at which cash-out costs more than the
  channel saves (separate from the consensus design search,
  `autonet-sims/DESIGN_SEARCH.md`).

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

## Tool mint (the only mint rail; Decision 2026-11-07)

- **Paid once, from staked reviews.** Nothing can observe a household's
  private use, so attestations count for nothing and daemons no longer
  gossip them (the local log only drives review reminders). A **bonded, pinned (code)** tool is paid once, in the close where
  its ToolBond review rounds **conclude**; later rounds still rate it but
  pay nothing.
- **Conclusion = the breakthrough gate on reviewers**: the distinct
  reviewing households (the author's own and sanctioned ones excluded) hold
  1% of REP and are 0.5% of REP-holding households. Below it the rounds
  carry over to the next close, faded by `review_decay` per close
  (Registry, default 1 = no fade).
- **Its share of the pool** is the REP behind its reviews: each review
  counts its household's REP share / n, n = the reviews that household made
  in that epoch, so reviewing many tools never multiplies a household's REP.
  Concluding tools split the epoch's pool (carry-over included) by that
  weight.
- **Reviewers** take 40% (ToolBond `reviewerShareBps`) of the tool's share
  **whatever the verdict**, split by accuracy (leave-one-out against the
  round's consensus) x the round's fade, among reviewers whose household
  holds at least `reviewer_min_rep_bps` of REP (Registry; default 0 = any
  REP). REP is a bar, not a multiplier, so it doesn't compound.
- **The author** takes the other 60% x the tool's rating (the mean of its
  REP-weighted charter-axis scores, negatives counted as 0). A composite
  keeps 70% of that and fans 30% over its declared dependencies, up to
  depth 4; cycles, missing or unpayable deps forfeit. What the author
  doesn't get stays unminted and carries over.
- **Excluded**: denied tools and every tool that depends on one, directly or
  further down (it can't be fetched whole; its author isn't slashed for
  it), unbonded tools, and sanctioned households' mint (withheld).
- **Bounty** (ToolBond, on chain): anyone can add ATN to a tool's bounty
  (`addBounty`). As each round finalizes, every settled review takes
  bounty x (its REP / the REP a tool needs to conclude) x its accuracy,
  paid to its household; a household whose REP alone covers the
  requirement can take it all. Sanctioned households get none. It is
  author money, not mint, so it earns no REP and REP may weigh it. The
  author gets the unspent rest back only with the bond.
- **The ATN stake** only moves ATN inside a round: each review loses
  stake x (1 - accuracy) to the round's pool, paid out by stake x accuracy;
  an unrevealed batch loses its whole stake to the treasury. REP moves the
  same way (see Voice). Reviews written before the household registers are
  held by its daemon and submitted to the open round once it does.

## Voice (REP)

- Soulbound. Claimed through `AutonetRepClaim`, 1:1 on cumulative
  **tool-pool** earnings (`agentMintTotal`). Service revenue earns no REP.
- Weighs reviews (by the reviewer's household REP), votes, elections.
- **Moves on reviews** (ToolBond): each review risks `degradeBps` (10%) of the
  household REP it weighed, losing that x (1 - accuracy), reassigned to the
  round's reviews by REP x accuracy; none is created. An unrevealed batch
  loses the full amount per review. A tool settled harmful costs its author's
  household H x `authorSlashBps` (100%) of its REP, which is **burned**, not
  given to the reviewers (a false harm call must not buy voting weight; the
  slashed bond still pays the reviewers who called harm right). So REP is
  earned only from the tool pool, but can be won or lost by reviewing.

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
  withholds the households' mint (not redistributed) and weighs their
  reviews at 0. Burning their farmed REP (`RepToken.burn`, the timelock is
  its admin) makes the proposal constitutional: For from two thirds of all REP.
- Holds below about a third of REP bribed; above half, the honest side
  **forks**.
- **These rules assume a spread of REP.** While a few founders hold most of
  it, the network runs on their goodwill: nothing here stops them. The
  design is judged from the end of that stage (the owner below half of REP).

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
