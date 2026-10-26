# Multi-jurisdiction: one daemon, switchable jurisdictions

Decisions ratified with admin on 2026-10-17. This replaces the earlier
draft, which assumed several jurisdictions running at once inside one
daemon; that remains a possible later extension (the design below is that
extension with exactly one slot).

## Decisions

1. **One jurisdiction at a time.** A daemon (one per machine, the existing
   singleton lock) participates in exactly one jurisdiction. Switching
   saves the selection and restarts the daemon. The old jurisdiction's
   substrate data is parked under its own key, not deleted, so switching
   back is a catch-up sync from the last anchor.
2. **Jurisdiction key and gossip topic** = `<chain_id>:<DAO address>`.
3. **Refuse to start without prerequisites.** Every address (substrate,
   rep_token, tool_bond, governor, timelock, service market) is resolved
   from the chosen DAO's Registry. If any is missing, the jurisdiction does
   not start and the daemon says which one. A missing RepToken must never
   silently run the close in the genesis regime.
4. **Epoch rules version.** A Registry key, `jurisdiction.autonet.epoch_rules`,
   bumped only when the epoch computation changes. Feature releases leave
   it alone. A daemon declares the versions it can compute and refuses to
   join a DAO whose version it does not support.
5. **Charter on chain.** The Registry holds the full charter text
   (`jurisdiction.autonet.charter`, JSON: axes with id and thesis, about
   600 bytes) instead of a hash plus URI. Each DAO defines its own axes;
   `correctness` and `simplicity` are mandatory (discovery ranks on them).
   Daemons read the charter pinned to the previous anchor's block, so a
   DAO vote takes effect at the next epoch for everyone. A new axis starts
   at zero for every tool; a removed axis's positions are dropped.
   Jurisdictions are self-contained: nothing compares across them.
6. **What survives a switch.** Tools and service definitions authored by
   this daemon's agents stay. Everything else switches: adopted tools, the
   library, reviews, positions. Nothing is published or listed in the new
   jurisdiction automatically; publishing there is a fresh act (stake in
   that DAO's ATN, bought through its vault if needed; position starts at
   zero). Agents keep their 0x identity; on-chain registration and REP are
   per jurisdiction.
7. **Registration is always owner-bound.** Plain `registerAgent` is
   removed. Every registration carries an EIP-712 binding signature. The
   lineage hash is dropped: the parent link plus the owner binding give
   the provable family tree back to the household (owner 0x).
   - A registered parent agent may sign its child's binding instead of the
     owner wallet. The child lands in the same household, so spawning adds
     no voice.
   - The owner wallet can revoke an agent and its whole subtree with one
     signature.
8. **ε = 0.** A household's mint weight is its REP share, with no floor.
   Newcomers earn as authors when REP holders use their tools, and earn
   REP by staking accurate reviews (decision 10); their own usage counts
   once they hold REP. Closes the fake-owner farming gap.
9. **Merging** = deploy a new DAO with a genesis REP list agreed off-chain,
   then one proposal anchors a posted state snapshot (hash on chain, blob
   served peer to peer, scrutinized beforehand). Members who don't show up
   simply didn't merge. Tokens move by ordinary treasury transfer
   proposals. No absorb mechanism, no vault changes.
10. **Staked reviews (one review type, audits included).** Ordinary reviews
    have paid nothing since v4.1 (2026-07-09) dropped the vet royalty,
    which fake agents could buy. Owner binding plus ε = 0 closes that hole,
    so reviews get paid again:
    - One review type: per-axis scores plus a harm score (below). Audits
      fold in; there is no separate kind.
    - Any stake from zero up, in ATN (buyable, so newcomers can enter;
      accurate reviewers earn ATN and so REP).
    - Commit-reveal, as audits do today, so a round can't be copied.
    - The rating still settles by REP weight only. The stake sets payoff,
      never weight: otherwise a big staker would move the rating toward
      their own score and be "right" by construction.
    - Reward: a governable reviewer share of the tool's mint (default 40%
      reviewers, 60% author). Within it, each reviewer's cut is stake x
      accuracy (closeness to the settled rating, which the close computes
      as credibility) x the review's weight share of the axis mass when it
      landed. Early reviews move the rating most and earn most, and late
      copies of a settled rating earn almost nothing.
    - Inaccurate reviewers lose part of their stake to the accurate ones.
      No new ATN is minted for it. Inside a round, stake settles by
      accuracy alone; the novelty term (1 / (1 + M / REP supply), M = REP
      mass already on the axis) scales only the mint share, where it does
      not cancel out.
    - Settlement is on chain in ToolBond (trustless-contracts, dao's repo).
      ToolBond is the single source of ratings: every review is on chain,
      batched per agent per epoch (one commit, one reveal). The close reads
      per-tool, per-axis means and M from ToolBond pinned to the previous
      anchor. Gossip reviews and the off-chain credibility number go away.
    - Rounds follow epochs, with no REP threshold. Commits during the
      epoch; one reveal window after it (DAO parameter, about a day). A
      missed reveal forfeits the whole stake, so revealing always beats
      withholding, whatever others revealed first. A round with no REP
      weight leaves stakes withdrawable (pulled, never pushed).
    - Harm is a gradient, not a flag: every review carries a harm score
      in [0, 1]. There is only one kind of review; "audit", if a UI wants the
      word, means a heavily staked one, never one with a negative result.
      Harm settles like any
      axis (REP-weighted), counts extra in accuracy (a DAO parameter), and
      the author's bond is slashed in proportion to the settled harm, with
      no dead zone; earnings soften it for tools that are actually used.
      The slashed bond goes to the reviewers who called the harm right.
      The appeal path overrides the settled harm.
    - No "vetted" flag: ratings are a gradient, and daemons, agents and
      owners decide how much REP backing is enough.
    - A missed reveal costs the worst case a revealed review could have
      cost: the whole stake and the full REP penalty, staked or not. So
      withholding never beats revealing, and ATN's price never enters.
      Reveals are relayable (anyone holding the plaintext can submit), so
      a daemon can leave a sealed backup with the owner's other devices.
    - A wrong review costs REP in proportion to how far off it was; REP is
      the reputation, so no separate credibility. This applies to every
      reviewer with REP, staked or not.
11. **Out of scope for this sprint:** the harness add-on (own sprint),
    simultaneous multi-jurisdiction, cross-jurisdiction provenance or
    ratings.

## Merging by genesis (operator steps)

Decision 9, as built. The seed is the new DAO's anchor 0: its payload
carries `"seed": true` and a `carry_cid` for the merged carry bundle
(tool registrations, vetting, positions). It mints nothing (agentMintRoot
zero, agentMintCid empty). Daemons install it once at their next close,
without the usual "observed the previous epoch" check, and replace any
pre-seed local carry state. A seed flag on any later anchor is ignored.

1. Agree the genesis REP list off-chain and deploy the new DAO with it
   (plus its Substrate, with the DAO timelock as governor and no anchors
   yet).
2. Build the snapshot:
   `python scripts/build_genesis_seed.py --chain-id <new chain id>
   --source chain:<rpc>,<substrate A> --source chain:<rpc>,<substrate B>
   --source-blobs <dir holding A and B blobs> --out seed_out
   --blob-store <daemon blob dir>`. Each chain source's latest anchored
   bundle is hash-verified against its chain. Sources are listed in
   priority order: on a conflict the first wins, and the conflict is
   printed. `--source bundle:<file>` takes a local bundle (unverified).
3. Post the blobs: the two files named by their cids in `seed_out`
   (bundle and payload). `--blob-store` puts them in a daemon's store
   (`<world data root>/<rpb>/blobs`), which serves them peer to peer.
4. Propose the anchor: target the new Substrate, calldata from
   `seed_out/seed_anchor.json` (`calldata`, or `anchor_args_ordered`), and
   the exact description `Anchor epoch seed 0x<epochRoot>`. epochRoot is
   sha256 over `autonet:genesis-seed:v1|<chain id>|<carry_cid>`.
5. Members verify before voting: fetch both blobs, check sha256 against
   `carry_cid` and `payloadHash`, check the payload's `carry_cid`, and read
   the printed counts and conflicts (or rerun the builder themselves:
   the bytes are deterministic).
6. Vote. Members who don't take part simply didn't merge. Tokens move by
   ordinary treasury transfer proposals.

Since 2026-10-26 every anchor carries an explicit cutoff, the seed
included: `build_genesis_seed.py --cutoff <unix seconds>` (default: now)
sets it, and it must fall after the new Substrate's `deployedAt`.

## Federation and merging (design, 2026-10-26, not built)

Agreed with the owner. A network can fuse from jurisdictions that started
separately, and split again, entirely through proposals. It builds on
decision 9 (merging by genesis) and the `jurisdiction.<id>.parent` key.

1. **The federation comes first.** A new DAO, the federation, is created
   with its members list, either at origination or later in a mint
   proposal. It can be fully set up (charter, standards, the new autonet)
   before any child joins, so each child can inspect everything before it
   votes.
2. **Joining is a vote in each child.** A child joins by an ordinary
   proposal that sets its `jurisdiction.<id>.parent` to the federation
   and accepts the federation's charter. A child whose charter differs,
   or whose members don't want to join, simply doesn't. Nothing is
   absorbed without the child's own vote.
3. **Federation without a merge is cheap.** Once children name it as
   parent, the federation's standards cascade down (robot standards do
   today). Taxes go up by transfer proposals, or accrue at the
   federation's level. Each child keeps its own autonet: Substrate, ATN,
   epochs.
4. **A full merge is a new autonet.** The federation's Substrate starts
   from a genesis seed built from the children's carry-over bundles
   (`scripts/build_genesis_seed.py` already merges several, with conflict
   rules). From then on the children's agents close epochs there.
5. **Money migrates at a parity.** Each child's ATN converts at a ratio
   set in the federation's Registry (a key per child token, set and
   changed by proposal like any other parity). Households exchange
   themselves, when they choose, through the federation's existing buy
   rail: `AutonetVault.buy(childAtn, amount)` mints federation ATN at the
   Registry parity `jurisdiction.parity.<child ATN address>` and sends the
   child ATN to the federation's treasury. No new contract, no snapshot,
   no forced migration: daemons switch to the federation's contracts, and
   the old token simply stops being used once its holders have exchanged.
   (A federation on another chain would need a real bridge.)
6. **Holdouts stay put.** A household that voted against its child
   joining is not forced across: the child's old Substrate keeps running,
   and whoever stays transacts there.
7. **Splitting is the same path in reverse.** A child leaves by a proposal
   that clears its parent key. A child that merged fully leaves by
   seeding a new autonet from the federation's carry-over, the same way.

Open:
- REP. Each DAO has its own RepToken. The federation's members list sets
  its REP; whether children's REP should map across, and how, is open.
- When a child that merged fully should stop anchoring on its old
  Substrate, or whether it simply winds down as its households leave.
- A split caused by a network partition (a region cut off from the
  chain) is not covered: such a region keeps working locally but cannot
  anchor until it reconnects or seeds a new autonet of its own.

## Backlog

Contracts (must land before the Shadownet redeploy):

- **C1 Substrate registration.** Remove `registerAgent`; drop `lineageHash`
  from storage, `registerAgentBound` and `AgentRegistered`; allow a
  registered parent agent (same owner) to sign the child binding; add an
  owner-signed `revokeSubtree`. Update the packaged ABI and the hardhat tests.
- **C3 (dao)** ToolBond: one staked review type (any stake, commit-reveal,
  per-axis scores plus optional malware flag); stake refunds and losses
  settled from the anchored close; governable reviewer share (default
  4000 bps).
- **C2 (dao)** Registry keys `charter` (text) and `epoch_rules`;
  AutonetParams `{challengeWindow, charter, epochRules}`; genesis REP list.

Daemon:

- **D1 Config from one Registry resolution.** Parse every address
  (including timelock and the DAO Registry) from the chosen DAO; refuse to
  start when one is missing. Remove the dead `discover_jurisdiction` block.
  Stale `0xD5691B7c` references go.
- **D2 Topic and data dir per jurisdiction.** `rpb_address` set explicitly
  to `<chain_id>:<dao>`; world data under the jurisdiction key; one-shot
  move of today's `default` world dir.
- **D3 Epoch rules version.** The daemon's supported set; refuse a DAO
  outside it.
- **D4 Charter from chain.** Read and verify the charter text, pinned to
  the previous anchor; the close, reviews, scorer and drift take the axis
  set as data instead of the fixed 6. New axis = zero, removed axis =
  dropped.
- **D5 Per-jurisdiction records.** Registration records and tool
  `published_in` keyed by jurisdiction, migrated on read; publish requires
  the author registered in the current jurisdiction.
- **D6 Switch flow.** WS `autonet_jurisdictions` (list, current) and
  `autonet_switch` (validate, save, restart); authored tools and service
  definitions kept, everything else parked.
- **D7 Registration flow.** Owner-bound only; parent-signed children;
  revoke subtree. Matches C1.
- **D8 ε = 0** in the close.
- **D10 Staked reviews** (decision 10): reviews carry a stake and go
  through commit-reveal; the close settles each round by REP, pays the
  reviewer share by stake x accuracy x mass share, and emits each
  reviewer's stake refund or loss for ToolBond to honor. Pairs with C3.
- **D9 Carry from a snapshot.** The carry loader takes its source anchor
  explicitly, so a genesis anchor can seed from a posted snapshot.
- **E1 E2E** on two local hardhat DAOs: switch A to B to A; identity kept,
  registration status per DAO, catch-up sync on return, A's mints still
  claimable.

Frontend (atn_web): jurisdiction picker and switch, per-jurisdiction
registration status, owner-bound registration only, revoke subtree.

Elsewhere: the indexer (EightRice/indexer) keys agents by address only; the
same 0x in two Substrates needs a per-Substrate record.
