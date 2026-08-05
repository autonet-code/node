# Resilience: the adversarial equilibrium

Status: living spec, written 2026-07-31. This is the network-level threat
model: why the logical architecture holds its shape under gaming, bad
actors, and collusion, and what happens if the settlement chain itself
misbehaves. It consolidates decisions ratified in `tool_substrate.md`
(the decision ledger — that doc stays authoritative for the mechanisms),
the econ-attestation audit (`experiments/econ_attest/`), and the
2026-07-10 canonical-network migration (the proof the escape hatch works).

**Scope split — this doc vs daemon security.** Two different attackers,
two different readers:

| Layer | Attacker | Defenses | Doc |
|-------|----------|----------|-----|
| Daemon-local | a malicious TOOL or AGENT running on *your* machine | secret vault (fail-closed allowance algebra, host binding, names-only audit), process isolation + vault, `tool_guard.py` adoption containment, tool-bound secrets | `secrets.md`, `cross_platform_isolation_design.md`, `tool_secret_binding.md` |
| Network economic | a malicious PARTICIPANT (sybil rings, colluding reviewers, wash traders) | everything in this doc §2–§4 | here |
| Settlement | a compromised or censoring CHAIN (sequencer, RPC) | deterministic off-chain consensus + consensus-gated migration, §5 | here |
| Code integrity | a TAMPERED daemon posing as an honest node | fingerprint enforcement + peer attestation (designed, pre-implementation) | `anti_tamper_design.md` |

They compose but do not overlap: the vault protects the owner from the
network's code; the equilibrium protects the network from the owner's
greed; migration protects both from the chain.

## 1. The consensus core: determinism, not authority

The load-bearing fact everything below rests on: **network results are
not decided by any authority — they are recomputed identically by every
honest daemon.** `federated_epoch_close(canonical_order(events))` is a
pure function over gossiped events; every honest daemon derives
bit-identical `{agent_mint, tool_positions}`. The chain's role is only
(a) commitment — a rotating submitter anchors the epoch root,
hash-chained via `prevEpochRoot`/`prevAnchorHash` so forks are rejected
on-chain too — and (b) settlement — each agent records its own mint with
a merkle proof against the anchored root (`recordTrainingForEpoch`,
2-field money-only leaf, idempotent per (agent, epoch), no admin keys,
only mint path).

Consequences:

- **A dishonest submitter can't forge an epoch.** An anchor that doesn't
  match what honest daemons computed is simply an anchor nobody can
  prove mints against, and the fork is visible to every peer that
  recomputed the close.
- **A tampered daemon forks itself out.** If it computes a different
  close, its proofs don't verify against the honest anchor; it excludes
  itself by construction. (Making tampering *detectable by peers* before
  it posts anything is the anti-tamper design's job; the economic layer
  is already safe against a lone divergent computer.)
- **State outlives the chain.** Every daemon carries the full substrate
  state; anchors are commitments to state the network already holds.
  This is what makes §5's migration a real option rather than a slogan.

## 2. Money: why the emission can't be pumped

**Fees-only emission (Decision 2026-07-10): the epoch pool is exactly
the fees burned that epoch.** Σ minted == Σ burned, by construction.
This closes the classic emission attacks structurally rather than by
tuning:

- **Wash-pumping the pool is a strict loss.** You pay 100% of the fee to
  reclaim at most a pro-rata slice of the burned half (sim: 0.14%
  reclaimed). There is no subsidy to farm; zero real demand → zero mint,
  and that is correct behavior, not a failure mode.
- **Not served = not paid.** Tasks are atomic; there is no partial
  credit to grind. Service settlement pays min(cumulative voucher,
  deposit) through EIP-712 channels — the theft ceiling is the deposit,
  and the 2.5% fee is taken at settlement through `payForService`
  (closes audit gap G1; no fee-free side rail).
- **Usage counts same-epoch-only.** No retroactive credit from the
  pre-demand dead period (sim: retroactive counting is ×1.6 more
  capturable — rings pre-farm free usage while honest users are idle).

## 3. Voice: why influence can't be cheaply bought or faked

REP (voice) is decoupled from ATN (money) everywhere it matters:

- **REP is a pull claim on *ratified earnings only*, 1:1, DAO-side
  (RepToken).** No close-path mint grants REP; spenders and buyers claim
  nothing ("money can be bought, voice must be earned"). You cannot lie
  about REP because you cannot lie about ATN you provably received.
  This is the D' invariant from the audit: mint must never grant the
  resource that gates mint weight, or sybils escape the cap (the earlier
  β-cap-grants-rep variant measurably leaked, share creep 0.10→0.28 over
  120 epochs; D' flatlines sybil voice share at 0).
- **Sybil identity floods are damped at the household.** Usage credit is
  collapsed to the proven owner wallet (the household) before log1p
  damping; owner-map and wire-key exclusions strip self-calls. Splitting
  one actor across K agents buys nothing the household didn't already
  earn. The residual is priced honestly below.
- **Review/drift pumping needs earned rep.** Position drift weight =
  rep_share × credibility with **no ε floor** — a zero-rep review moves
  nothing. (Contrast the pre-v4.1 sims: sybil ε-reviews pushed a head to
  +0.97 and captured 21× mint at K=100. That surface is gone.)
- **Colluders get docked retroactively and reversibly.** Every close
  re-scores each household's carried review centroid against the tool's
  current drifted head; deviation docks the household's drift weight,
  convergence restores it (symmetric — no stabilization moment for an
  attacker to freeze). Sim-attested: a captured score reverses within
  ~50 epochs of rep-backed usage, and the early capturers dock to the
  floor retroactively.
- **There is no gate to buy.** The v4.1 lesson, worth stating as a
  principle: **every approval gate is a bribery market.** The vet
  greenlight quorum was itself the cheapest attack in the audit (two
  free sybils cleared it), so it was deleted, not hardened. Tools mint
  from first attested use; vets survive as inspection reviews (move
  position, mint nothing); agents see the trust picture (review count,
  rep-weighted axis scores, author rep) and self-select.
- **Burial gates discovery, never use.** Ranking lift is multiplicative
  on topic match (factor in (0,2), never zero): a lone tool in an empty
  niche surfaces regardless of score, and adoption keeps feeding
  rep-backed reviews that can move consensus back. Low-quality spam dies
  by burial, not by a prunable (and therefore attackable) kill switch.
- **Composition farming is capped.** Mint fan-out over the declared-dep
  DAG keeps a 0.7 root share and depth ≤ 4 — a lasagna of trivial
  wrappers can't multiply credit.

**The paradigm behind the sims (2026-07-09):** rep-weighted consensus
*is* a tool's quality, definitionally. There is no hidden ground truth
to deviate from, so the sims measure **consensus-capture cost** — the
rep share needed to move a score against the live review flow — not
attack "success." Security here means capture is expensive, visible,
and reversible, with the correction channel (rep-backed reviews from
real usage) never closable.

## 4. Honest limits: what is priced, not prevented

Deliberately documented — a threat model that claims completeness is
lying:

- **Unlinked-wallet wash trading buys REP at ~the fee rate.**
  Self-dealing across two unlinked wallets is undetectable in principle
  (wallets are free; household collapse can't hold). A washer pays the
  2.5% fee per cycle and claims REP on the rest: voice at roughly 2.5
  cents/REP. Mitigations are economic, not detective: the fee is the
  price lever; genesis REP seeding multiplies the attacker's bill while
  the network is young (and dilutes automatically as real earnings mint
  REP — no handoff ceremony); and every washed cent funds the treasury
  and the honest pool. Fee value: OPEN pending the wash-cost vs
  honest-volume elasticity sweep.
- **Remote services are unknowable in principle.** Remote execution
  can't be verified, so services get NO substrate standing, mint, or
  verdict-layer claims — trust is behavioral (identity, atomic payment,
  track record), and the blast radius of a bad provider is the channel
  deposit. Keeping services *outside* the trust substrate is itself the
  defense: the unverifiable rail can't contaminate the verifiable one.
- **A machine-owner attacker can't be stopped by self-checks.** No
  software self-attestation survives the owner patching the checker. The
  bar (per `anti_tamper_design.md`) is: tampering detectable by honest
  peers, self-correcting for accidental corruption, and not
  circumventable via the cheap config-repoint path. Peer-verified
  attestation is designed, pre-implementation.
- **Dormant teeth, kept in-tree:** the violator-pays mint gate
  (`mint_gate.py`, `apply_gate=False`) and the CON-bust/defender loop
  are deliberately parked, not deleted — they can be armed forward-only
  through governance if the priced exposures above turn out underpriced.

## 5. The settlement layer: censorship, capture, and the migration hatch

The chain (Etherlink shadownet today) is a rollup with a sequencer. The
threat model there is honest: a sequencer can **censor or reorder**; it
cannot **forge**, because mints need merkle proofs against anchors every
honest daemon already ratified, and Substrate.sol has no admin keys.
That leaves two failure modes, both covered:

**Silence → halt.** Work halts if the governance heartbeat is missed.
This is a hard constraint, not a courtesy: a network that can't reach
its governance surface fails *closed*, so a censoring chain can pause
the economy but cannot redirect it.

**Capture → consensus-gated migration.** Evolution is forward-only:
there is no rollback mechanism, and there is no need for one, because
the chain is replaceable and the state is not on it. If the settlement
chain is compromised, censoring, or simply outgrown, governance
redeploys the contract suite onto a fresh Substrate (clean genesis) and
daemons re-register and re-anchor — the substrate state, tool library,
and earnings history travel with the daemons, and REP claims re-derive
from ratified earnings.

This is not hypothetical. **It has been executed, live, twice in three
days** (2026-07-08 genesis-v2, then 2026-07-10 when the canonical
network flipped to the platform-born jurisdiction): the werule-created
DAO had the autonet stack attached via a 2-call governance proposal
(setMinter + parity), the charter was re-anchored under the new
timelock, `registry.json` (the address-of-record daemons resolve the
network from) was flipped, and daemons re-registered against the new
Substrate. The same choreography — governance proposal → fresh suite →
registry flip → daemon re-registration — is the escape hatch for a
hostile chain, with one addition: the destination chain would differ,
and the last honest anchor is the genesis reference.

Two design consequences worth making explicit:

- **The anchor is the recovery point.** Because anchors are hash-chained
  commitments to state every daemon holds, "the last anchor all honest
  daemons agree on" is a well-defined migration genesis even if the
  chain's tail is censored or reorged.
- **Cheap external insurance is available if wanted (OPEN, unbuilt):**
  checkpointing the epoch anchor hash to Ethereum L1 every N epochs —
  one tiny transaction, no per-operation cost — would put the recovery
  point on a maximally neutral chain without running the economy there.
  Running *live* on multiple chains was considered and rejected: ATN has
  no trusted multi-chain issuer (the thing that makes USDC multichain is
  Circle), so multi-chain state means a bridge, and a bridge is a worse
  trust assumption than the sequencer it would hedge.

**Flag-day discipline is part of the same story:** changes to the close
output/CID, contract ABI, or merkle leaf shape require every daemon on
the new build and a fresh Substrate before the next close — otherwise
closes and proofs fork. Migration is cheap *because* it is total; the
network never runs two truths at once.

## 6. The equilibrium, in one paragraph

Money is demand-backed (mint == burned fees), voice is work-backed (REP
== ratified earnings), quality is consensus-backed (rep-weighted drift
with credibility memory), and truth is computation-backed (every honest
daemon recomputes the close). Each rail is denominated in something the
attacker must actually spend — fees, provable service delivery, earned
reputation, canonical code — and the one resource that is free to forge
(identities) is collapsed to households and carries zero weight until it
earns some. What can't be prevented is priced, what can't be priced is
kept off the trust substrate, and if the settlement layer itself turns
hostile, the network halts, votes, and walks — carrying its state with
it, forward-only.
