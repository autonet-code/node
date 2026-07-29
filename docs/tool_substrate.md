# Tool substrate v3: tools as the ONLY substrate item; reviews replace debates

Status (current, beta on testnet, live on the Etherlink shadownet as of
v0.7.0): FEES-ONLY EMISSION + REP-FROM-EARNINGS, ratified + BUILT
2026-07-10 (see the `Decision (2026-07-10)` section, the current design of
record). It amends v4.1 in three places: (1) the ATN epoch pool = BURNED
SERVICE FEES ONLY (no base emission, so zero volume mints zero ATN,
Σ minted == Σ burned); (2) the close computes MONEY ONLY: the v4.1
supply-pegged β cap and the rep/ATN mint split are DELETED; (3) REP is no
longer minted on the close path: it is a pure DAO-side pull claim
(RepToken, 1:1) on ratified ATN earnings, and the review weighting reads
RepToken checkpoints. Substrate.sol is now a PURE MONEY contract. The
authoritative payload is schema 3 with a 2-field `(agent, amount)` leaf.

The prior `Decision (2026-07-09)` v4.1 section retired the v3 vetting GATE
(tools mint from first attested use) and introduced continuous
reversal-aware credibility + rep-weighted drift (no ε floor), and those
SURVIVE. Its rep/ATN mint split and β cap do NOT (deleted 2026-07-10). The
v3 body below is retained for the parts still live (reviews, position
drift, composition, adoption, Services split) with inline markers where a
rule later changed.

v3 status note (2026-07-08, historical): ratified in discussion (see the
Decision section below). Superseded v2 (2026-07-04) in four places: mint is
usage-only, attestations carry per-axis review scores that drift the
tool's charter position at close, work units leave consensus, and the
CON/PRO debate machinery retires from the live path. Everything else
(trust classes, vetting, composition, adoption, Services split) stands.
v2 note kept for history: the **attested trust class and standing decay
are retired** from the substrate; remote/endpoint-backed offerings moved
to the Services market (`docs/services_market.md`). Tools and Services
are separate economies unified only at the agent's interface (one
inference probe, one MCP-shaped surface).

## The line: verifiability

The substrate's verdict layer only holds items whose behavior is
**locally verifiable**: pinned code can be re-run by anyone; a remote
endpoint cannot (its behavior is unknowable in principle), so it never
enters the verdict layer.

PHASE-10 AMENDMENT (2026-07-05, docs/phase10_results.md): the stronger
motivating claim that used to live here ("executable ground truth
makes debate decisively better than prose debate", H1) was REFUTED
by its pre-registered bar and, per the pre-commitment, no longer
motivates the design. Tool mint launches gated on vetting + the
damper alone. What the measurement DID show (exploratory, not
re-litigation): evidence-backed standing separated defective from
correct tools perfectly (AUC 1.000 in every sweep cell, deterministic
under sybil flood where text ranking leaks), and it is the only arm
that protects falsely-accused correct tools, so replayable CON
evidence remains a worthwhile rail on its own merits; it just may not
be SOLD as the thing prose debate lacked, because well-priored text
debate ranked nearly as well.

## Decision (2026-07-08): substrate v3, reviews replace debates

Ratified in discussion. The v2 refactor stopped halfway: work units
still rode full consensus while earning nothing, tool charter positions
were static, mint was scaled by debate standing nobody engaged with,
and ranking read claim standing. v3 completes the paradigm:

1. **Manifest defines the tool.** Its embedding sets the topical
   position (embedding tail); the 6-dim charter head enters at ZERO
   (neutral): a tool EARNS its alignment/usefulness position, the
   author never claims one.
2. **Reviews, not debates.** The agentic loop's post-use attestation is
   a self-report on the agent's OWN usage: there is no opponent, so
   "debate" was the wrong name. Attestations may carry per-charter-axis
   signed scores in [-1, +1].
3. **Reviews rank discovery.** Library retrieval ranks best-reviewed
   tools first (usefulness axes of the drifted head lift the cosine).
4. **Usage alone mints.** `mint = usage_term`, the damped, exclusion-
   filtered attested-usage mass. No standing multiplier, no violator
   gate on the tool rail. Reviews affect earnings only indirectly, by
   steering future usage through ranking. **[SUPERSEDED by v4.1: mint is
   still usage-only but now rep/ATN-SPLIT under a supply-pegged β cap:
   zero-rep usage mints ATN, not reputation. See Decision 2026-07-09.]**
5. **Position drifts at close** as the mint-weighted running centroid
   of review axis scores:
   `head' = (mass·head + axis_mass·axis_mean) / (mass + axis_mass)`,
   `mass' = mass + axis_mass`, where `axis_mass` is the same per-caller
   log1p-damped, exclusion-filtered evidence mass that prices mint,
   restricted to axes-bearing attestations (axis-less usage mints but
   does not move position). Prior: zero head, mass = 1.0 damped unit
   ("the author counts as one damped attestation"). No free drift-rate
   parameter; heavily-used tools have proportional inertia.
6. **No pruning.** Low-rated tools die by ranking burial: not
   retrieved → not pulled → not used → not paid. The carry-over map
   grows monotonically; mechanical GC (drop digests with zero usage for
   N epochs, re-registration re-admits) is a someday note, not design.
7. **Work units leave consensus.** The conversation/capsule feed writes
   to the daemon-local ArtifactIndex only (retrieval feedstock); no
   sprouts, no gossip, no close participation. Distillation into a
   published tool is the sanctioned way experience enters consensus.
8. **Debate machinery retires from the live path**: `submit_con` /
   `submit_support` and the violator-pays gate on the tool rail are
   removed; `mint_gate.py` stays in-tree, dormant. The CON-triggered
   bust of a greenlit tool goes dormant with it.

**Known-open risk (accepted at v3; ADDRESSED at v4.1):** reviews drive
both ranking and position; a sybil ring of callers could pump a tool.
v3's standing defenses were the vetting entry gate + log1p damping +
owner/wire exclusions + attestation cost. **v4.1 (Decision 2026-07-09)
replaces the entry gate with forge-resistant scores:** drift weight =
rep_share × credibility with NO ε floor (zero-rep reviews move nothing),
continuous reversal-aware credibility docking, and the supply-pegged β
cap on zero-rep ATN mint. The econ-attestation sims (`experiments/
econ_attest/`) confirmed the v3 gate itself was the cheap attack (two
free sybils cleared `VET_QUORUM`), which is why it is retired rather than
hardened. Covert harm invisible to satisfied users is now surfaced (not
gated) by the trust picture at search time.

## Decision (2026-07-08, addendum): reputation-weighted voice

**[VOICE SOURCE SUPERSEDED SAME DAY: this addendum was ratified as
"balance-weighted voice" (household_ATN/supply), then superseded that
evening by "ATN = money, reputation = voice": voice weight reads
SOULBOUND REPUTATION, not ATN balance. The mechanism (household collapse,
linear weight, ε floor, snapshot-pinned reads) is unchanged; only the
SOURCE moved from balance to reputation. This section is corrected inline
below (was divergence D1). v4.1 (Decision 2026-07-09) then adds: reviews
are additionally CREDIBILITY-weighted, and the ε floor is dropped from
DRIFT weight, surviving only on the MINT side.]**

Ratified in discussion, same day as v3, as the answer to the sybil
trade above. Premise accepted first, **usage is unverifiable**:
attestations are self-reported and nothing stops a custom framework
from fabricating them. So the defense doesn't verify activity, it
prices the identity behind it. Wallets are free; the only scarce,
verifiable anchor in-system is ATN itself.

1. **The household is the economic unit.** A caller's household is its
   proven owner wallet (Substrate.sol's EIP-712 owner binding, where the
   owner signs at registration, so households can't be fabricated);
   unbound callers stand as their own household. Registering an agent
   is what lends it the owner's voice.
2. **Collapse before damping.** Usage counts and review cells pool per
   household BEFORE log1p: N co-owned agents are ONE voice
   (`log1p(Σ counts)`, not N log1p terms). This closes the per-agent
   amplification the old per-caller damper allowed. The author-house
   comparison subsumes the self- and same-owner exclusions; the wire
   dedup applies to the household's pooled sender keys.
3. **Voice weight = ε + household_REPUTATION / rep_supply** (corrected
   from balance; divergence D1), where household_reputation is the sum
   of soulbound reputation over every agent bound to the owner
   (reputation is minted only by `recordTrainingForEpoch`, stays on the
   agent address; a bare owner wallet never trains and carries zero, so
   there is no owner-wallet term; see `voice_state.py`). **LINEAR in
   reputation by design**: linearity is splitting-invariance, so dividing
   earnings across any number of wallets or agents never gains weight.
   Resist any future urge to damp it (log/caps reintroduce a splitting
   advantage). **[v4.1: this ε-floored weight is the MINT-side voice;
   drift weight uses the raw rep_share × credibility with NO ε floor.
   See Decision 2026-07-09.]**
4. **ε (`VOICE_EPSILON`, provisional 0.05) is the floor** for unknown
   or zero-balance households: it bounds what a throwaway identity can
   contribute AND bootstraps a cold-start network (supply = 0 → every
   voice = ε, i.e. uniform). The ε floor is the residual sybil surface:
   damage is bounded at ε per fabricated identity while the real
   economy outgrows it.
5. **One weight, both rails.** The same household weight multiplies
   the damped usage term (mint) and the review evidence mass (position
   drift): a voice that can't mint can't move position either.
   **[SPLIT at v4.1: the two rails use DIFFERENT weights. Mint keeps
   the ε-floored weight, drift uses raw rep_share × credibility with no ε
   floor. A zero-rep voice can still mint ATN but moves no position.
   Decision 2026-07-09. The v4.1 β cap on the mint-side weight was DELETED
   2026-07-10 (fees-only); mint is the plain ε-floored weight now.]**
6. **Snapshot-pinned reads, checkpoint-served.** `voice_weights` (and
   the owner map read with it) is a close input refreshed by the
   driver's `voice_source` hook just before each close
   (`nodes/common/voice_state.py`), with every input derived AS OF the
   previous epoch's anchor block
   (`getAnchor(anchorCount-1).blockNumber`, stored on-chain at
   submission). The snapshot is on-chain, agreed, and pre-dates the
   epoch, so all daemons derive identical maps no matter when their
   refresh fires, and a wallet funded mid-epoch (after seeing what's
   worth pumping) carries no weight until the next epoch. NO ARCHIVE
   NODE NEEDED. **[REP SOURCE CORRECTED 2026-07-10: the read moved off
   Substrate's ATN checkpoints onto RepToken (DAO). RepToken is ERC20Votes
   in TIMESTAMP mode, so `voice_state.read_voice_state` pins on the
   anchor's `timestamp` (not block number) and reads
   `getPastVotes(addr, ts)` / `getPastTotalSupply(ts)`, so voice == voting
   power, the correct governance semantic. Substrate.sol's own reputation
   surface is DELETED (pure money contract). The original 2026-07-08 text
   below describing Substrate's `balanceOfAt`/`atnTotalSupplyAt` IVotes
   checkpoints is the balance-era mechanism, kept as history.]** The
   original text: Substrate.sol's ATN carries IVotes-MECHANISM
   checkpoints (`Checkpoints.Trace208` history pushed on every
   mint/transfer; `balanceOfAt` / `atnTotalSupplyAt` served from
   current state, deliberately WITHOUT the delegation layer, because
   `getPastVotes` semantics return 0 for undelegated accounts and
   would mute every wallet by default), and the agent set + owner map
   derive from `AgentRegistered`/`OwnerBound` event logs up to the
   snapshot block (last binding per agent wins). No anchor yet = no
   agreed snapshot: empty maps, close runs `weights=None` (uniform
   1.0, correct for epoch 1, nothing has minted). Chain tests:
   `tests/test_voice_snapshot.py` (incl. the pin property: a transfer
   after the anchor does not change the epoch's weights).

Character shift accepted openly: capital = voice. Holders are the
actors with the most to lose from junk mint debasing ATN, and every
legitimate agent has a funded owner by construction ("AI can only
execute if tokens are spent"), so the mute set is precisely the
throwaway wallets. What this deliberately does NOT do: charge for tool
use (tools stay free), verify usage (impossible), or prune anything.

## Decision (2026-07-09): v4.1 gradient trust, the gate becomes a gradient

Ratified in discussion ("ok let's go for it") after the econ-attestation
audit (`experiments/econ_attest/`: attacks.md, game_model.md,
attestation.md, sim/). The audit ran the REAL close code over adversarial
populations and confirmed four cheap attacks the v3 design admitted at
scale: vetting collusion (two free unbound sybils clear `VET_QUORUM=2.0`,
the bust dormant → zero-cost greenlighting), an ε-faucet (K dust
identities skim the pool linearly, 0.67 share at K=200), drift pumping
(sybil ε-reviews pushed a head to +0.97, 21× mint capture at K=100; rank
is far cheaper to pump than mint), and discovery spam (unreviewed manifests
burial can't grip). v4.1 dissolves the gate into a gradient and moves the
security surface onto forge-resistant scores. SIM-VALIDATED and BUILT
(close + contract + surfaces); evidence trail:
`experiments/econ_attest/sim/results/summary_v4_1.md`.

The paradigm statement (user, 2026-07-09), because it reframes what the
sims even measure: **the rep-weighted consensus IS a tool's quality,
definitionally.** There is no "majority attack" and no "honest minority":
adversarial sims measure CONSENSUS-CAPTURE COST (rep-share needed to
move a score against the live review flow), not deviation from a hidden
truth. Correction channel: **burial gates DISCOVERY, never use.** The
ranking lift is MULTIPLICATIVE on topic match (`base·(1+tanh(rating))`,
factor in (0,2), never zero), so a lone tool in an empty semantic niche
surfaces regardless of score (novelty buys discovery, quality keeps it,
sim-attested), and adoption/delegation usage keeps feeding rep-backed
reviews that can move consensus back.

The changes, each superseding a v3 rule:

1. **The vet GATE is retired.** Tools mint from first attested use: no
   greenlight quorum, no vet royalty, no candidate pool. Rationale: a vet
   is an unverifiable claim anyway, and the collusion gap only exists
   because there is a gate to buy; risk-averse owners/agents supply their
   own inspection incentive. (Supersedes the Vetting section's greenlight
   gate and the Mint section's "greenlight-gated, royalty-split.")
2. **Vets become inspection reviews.** One rail. A vet survives as a
   per-axis review carrying NO usage receipt: it moves the tool's
   position (drift) and mints nothing (`vet_axis_reviews_by_caller`,
   merged with usage reviews at drift time). This gives burial grip on
   unreviewed spam.
3. **Search shows the trust picture** (review count, rep-weighted axis
   scores, author rep) and agents self-select: trust well-scored, avoid
   bad-scored, inspect the unreviewed. No gate decides for them.
4. **Scores are now the SECURITY surface, so they are forge-resistant:**
   - **Drift weight = rep_share × credibility, with NO ε floor.**
     Zero-rep households carry ZERO drift weight (the ε floor stays on
     MINT, for bootstrap; it is gone from drift). Author prior mass 1.0.
     In the local/genesis regime (no chain rep) drift keeps weight 1.0,
     the pre-v4.1 behavior.
   - **Rep/ATN mint decoupling (D').** Zero-rep-weighted usage mints ATN
     but grants NO reputation: an author's reputation increment is only
     the portion of usage_term attributable to rep-holding callers
     (`rep_fraction`, threaded out as `node_agent_rep` → `agent_rep`).
     Mint must not grant the resource that gates mint weight, or sybils
     escape the cap (the v4 β-cap-grants-rep version LEAKED, share creep
     0.10→0.28/120 epochs; D' flatlines sybil VOICE share at 0).
   - **Supply-pegged β cap on zero-rep ATN mint weight.**
     `β(S) = max(BETA_MIN, exp(−S/BETA_S0))`, so β≈1 at genesis (newcomer
     demand prices honest work), decaying toward `BETA_MIN` as reputation
     supply matures. The aggregate ε-weight of zero-rep households is
     capped at β of total mint weight (uniform pro-rata scale-down when
     it binds; pure-zero-rep genesis stays uncapped so the economy can
     bootstrap). β is ATN-side only: D' already keeps it off voice.
     A dust ring cannot TIGHTEN β against honest users: under D' its mint
     grants no rep, so it never moves supply in the attack direction
     (sim: worst tightening 0.0). Parameters: `BETA_MIN=0.05`,
     `BETA_S0=5000.0` (50 epochs × 100 pool, the sim's engineering pick,
     near-zero honest distortion; strict-≤5%-skim alternative is S₀=10 at
     ~0.31 early distortion). **Recalibrate S₀ to the real epoch cadence
     at launch** (the 300-epoch sim horizon is short vs a real network).
   - **Continuous reversal-aware credibility.** Every close re-scores each
     household's carried review centroid against the tool's CURRENT
     (just-drifted) head, on tools with review mass ≥ `CRED_MASS_FLOOR`:
     deviation > `CRED_DELTA` docks the household's drift-weight
     multiplier, convergence restores it (symmetric, no stabilization
     moment for an attacker to freeze). Multiplies DRIFT weight only,
     never mint; the rep token is never burned, because credibility is carried
     close state (rebuildable cache, same contract as `tool_positions`),
     alongside the carried per-household `tool_review_book`. Parameters
     (SEEDED, post-launch tunable): `CRED_DELTA=0.7` (EMERGED, honest
     false-positive dock 0.02%), `CRED_MASS_FLOOR=3.0`, `CRED_FLOOR=0.1`,
     `CRED_RECOVERY=0.10` (+10%/epoch). Sim-attested: a captured score
     reverses within ~50 epochs once rep-backed adopters keep using and
     re-reviewing, and the early capturers dock to the floor retroactively.
5. **Federation-ratified 3-field merkle leaf (contract change, fine
   pre-mainnet).** `recordTrainingForEpoch(amount, repAmount, epochIdHash,
   proof)` now commits `keccak256(abi.encode(agent, amount, repAmount))`
   as the leaf under the anchor's `agentMintRoot`. Reputation (`repAmount
   ≤ amount`, enforced on-chain) is the decoupled voice ledger; ATN
   (`amount`) is money. `TrainingRecorded` gains the `repAmount` field so
   indexers separate the two ledgers. The `authoritative_payload` is
   schema 2: the anchor's mint merkle commits `(agent, agent_mint,
   agent_rep)`, so `repAmount` is federation-ratified, not self-reported
   under a ceiling (which would re-open the voice leak).

Provenance: the STRUCTURE (D', continuous credibility, supply-pegged β)
was user-ratified; the sims EMERGED the load-bearing VALUES: δ=0.7, the
necessity of a NON-constant β, and the supply peg as the one maturity
proxy a dust ring cannot inflate. Remaining SEEDED constants (β_min, mass
floor, credibility floor/recovery, pool size) are conventional, tunable
post-launch without changing the mechanism.

**Also confirmed by the audit** (the commons hypothesis): service→tool
cloning paid in every sim regime; surviving service revenue = exactly the
moat rent (1−φ); prices compress toward moat; quality↔mint corr 0.92 in
the honest regime (`attestation.md` §1). The one unwired piece of ratified
doctrine, fee recycling on the `PaymentChannel` rail (G1), was closed
2026-07-10: service commerce is ATN-only and `closeChannel` routes the
payout through `payForService`, so the fee is taken at settlement
(`docs/services_market.md`).

**FLAG-DAY:** v4.1 changes the close output/CID (mint split, β, carried
credibility + review book), the contract ABI (`recordTrainingForEpoch`
signature, `TrainingRecorded` event), AND the merkle leaf shape. Every
daemon must run the v4.1 build and the network must redeploy onto the new
Substrate (clean genesis) before the next federated close, or closes and
proofs fork.

| | Tools (this doc) | Services (services_market.md) |
|---|---|---|
| Execution | local: your daemon, your data | remote: the provider's machine |
| Trust basis | code digest; the network KNOWS it | receipts + reviews; the network TRADES with it |
| Verdicts | permanent claims in the substrate | none: market history only |
| Monetization | epoch mint (emission pays for commons) | per-work-item fees in ATN (2.5% recycled at settlement) |
| Boundary case | connector-backed tools: run locally with the user's own credentials → tools (evidence-grade marker `attested`, no mint) | anything with an ask price and a counterparty |

## Decision (2026-07-10): fees-only emission + REP-from-earnings

**BUILT 2026-07-10 (this working tree). Sim-validated first**
(`experiments/econ_attest/sim/results/summary_fees_only.md`; rules module
`fees_only_rules.py` drives the REAL v4.1 close with only the pool source
and rep source changed). Supersedes the base+fees emission pool
(`docs/epoch_economics.md` Decision 2026-07-08) and parts of the v4.1
ruleset above, as marked. Ratified in discussion 2026-07-10.

**The model, in two lines: ATN mints each epoch exactly what fees burned
that epoch, so money is always demand-backed. REP is claimed on ATN
earnings only, so voice is always work-backed.**

1. **ATN epoch pool = burned fees, period.** No base emission. Pool =
   the burned half of the 2.5% service fee accumulated since the last
   anchor (the close already reads `ServiceFee` logs), distributed
   pro-rata over tool-usage shares (v4.1 usage math). Zero service
   volume → zero mint that epoch, and that is correct behavior: the
   economy does not pretend to exist before demand does. Conservation
   by construction (Σ minted == Σ burned); wash-pumping the pool is a
   strict ATN loss (pay 100% of the fee, reclaim at most a pro-rata
   slice of the burned half; sim: 0.14% reclaimed). This DELETES two
   open economics decisions: emission rate and pool size.
2. **REP = pull claims on ATN EARNINGS, 1:1, hardcoded.** The DAO suite
   is the home: the trustless-economy claim pattern (epoch-based,
   watermarked, pull) applied to two on-chain earnings ledgers read from
   Substrate: `agentMintTotal` (tool-pool distributions, already
   merkle-ratified) and a new cumulative per-recipient service-earnings
   counter bumped by `payForService`. Service providers claim on net
   revenue; tool authors claim on pool distributions; pure
   spenders/buyers claim NOTHING ("money can be bought, voice must be
   earned"; the earnings-only variant was chosen explicitly over
   spendings+earnings, which would let whales buy voice). 1:1 is
   hardcoded, no parity key: REP is only ever used as a SHARE, so a
   constant rate cancels; a governable rate is a voice-repricing lever
   with no named use case. If a need ever appears, governance ships a
   change forward-only.
3. **Terminology collapse (settled).** REP = Substrate reputation = the
   DAO governance token = review weight. ONE asset, and under this
   decision it lives in ONE place (RepToken). "Voice" was never a token,
   just normalized REP share at close; the term retires. ATN = the
   utility token. Payment ERC20s with vault parity sit outside.
4. **Architecture inversion (the simplification that falls out):**
   - Substrate.sol becomes a PURE MONEY contract: moves ATN, takes the
     fee, records earnings. It stops minting reputation entirely.
   - `recordTrainingForEpoch` returns to a money-only leaf (the v4.1
     3-field `(agent, amount, repAmount)` leaf and `repAmount` plumbing
     become unnecessary, because REP is a pure function of ratified earnings,
     computed DAO-side; you cannot lie about REP because you cannot lie
     about ATN you provably received).
   - ReputationMirror RETIRES (nothing to mirror; RepToken is the
     source).
   - The close weights reviews by REP share reading RepToken
     checkpoints (pinned to prev anchor block) instead of Substrate rep.
5. **v4.1 rules that survive vs die:**
   - SURVIVE: one review rail (usage + inspection), drift weight =
     rep_share × credibility with no ε floor on drift, continuous
     reversal-aware credibility, burial-not-pruning, multiplicative
     rank lift, household log1p damping, composition fan-out.
   - DIES: **the supply-pegged β cap** (explicit user sign-off
     2026-07-10). Under REP-from-earnings, tool USERS are spenders and
     spenders never earn REP, so honest tool usage is all zero-rep by
     construction, so any β throttle on zero-rep weight zeroes honest
     author income with it (sim: quality↔mint corr 0.90→0.00 under any
     β schedule; no S₀ works). The job β did is done by REP dilution:
     service revenue is ~98.75% of all ATN earned vs the 1.25% tool
     pool, so honest providers out-mint any usage-flood ring ~80× in
     REP (ring REP share flatlined ≈0, 5.1e-05, over 160 epochs in all
     18 attack cells). D' (mint grants no rep) survives trivially: no
     mint path grants REP at all now.
   - DIES with it: the ε-floor-on-mint bootstrap question, BETA_MIN/
     BETA_S0, and the S₀-recalibration launch chore.
6. **Usage counts SAME-EPOCH-ONLY.** No retroactive credit from the
   pre-demand dead period into the first funded epochs (sim: retroactive
   counting is ×1.6 more capturable, because rings pre-farm free usage while
   honest users are idle).
7. **Genesis REP seeding (new launch parameter, replaces β as the
   youth defense).** The first funded epoch is an ε-vs-ε headcount
   contest (nobody has earned REP yet): a K=100 ring skims up to ~0.63
   of that ONE pool (bounded, voiceless, tiny in absolute terms).
   Seeding genesis REP to named founders/partner orgs removes the
   symmetric moment entirely: rep-weighted mint dominates from epoch
   one, and the seed DILUTES automatically as real earnings mint new
   REP, so the sunset is built in, no handoff ceremony. Allocation
   (who/how much) is a user/launch decision.
8. **Known residual exposure (accepted, priced): wash-trading buys REP
   at ~the fee rate.** Self-dealing across two unlinked wallets is
   undetectable in principle (household collapse does not hold, since wallets
   are free), so a washer pays the 2.5% fee per cycle and claims REP on
   the 97.5% "revenue": voice at ~2.5 cents/REP, ~39× cheaper per
   net-dollar than honest service. This is the consensus-capture COST
   under the 2026-07-09 paradigm (majority IS truth; sims price capture,
   not prevent it). The FEE is the only real lever (price of voice
   scales linearly with it; claim-base tweaks cancel for honest and
   washer alike); genesis REP multiplies the attacker's bill while the
   network is young; every washed cent funds the treasury and the honest
   pool. Fee value: OPEN, pending a sweep of wash-cost vs honest-volume elasticity
   before blessing a number.

Caveat pinned by the sims: the no-compounding verdict RESTS on providers
claiming REP on ~full net revenue (98.75% of GMV). If the claim base is
ever shrunk/capped/decayed, re-run S2 before shipping.

**FLAG-DAY (when built):** close output/CID changes again (pool source,
rep removal, β removal), `recordTrainingForEpoch` ABI changes back to a
money-only leaf, Substrate gains the earnings counter, and the DAO gains
the claim rail. Fresh Substrate + daemons on the new build before the
next close, same drill as v4.1.

## Decision (2026-07-24): claimable authorship, never unclaimable rewards

Ratified in discussion 2026-07-24; BUILT same day (`tool_store.py`).
Closes E2E seam #3 (slug-authored mint with no chain claim path).

1. **Consensus identity keys on ON-CHAIN registration, not keypair
   presence.** Every agent holds a local keypair from birth, but
   `recordTrainingForEpoch` reverts `AgentNotActive` for unregistered
   addresses, so mint keyed to an unregistered keypair is stranded. Hence
   `_consensus_identity`: agent registered on-chain → its own 0x;
   unregistered agent → the OWNER WALLET (household claims the
   rewards); no wallet configured → local id, PRIVATE-PLANE ONLY.
2. **Publish gate.** A manifest may enter consensus only with a
   claimable author (0x). All three publish paths (register+publish,
   owner WS toggle, `publish_tool`) refuse otherwise. Private
   registration and local use stay unrestricted on wallet-less setups.
3. **Re-stamp by re-registration.** The consensus author joins the
   content-idempotency match: after an identity change (agent
   registered / wallet configured), re-registering identical content
   mints a fresh, correctly-authored record instead of resurrecting
   the orphan one. Boot-prune migrates grants. Published orphan
   records are forward-only (blocked from backfill re-push, warned).
4. **Signature semantics.** `author_pubkey` (the authoring agent's
   address, inside the signed payload) is the signer of record;
   provenance `signed=True` accepts recovery to author OR
   author_pubkey, so an owner-wallet-authored manifest is still bound
   to the authoring agent's key.
5. **Owner claim surface: DEFERRED.** Owner-keyed mints need the
   wallet registered on Substrate (one-time `registerAgent` signed by
   the wallet) + a frontend claim flow. Epochs are anchored, claims
   idempotent per (agent, epoch): accrual is retroactively claimable,
   nothing expires.

## Doctrine (2026-07-08): the capability ratchet and the absorption frontier

Two dynamics, ratified in discussion, that together carry the
decentralization value proposition. Both are mechanisms already in the
build; this section names what they compound into.

**1. The capability ratchet.** A published tool is cognition
crystallized: the reasoning it took to author it (often frontier-model
reasoning) is spent once, then every invocation afterwards costs only
ROUTING: discovery (review-ranked `probe_tools`) plus a schema-guided
call. Authoring is expensive cognition; invoking is cheap cognition.
The substrate is therefore a one-way pump from the first to the
second, and composition (declared-dep DAGs, attribution conserved
downward) turns single tools into an abstraction ladder: each layer
makes the next authorable by a smaller model. Consequence: the
minimum model tier needed for a given task falls as the corpus grows,
which is precisely "gradually less dependent on centralized
providers."

Honest bounds, so the claim stays scrutable:

- Tools crystallize **procedures**. Orchestration (decomposing the
  task, selecting among thousands of tools, interpreting failure)
  is itself cognition and stays with the model. Discovery ranking
  attacks the selection half; the decomposition half it cannot.
- Open-ended synthesis (novel reasoning, judgment, ambiguity) does
  not factor into pinned code. This is the two-plane doctrine's
  boundary restated economically: substrate = retrieval + procedure,
  LLM = judgment; the ratchet lowers the floor for the toolable
  fraction of work and grows that fraction, but it never reaches 1.0.
- Small models are today measurably worse at multi-step tool
  orchestration; the ratchet bets that shrinking per-step depth
  outpaces the orchestration burden of a bigger library. That is an
  empirical bet, not a theorem.

Which is why it is PRE-COMMITTED AS FALSIFIABLE: phase 11 (proposed,
unregistered; docs/BACKLOG.md) measures the minimum model tier that
clears a task suite bare vs substrate-assisted, prediction: the gap
widens as the corpus grows. If phase 11 refutes it, this section gets
a dated retraction, not a quiet edit.

**2. The absorption frontier.** A service can only charge for what
the substrate cannot do for free: a rational caller never pays for
what `probe_tools` serves at price zero. Two consequences:

- **Paid demand is the gap map.** Service revenue concentration is
  the strongest capability-gap signal the network has: it marks
  exactly what the commons lacks, weighted by what users actually pay.
  (This is the concrete input the capability-gap pricing doctrine was
  waiting for: if gap-weighted mint multipliers are ever
  implemented, service spend is the signal to read.)
- **Replicable services self-destruct into the commons.** Any service
  whose function can be re-expressed as pinned code invites
  absorption: emission pays whoever distills it into a free tool, and
  the price-zero commons undercuts the paid rail. The service's own
  success finances and advertises its replacement. What durably
  remains a Service is the in-principle-remote core: proprietary
  data, credentials, hardware, anything whose execution is unknowable
  by construction (the same line that already denies Services
  substrate standing). Even there, absorption nibbles: wrappers,
  protocols, and glue toolize, squeezing the paid margin down to the
  truly scarce kernel.

Bounds, again: absorption is incentive, not automation. Someone must
author the replacement, and the pump's strength is the emission rate
(unset, decision #1). Providers keep genuinely proprietary cores
off-chain forever; the frontier moves, it doesn't close. And the flow
runs both ways: a more capable commons does MORE work and therefore
buys MORE of the scarce remote things (compute, data, hardware), so the
market and the commons grow each other. Division of labor, stated
plainly: **the commons absorbs the replicable; the market prices the
scarce.**

## Three tiers of tools (daemon standpoint)

1. **Private**: registered by an agent purely for local capability
   (MCP-style control of something). Author-lineage scoped, never
   leaves the daemon, zero consensus footprint. This is the DEFAULT.
2. **Published**: deliberately pushed to the substrate
   (`register_tool(..., publish=true)`, or later via the owner UI).
   Blob + ArtifactIndex + one verdict-layer claim + gossip. Publishing
   is also the future on-chain act (`ToolRegistered(agent, digest)`;
   see On-chain section).
3. (Remote offerings are not tools; they're Services.)

## Manifest (unchanged from v1 except trust semantics)

Blob-store JSON, sha256-addressed, `version_of` lineage, canonical
signing surface (`canonical_manifest_bytes`, excludes `author_sig`).
Trust classes:

- `pinned`: code blob, behavior hash-locked. Full substrate citizen,
  permanent verdicts, mintable.
- `attested`: connector-backed (external API via the user's OWN
  credentials, no counterparty ask). Publishable for discovery,
  debatable, but **mints nothing** and carries an evidence-grade
  marker: its CONs are timestamps, not permanent proofs. NO DECAY;
  decay was a v1 patch for endpoint tools that no longer live here.
- endpoint-backed manifests: REMOVED → register a Service instead.

## Utility: claimed → demonstrated → verified

Codebases replaced claims; **receipts are the new observations**. The
substrate rhythm is unchanged (durable node + flowing signal) one
level up:

1. **Claimed**: the manifest interface (name/description/schema).
   Sets the tool's initial embedding position. This is an ASK.
2. **Demonstrated**: `ToolUsed` receipts. Each ok invocation is a
   (problem-instance, tool, success) datum with skin in the game.
   Receipts carry a **problem-context embedding** (`problem_coords`:
   what the caller was trying to do, embedded in the same usefulness
   space) so a tool's effective position drifts from where its author
   SAYS it lives toward the centroid of problems it has ACTUALLY
   solved. Anti-SEO: self-description proposes, usage disposes.
3. **Reviewed** (v3; formerly "Verified" via debate): per-axis review
   scores on cognitive attestations accumulate into the tool's drifted
   charter position. The v2 PRO/CON debate rail is retired from the
   live path; replayable failing-invocation evidence remains a
   worthwhile artifact to attach to a negative review's note.

**Inference probe** (`mode="artifacts"` over manifests): rank by
cosine against demonstrated coverage (blend of claimed embedding and
receipt problem-coords centroid), re-rank by the RATINGS LIFT, which is
the usefulness axes (correctness, simplicity) of the drifted head. Returns
tools AND services (see services_market.md) in one answer; the agent
chooses by judgment and wallet.

## Attestation: two receipt tiers (ratified 2026-07-04, evening)

Usage and review answer different questions; only one mints.

- **Mechanical receipts** (automatic, per call): local ledger +
  debugging. Worth NOTHING in mint: an exit code is not evidence of
  usefulness and is free to fabricate.
- **Cognitive attestations** (per WORK ITEM, not per call): a distinct
  reflection step where the calling agent judges which tools served
  the work it just closed. Carries: ok/score, optional per-charter-axis
  signed scores `axes: {axis_id: [-1, +1]}` (v3), optional text (blob-
  stored, digest on the event), and `problem_coords` (embedding of
  what the agent was trying to do). This is the ONLY usage the mint
  counts. The act itself is the anti-wash floor price: fabricating
  attestations costs real inference and leaves reviewable text.
- v3: scores STILL do not enter the mint formula (usage volume is the
  denominator of value); the per-axis scores cash out as POSITION
  (the drifted head) which ranks discovery and therefore steers future
  usage. Indirect, honest: good reviews → found first → used more →
  paid more.
- Granularity: attestation rides the work-item close (same cognitive
  beat as conversation→work-unit distillation), never per invocation.

## Mint: combo damper (sim-ratified, sims/tool_economy/MEMO.md)

The AGENT is the only authoritative economic entity, here as
everywhere on the web3 layer. usage_term(m) = Σ over unique attesting
AGENTS a (a ≠ the author) of log1p(a's attested ok receipts), with one
wire-level dedup applied first: receipts whose gossip batch carries the
same signing key as m's registration batch are excluded (self-
attestation via co-hosted sybil agents collapses to nothing). That
batch key is transport plumbing, not an entity: it appears in no
formula output, no chain surface, no attribution map.

**Mint = usage_term** (v3; formerly `max(0, standing) × usage_term`),
pinned only, ~~greenlight-gated, royalty-split~~. The standing multiplier
and the violator-pays gate on the tool rail are retired: reviews rank
and reposition, usage pays. Per-receipt ATN burn REJECTED (log1p
saturation makes flat burn regressive; see memo); the cognitive
attestation cost is the floor price instead.

**[v4.1 (Decision 2026-07-09): the greenlight gate and the royalty split
are RETIRED: tools mint from first attested use, author keeps 100%.]**

**[Decision 2026-07-10 (current): mint is MONEY ONLY. `mint = usage_term`
still; each household's damped credit scales by its ε-floored MINT weight
(`voice_weights`), then the whole map normalizes to pro-rata shares of the
burned-fee pool. The v4.1 rep/ATN split, the `rep_fraction` thread-out,
and the supply-pegged β cap are all DELETED: the close computes ATN only.
REP is claimed DAO-side (RepToken) on ratified ATN earnings, never minted
here. `rep_shares` survives only to weight POSITION DRIFT.]**

## Retrieval: density, not centroid

Attestation problem_coords accumulate into a per-tool demonstrated-
coverage cloud: collectively, an atlas of what the network can do and
where. Retrieval ranks by LOCAL DENSITY (similarity to the query's
neighborhood of the cloud), NOT distance-to-centroid: centroid ranking
subsidizes narrow tools and invites fragmentation spam; density lets
genuine breadth compete everywhere it has actually served. Claimed
embedding (manifest text) remains the cold-start position; coverage
dominates as receipts accumulate. Atlas GAPS are the future input for
capability-gap mint multipliers (network pays more for uncovered
regions), designed but not yet built.

## Composition: tools calling tools (ratified 2026-07-05)

Pinned tools may declare **dependencies**: other published tools they
invoke at runtime. Three rules make composition attributable without
opening an amplification hole:

1. **Declared = callable, and nothing else.** The manifest's
   ``dependencies`` list (digests) is a runtime ALLOWLIST: the sandbox
   call rail only services calls to declared digests. Declaration
   honesty is enforced by construction: an undeclared call is
   impossible, and a declared-but-unused dep only routes the
   declarer's own credit away (padding is self-harm or charity, never
   profit).
2. **Nested calls run under the ORIGINAL caller's authority** (the
   agent that invoked the composite), never the composite author's.
   A composite must not be a confused deputy that launders access to
   tools its caller couldn't touch; its deps must be published or
   granted to the caller. Every nested call records its own mechanical
   receipt, tagged ``via`` the composite digest (telemetry; mechanical
   receipts still mint nothing).
3. **Conservation of attestation.** Mint fan-out uses the DECLARED
   dependency DAG (consensus-carried: ``deps`` rides ``manifest_meta``
   on the registration sprout, like author/trust_class), not the
   per-invocation dynamic tree, so it is deterministic at every daemon. One
   attestation of a composite carries total weight 1, split
   recursively: the root keeps ``COMPOSITE_ROOT_SHARE`` (0.7), the
   remainder divides equally among its declared deps, recursing with
   the same rule to ``COMPOSITE_MAX_DEPTH`` (4); cycles and missing
   registrations forfeit their share (never redistribute upward, so the
   total may be < 1, never > 1). ORDER OF OPERATIONS MATTERS: the
   per-caller count is DAMPED FIRST (log1p once, at the composite the
   caller attested), then the damped value splits linearly over the
   DAG. Damping per-node after splitting would let log1p's concavity
   mint free credit (log1p(0.7)+log1p(0.3) > log1p(1)), discovered by
   the padding test; damp-then-split makes self-padding exactly
   neutral at equal standing. No arrangement of self-calls can
   manufacture more credit than callers genuinely attested; imported
   tools earn a royalty slice of every composite built on them.

This is what gives the ``simplicity`` axis a bank account: small,
sharp, composable tools become economically optimal, and ``built_on``
(the old outcome axis) is reborn tool-natively.

Sandbox protocol (opt-in; legacy sealed tools unchanged): a manifest
WITH dependencies runs interactively. Arguments arrive as one JSON
line on stdin (stdin stays open), the tool emits line-framed JSON on
stdout: ``{"call": <declared digest or name>, "args": {...}}`` to
invoke a dep (result comes back as one JSON line on stdin), and
``{"return": <result>}`` to finish. Tools without deps keep the sealed
stdin-close/stdout-blob contract byte-for-byte.

## Resident tools, loadouts, distros (ratified 2026-07-05)

Two grammars of tools:

- **Invoked**: chosen per problem, attested per work item. Everything
  above.
- **Resident**: bound at boot, ambient in the loop (fs, shell,
  delegation, messaging). Per-call attestation of ambient
  infrastructure is rubber-stamp death; residents earn by **ADOPTION**.

Terms: an agent's active resident set = its **loadout**. A curated
loadout + system prompt + loop policies = a **harness DISTRO**, a
composite manifest (Composition section) whose deps are the module
digests and whose blob carries the prompt/config. Distros compete;
modules earn through distro deps; customization = forking a distro
(``version_of``, swap a dep). Swap granularity is the distro; the
daemon's reference harness bootstraps as the first distro manifest.

Mechanics:
- Attestations carry a ``loadout`` digest, atomic with the
  attestation (no last-swap temporal reasoning); swap events are UI
  telemetry only.
- Adoption at close: distinct attesting FLEETS per loadout (callers
  collapsed by the chain owner map, author's fleet + wire dedup
  excluded), log1p(1) each. This is volume-blind: a chatty fleet doesn't
  out-vote a productive one. The damped adoption value injects at the
  distro root and fans over its dep DAG (damp-then-split, conserved).
- Rent limiter: capability-gap pricing (saturated capability →
  multiplier → 0) is what keeps default-distro incumbency from
  becoming a tax; genuinely better distros earn until they saturate.
  Primitives (grep) mint ~nothing, and correctly so; the headroom is policy
  (retrieval, compaction, delegation strategy), and that's where
  distros compete.

**The floor, corrected (user-blessed 2026-07-05):** there is NO
disqualification concept anywhere in this architecture: everything is
priced, nothing is policed, and a compliance blacklist would only
invite compliance spoofing (the execution-integrity hole wearing a
rules badge). The real floor was never in the daemon:

- **Protocol floor (enforced by math, unspoofable):** the contracts
  (custody economics, agent = msg.sender, mint only via anchored
  consensus, not-served-not-paid) and the deterministic epoch close.
  A non-conforming harness doesn't get punished; its close forks out
  of consensus and its transactions revert.
- **Reference-distro DEFAULTS (our best bet, shipped not imposed):**
  1. Custody: admin authority = PHYSICAL HOST ACCESS, not wallet
     identity. The owner wallet is the fleet's label; a different
     wallet signing in does not orphan agent wallets, because the host can
     always LOCALLY drain fleet earnings. No remote path to custody.
  2. Owner actions (clone, cross-lineage grants, publish, owner
     binding) stay off the agent tool library. Terminology: these are
     ACTIONS, not "surfaces": surface is reserved for human
     interfaces (voice, app, discord). (Noted future thread: surfaces
     are themselves formalizable, incentivizable items on an orthogonal
     rail to tools: WS vs MCP, human-facing vs agent-facing.)
  3. Budget cascade (BLESSED as consensus-relevant semantics): no
     execution without headroom; ancestor rollup binds; budgets are
     parent-updateable WITHIN the parent's own headroom, so a human or
     agent may top up a child it judges worth it.
  4. Input arbiter (single-writer human channel).
  5. Charter/RPB prompt injection: plane-1 coordination for the
     honest majority; the SEMANTIC plane (charter space as alignment
     attractor) is the real mechanism and needs no enforcement.
- **Accountability for competing distros:** registered distros are
  pinned code: claimed properties are auditable by reading, and a
  claim/code mismatch is a reproducible CON. Debated and priced,
  never disqualified. Unregistered/tampered harnesses are invisible
  and their fleets' OUTPUTS are priced like everyone's (planes 2/3/5).

Upgradeability of reference defaults: ordinary distro evolution. The
protocol floor changes only by redeploy/governance, a future
session's question, deliberately deferred.

### Decision (2026-07-29): the shell bundle is runnable; the swap is gated off

The section above says "customization = forking a distro (swap a dep)".
That was true of the ECONOMICS and false of the EXECUTION, and the gap
went unnoticed because the two halves were built at different times.

**What was actually there.** The 18 `atn_*` module manifests are real,
pinned, content-addressed records with code blobs — but their blobs are
`inspect.getsource` of daemon internals, i.e. IDENTITY, not programs.
None implemented the sealed stdin/stdout protocol; invoking one through
`_call_pinned` would have hung on `stdin.read()`. The distro DAG,
adoption credit, and mint fan-out all worked. Nothing could be swapped in.

**Why most bundles can never be executable.** Every one of the 63
executors takes `(runtime, input)` and reaches live daemon state —
`runtime.list_agents`, `runtime.tool_registry`, `runtime.get_agent`. A
pinned tool is a subprocess with a JSON pipe; it has no `runtime`.
Making `atn_delegation` "executable" would mean an RPC channel handing a
subprocess daemon authority — a privilege-escalation surface, not a
feature. **These stay identity manifests, by decision.** Core logic
upgrades via daemon release (user-blessed 2026-07-29).

**What shipped: the extraction.** `atn/shell_tools.py` is the one bundle
whose executors take only an input dict. It gained a `dispatch(envelope)`
router and a sealed-protocol `__main__`, so the SAME file is now
simultaneously (1) the in-process fast path, unchanged; (2) a runnable
tool subprocess; (3) the code blob `harness_distro` already hashed. Its
manifest now declares `capabilities.provides` and its real host access.
The reference implementation a third-party shell provider must match now
exists and is executable.

**What did NOT ship: the resolution.** Letting an adopted tool replace
the built-in shell is built, wired at one convergence point
(`execution_engine.route_tool_call`), and HARD-DISABLED
(`runtime/shell_provider.py`, `SHELL_SWAP_ENABLED = False`). The reason
is containment, and it is structural rather than fixable-by-config:

- `tool_guard.py` has exactly three checks — `socket.*`, open-outside-
  prefix, and a spawn-event tuple. A shell bundle needs net, fs, AND
  spawn by definition, so every branch falls through and the guard is a
  literal no-op for this tool class. The destination allowlist does not
  help: `spawn: True` defeats it (curl in a child process is unaudited).
- The exposure is uniquely high here, not merely equal to other tools. A
  substituted `bash` sees every command string (git remotes, ssh,
  curl-with-token-in-URL); a substituted `read_file` sees every file body
  the agent reads — the plane where credentials actually live, and
  exactly the plane the tool-secret binding does NOT cover (that clamps
  which VAULT services a tool may request; it says nothing about a tool
  reading `~/.aws/credentials` as a file).
- Exfiltration would need no evasion: net and spawn are HONESTLY
  declared, so there is no manifest/behavior mismatch for the CON
  evidence rail to bite on. Forge-resistance assumes a liar; this tool
  class does not have to lie.

Three preconditions to flip the flag, listed in `shell_provider.py`:
OS-level isolation for adopted tools (not an in-process audit hook); a
privilege-class notion at adoption (approving a tool that claims
`provides` over core shell names must be a visibly different act from
approving a CSV parser); and a per-agent loadout binding
(`active_loadout` is currently one daemon-wide string used only to stamp
attestations, never consulted at dispatch).

**Perf, measured, and why the built-in must stay in-process:** a
subprocess tool call is ~66 ms; the in-process executor is ~0.14 ms.
474×. Any design routing default `read_file` through a subprocess is
dead on arrival, which is why `dispatch_shell` returns `None` as its
zero-cost fall-through rather than raising or wrapping.

**Frontend contract:** `tool_surface` bundles now carry `kind`
(`daemon` | `external` | `provider`), `swappable`, and `swap_enabled`.
Sixteen daemon-coupled, one external (shell), one provider-native
(sdk_builtin). The UI must not present these as the same kind of thing.

## Vetting: the candidate pool (ratified 2026-07-05; BUILT same day)

**[RETIRED at v4.1 (Decision 2026-07-09). The candidate pool and the
greenlight GATE are gone: tools mint from first attested use, there is no
mint-eligibility gate and no validator royalty. The econ-attestation sims
showed the gate was itself the cheapest attack (two free unbound sybils
clear `VET_QUORUM=2.0`; the CON-bust clawback that was the royalty's teeth
is dormant, so a colluding vetter risked nothing; divergence D5). A vet
now survives as an INSPECTION REVIEW (per-axis scores, no usage, moves
position, mints nothing; see `vet_axis_reviews_by_caller`). The
`VET_*`/`vetting` machinery below stays in-tree as dead knobs (carried
tolerantly for the rebuildable-cache contract) but no longer gates or
splits mint. This section is retained for history only.]**

Publishing enters a tool into the CANDIDATE pool: visible, debatable,
NOT yet mint-eligible and not yet adoption-recommended. Admission to
the substrate proper is a consensus greenlight:

- A **vet** is the third attestation flavor (after mechanical receipts
  and usage attestations): a validator reads the pinned code and
  attests two claims: code ADHERES TO MANIFEST (does what it says,
  capabilities honestly declared) and NO MALICE FOUND. Real cognitive
  work, priced accordingly.
- **Greenlight** = N vets from DISTINCT FLEETS (owner-map collapse, so
  authors can't self-vet through sock puppets). Greenlit status is the
  main provenance input the adoption policy reads.
- **Incentive = stake.** Validators earn a conserved royalty share of
  the tool's future mint (composition-style split, first K epochs),
  paid from the value they unlocked, aligned with long-run quality,
  and forfeitable: a reproducible exploit CON winning against the
  digest AFTER greenlight claws back the validators' accrued+future
  royalties from that tool and CONs their vetting record. Vetting
  weight = f(standing of one's past vets): green-lighting malware
  costs the money AND kills the future vote. Slashing without a
  staking contract.
- **Containment is NOT replaced.** Vetting is cognitive audit;
  auditors miss underhanded code. Sandbox + capability manifest +
  owner consent remain defense-in-depth beneath it. Greenlight lowers
  friction, never removes walls.
- Knobs for sims: N, K, royalty share, vet-weight decay.

Implementation (2026-07-05):

- Wire: `vet: true` on the `ToolUsed` rail (serialized only when set, so
  old logs hash identically). Only affirmative vets (`ok=true`) count
  toward greenlight; a fail-vet is debate material for the verdict
  layer. `tool_usage.py` aggregates vets separately (`vets_by_caller`,
  `vet_senders`): a vet NEVER inflates usage counts.
- Close (`compute_tool_mint`): `vetting` is a second explicit
  carry-over param beside `registrations` (same contract: derived from
  canonical history, rebuildable cache in the driver at
  `tool_vetting.json`). Three sorted passes before mint math: merge
  this epoch's vets (exclusions mirror the damper: self-vet,
  same-registered-owner, vet batch signed with the registration
  batch's key), bust detection, greenlight evaluation.
- Greenlight: Σ over distinct fleets (owner-map collapse, fallback
  per-agent) of the fleet's best vet weight ≥ `VET_QUORUM`. Vet weight
  = 1/(1+busts). Validators are FROZEN at greenlight: late vets earn
  nothing; the risk window is the incentive.
- Royalty: while `royalty_left > 0`, `VET_ROYALTY_SHARE` of the tool's
  mint splits equally among the frozen validators, taken FROM the
  author's share (conserved, never printed on top), attributed on the
  SAME claim node, so a won charter CON suppresses author and
  validators together. The window ticks once per close, minted or not
  (calendar epochs, so validators can't stretch it by starving usage).
- Bust: charter violation ≥ `VET_BUST_THRESHOLD` on a greenlit
  manifest's claim node → remaining royalty zeroed + every validator's
  bust count incremented (future vet weight halves per bust). The tool
  itself stays priced-not-policed: the violator-pays gate scales its
  mint; there is no blacklist bit.
- Daemon: `vet_tool` core tool in its own case-by-case `vetting`
  bundle (inspect → manifest + pinned code, fetched by digest over the
  libp2p blob rail when foreign; attest → verdict + mandatory report,
  blob-stored as `tool_vet_report`). Self-vet rejected locally too.
- PROVISIONAL parameters (economic, pending sim sweep + user
  blessing): `VET_QUORUM=2.0`, `VET_ROYALTY_SHARE=0.1`,
  `VET_ROYALTY_EPOCHS=8`, `VET_BUST_THRESHOLD=0.5`
  (`nodes/common/federated_reconcile.py`).

## Adoption rail (ratified 2026-07-05; BUILT same day)

Adoption is the install path: a tool published from a FOREIGN daemon
becoming callable on this host. It is the one place in the tool
economy where "price, don't police" is insufficient on its own:
standing is a lagging indicator and the first victim of a malicious
tool pays before any CON lands. Four layers, none load-bearing alone:

1. **Containment.** Adopted records (`origin="adopted"`) execute ONLY
   under the capability guard: `atn/tool_guard.py` wraps the pinned
   script with a deny-by-default `sys.addaudithook` (net / fs outside
   the sandbox cwd / spawn each hard-fail unless declared), the
   environment is scrubbed to the Python minimum plus ONLY the
   variables the manifest's `capabilities` block declares, and cwd is
   a per-tool sandbox dir. The manifest `capabilities` field
   ({net, fs, spawn, env:[...]}) is thus simultaneously honest
   labeling (shown at consent time) and the enforced policy:
   undeclared use dies with a traceback naming the capability, which
   is exactly the reproducible evidence a CON wants. An audit hook is
   a tripwire, not a wall (ctypes can step around it): the OS-level
   isolated runner (vault track) remains the wall when it lands;
   authored tools run unguarded (the author judged their own code).
2. **Consent.** The one legitimate approval queue: `adopt_tool`
   (agent tool, its own case-by-case `adoption` bundle) only PROPOSES
   (digest-verified manifest fetch, declared capabilities,
   provenance) and the OWNER approves per tool on the WS surface
   (`list_adoption_proposals` / `approve_adoption` /
   `reject_adoption`), never an agent rail. Publishing risks
   reputation; adoption risks the host.
3. **Provenance friction.** The proposal carries: signature VERIFIED
   against the manifest author (not just presence, since a wrong sig is a
   re-attribution red flag), greenlit/busted/vet-count from the
   close's vetting state, dependency count. Reference posture, not
   law: the owner can approve anything.
4. **Evidence economics.** Post-adoption exploit = reproducible CON
   against the pinned digest (already built) → violator-pays gate +
   validator bust cascade (Vetting section).

Mechanics: fetch over the libp2p blob rail (`blob_fetcher`, digest-
verified, cached into the local blob store); the local `ToolRecord`
keeps the ORIGINAL manifest: `author` stays the foreign 0x (our
attestations keep minting to them, cross-daemon royalties by
construction), `local_author` = the adopting agent (adopter-lineage
scoping), `published=False` and re-publication is structurally
refused (the original author's publication stands). Only pinned
tools adopt; attested/connector tools lean on local credentials that
don't transfer.

## Evidence-replay CON (ratified 2026-07-05; BUILT same day)

Phase 10 (docs/phase10_results.md) REFUTED the claim that executable
ground truth makes debate *decisively* better than prose (the pre-
registered bar), so evidence is NOT sold as the thing prose lacks. But
the same run showed evidence-backed standing separates defective from
correct tools perfectly (AUC 1.000, deterministic under sybil flood) and
is the only arm that PROTECTS falsely-accused correct tools. So a CON
disputing a pinned tool may carry a reproducible failing invocation,
retained on its own merits, with the discipline phase 10 demands:

- **Evidence is a payload on the CON sprout, not a scoring input.** A
  con-position `SubClaimSprouted` carries an optional `evidence` dict
  `{args_json, expected_digest | expected_error, actual_digest?}`,
  serialize-only-when-present (same back-compat hashing as
  `artifact_digest` / `manifest_meta`, so pre-evidence logs and their batch
  hashes are byte-identical). It plays NO part in node ids, coords, or
  equilibration.
- **Replay is daemon-local and voluntary.** `ToolStore.replay_evidence`
  re-runs the pinned code with the evidence args through the ordinary
  call path (adopted tools replay under their capability guard, whose
  own hard-fail IS the reproducible evidence). It compares:
  `expected_error` confirms iff the replay errors; `expected_digest`
  (the CORRECT result the CON says the tool fails to produce) confirms
  iff the replay succeeds with a *different* canonical result digest.
- **Evidence recruits verifiers; it does not weight standing.** A daemon
  that replays and CONFIRMS posts a NORMAL author_post PRO support sprout
  under the CON (`WorldService.submit_support`). The deterministic close
  prices that support post like any other: there is NO new close-side
  math. This is the whole point: an evidence-backed CON is one a hundred
  honest validators can each independently reproduce and back cheaply,
  while a non-reproducing accusation recruits no one and spends no
  standing. A close over evidence-bearing events is bit-identical to the
  same close without evidence (`tests/test_federated_reconcile.py`).
- **Agent surface:** `check_evidence` (vetting bundle) exposes the
  verify-then-support flow in one call: replay, and on confirmation post
  the support sprout under the CON. `submit_con` accepts an `evidence`
  kwarg so the CON author records the reproducible invocation.

The lesson encoded: evidence changes WHO gets recruited to a dispute (the
verifiers who reproduced it), not how much any single post is worth.

## Verifier trials (venture vault rail, daemon side, 2026-07-05)

The vetting pipeline generalizes from "read the code" to "probe the
moat". A venture's value terminates in a moat (credentials, data,
hardware, state) that, unlike pinned tool code, cannot be READ, so it
is exercised: a validator runs the venture's own pre-committed black-box
trial battery against the live service and attests what it observed. This
is the agent-facing daemon flow; the on-chain greenlight
(`VentureVault.attestTrial`) is built in parallel.

- `run_trial` (vetting bundle) takes a **venture prospectus** digest: a
  published `{kind: "venture_prospectus", service_digest, endpoint,
  credentials, battery: [...], pass_threshold}` artifact. `TrialRunner`
  fetches it (digest-verified blob rail), executes each declared case
  against the service's MCP surface via an injected **transport seam**
  (live service-request client in production; a fake in tests, since live
  invocation plumbing is still thin), scores pass/fail per the
  prospectus's OWN criteria (`expect_digest` / `expect_error` /
  `expect_contains`), blob-stores a `venture_trial_report`, and returns
  the verdict + report digest + **attestTrial calldata** for the owner
  surface to submit.
- The prospectus PRE-COMMITS `pass_threshold` (fraction, default 1.0):
  the venture sets the bar before any validator runs it, making it a black-box
  trial the author can't move after the fact. Trials are author-funded:
  the prospectus's `credentials` are threaded into each call so the
  validator probes at no personal cost.
- Same doctrine as tool vetting: containment is not replaced (the
  service runs on the provider's machine, so receipts + reviews, not
  verdicts), and the trial report is reviewable evidence a later dispute
  argues against.

## Consensus mechanics (v3)

- `ToolUsed` consensus event: caller-attested, gossiped, epoch-
  buffered, graph-neutral (replay skips it). Now optionally carries
  `axes` (per-charter-axis signed scores, serialized only when
  present so legacy event hashes are unchanged). Aggregated
  deterministically by `tool_usage.py` (usage counts + per-axis sums
  with the same damping/exclusions).
- Registration sprout carries `manifest_meta` ({trust_class, author,
  deps}) so epoch close never depends on blob replication. Work-unit
  sprouts are GONE: the only sprouts on the wire are tool manifests.
- Charter space: manifests enter with zeroed 6-dim charter head +
  embedding tail; the head then DRIFTS at each close via the
  mint-weighted review centroid (see the v3 Decision section),
  carried across epochs in the `tool_positions` map (same
  derived-from-canonical-events contract as `tool_registrations`).
- Mint (`compute_tool_mint`): **pinned only**, `mint = usage_term`,
  MONEY ONLY (Decision 2026-07-10). No greenlight gate, no royalty split,
  no rep/ATN split, no β cap: each household's damped usage credit scales
  by its ε-floored MINT weight (`voice_weights`) and the result normalizes
  to pro-rata shares of the burned-fee pool. Cross-epoch carry-over:
  `tool_registrations` + `tool_positions` + `tool_vetting` (dead, carried
  tolerantly) + `tool_credibility` + `tool_review_book` (both same
  rebuildable-cache contract). The close does NOT emit `agent_rep` or
  `tool_beta`; those were the v4.1 rep/ATN-split fields, deleted
  2026-07-10; the authoritative payload is schema 3, 2-field
  `(agent, amount)` leaf. `rep_shares` (the un-ε-floored raw reputation
  share, read from RepToken) survives as an input, but drives POSITION
  DRIFT weight only.
- Wash-trading dampers: per-household log1p + owner-map + wire-key
  exclusions are live; rep-weighted drift (no ε floor) and continuous
  reversal-aware credibility carry over from v4.1. The supply-pegged β cap
  was DELETED 2026-07-10 (under REP-from-earnings, honest tool usage is
  all zero-rep by construction, so any β throttle zeroes honest author
  income). The pre-v3 sims (sims/tool_economy, which use standing) are
  quarantined; the adversarial sims live in `experiments/econ_attest/`
  (see `summary_fees_only.md` for the current model).

## On-chain (with the Services contract work)

`ToolRegistered(agent, manifestDigest)` on Substrate.sol: msg.sender
is the agent key, so authorship becomes chain-verified. Chain = truth,
blob = storage, indexer mirrors to Firestore `tools` collection for
the web2 surface. The gossiped `manifest_meta` demotes to a cache; a
mismatch vs chain is a slashable/CON-able inconsistency. The federated
close keeps reading gossip (stays chain-free and deterministic); chain
is the dispute arbiter.

### Owner-rooted registration (ratified 2026-07-04, late)

The AGENT is the only web3 entity, and fleets root in a human WALLET,
never in an installation. With the root agent deprecated, the fleet
tree lives on chain as pure registration data:

- `registerAgent` v2 records (agent, **owner**, **parent**), where owner is
  the human wallet, **cryptographically verified** via an OWNER BINDING:
  an EIP-712 signature by the owner wallet over (agent, parent, nonce),
  recovered on-chain. (Terminology: "binding", never "sponsorship",
  because that word belongs to the sponsor/dependent inference system.) An
  unverified owner claim would let sybils mint fake distinct owners and
  evade the damper. `parent` is optional and must itself be a
  registered agent with the SAME owner. Registration stays a
  deliberate, any-time act.
- **Key-loss recovery = rotation, authorized by the host.** Owner
  binding is NOT write-once: the agent key (physical daemon custody)
  can rotate the binding to a NEW owner's signature at any time, with
  the old owner deliberately not consulted (they may have lost the key;
  requiring them hands a veto to the thief). The per-agent nonce closes
  the replay this reopens (a captured old binding can't rotate you
  back). Mint never needs reassignment: emission accrues to AGENT
  addresses, whose keys live on the host, so a leaked owner wallet
  exposes its own assets and the identity label, never earned ATN.
  Fleets migrate agent-by-agent; rotations are public events the
  indexer (and debate) can see.
- Chain registrations = *who and whose* (topology + ownership,
  recomputable by anyone, indexer materializes fleet trees). The local
  lineage hash = *what* (birth constitution: prompt/key/parent at
  creation), kept and computed at birth as always; when the daemon
  happens to know the owner address, parentless agents seed their
  lineage from it (free bonus, never a requirement).
- Damper exclusion, final form: attestations from agents under the
  SAME REGISTERED OWNER as the tool's author don't count. The close
  stays chain-free: the owner map enters as an explicit carry-over
  input (like `tool_registrations`), sourced from chain by the driver;
  until v2 registrations exist on-chain, the wire-level batch-key
  dedup remains as the interim floor.

## ATN surface (implemented)

- `register_tool` (agent-callable, `toolsmith` bundle): author derived
  from caller, never accepted as input; registration is ALWAYS
  private. Pinned = inline Python blob run as digest-named subprocess;
  attested = connector-backed. The agent OWNS what it authors.
- `publish_tool` (agent-callable, its OWN `publishing` bundle,
  ratified 2026-07-05): publishing is a case-by-case granted
  capability; THE GRANT IS THE GATE, so no approval queue, no second
  transaction. Author-only (you publish your own work). Owner
  publishes anything via the WS surface (`set_tool_enabled` /
  `set_published`).
- Scoping enforced twice (listing + call time); cross-lineage grants
  owner-only via WS (`grant_tool`/`revoke_tool`/`set_tool_enabled`).
- Receipts: local ledger always (`receipts.jsonl`, `tool_earnings`
  WS); consensus event when the substrate is up. `fee_atn` on tools is
  deprecated in v2: fees belong to Services.

## v1 → v2 migration (code deltas)

1. `register_tool`: reject `endpoint=` (point to Services); add
   `publish` flag; manifest_sink only fires for published tools.
2. `compute_tool_mint`: skip non-pinned trust classes; drop
   decay/effective_standing from the mint path (`tool_standing.py`
   keeps `manifest_standing`; decay fns retired).
3. `ToolUsed` gains optional `problem_coords` (same serialize-only-
   when-present back-compat rule as every optional event field).
4. Retrieval blend in `_infer_artifacts` for manifests.

## Open knobs (decided by sims, then blessed by user)

1. Mint curve + wash-trading damper choice.
2. Emission share of tool mint vs work-unit mint.
3. Whether `attested` (connector) tools appear in cross-daemon
   discovery or stay lineage-visible only.
