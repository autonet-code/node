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
mint leaf is 2-field `(agent, amount)`; the authoritative payload was
schema 3 at this decision and is schema 4 now (adds `carry_cid`, the
verifiable carry-over commitment; see Consensus mechanics).

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

## Decision (2026-10-12): anchor authority + on-chain conservation

**Layers 2 and 3 below (REP signatures, genesis set) are SUPERSEDED by
Decision (2026-10-16): governed anchors.** Layer 1 (on-chain conservation)
stands.

User-approved breaking change, BUILT 2026-10-12. Closes a critical hole:
`submitAnchor` had no caller check and `recordTrainingForEpoch` only
checked the proof against whatever root was anchored, so anyone could
anchor a fake epoch paying themselves, mint unbounded ATN, claim REP 1:1
DAO-side and take over the DAO (or anchor first and fork honest daemons
out). Two layers, both enforced by Substrate.sol:

1. **On-chain conservation.** `payForService` adds its burned fee share to
   `burnedSinceAnchor`. Each `submitAnchor` closes that window into the
   anchor's `windowBurned` and sets its `mintPool` to the PREVIOUS anchor's
   `windowBurned` (lag-1, exactly the window the close already priced
   from). `recordTrainingForEpoch` reverts `EpochPoolExceeded` once an
   epoch's cumulative mint would pass `mintPool`. The first anchor's pool
   is 0. Daemon side: `voice_state` now reads the pool from the snapshot
   anchor's `windowBurned` (not a log sum), and the close floors every
   share to integer base units (`floor_mint_to_pool`) so the leaves sum
   to <= the on-chain pool exactly; `scale_mint` rounds instead of
   truncating so a floored k/1e6 scales back to k. Dust stays unminted.
2. **Anchor authority.** `submitAnchor(..., bytes[] signatures)` needs
   EIP-712 `AnchorApproval(string epochId,bytes32 epochRoot,bytes32
   prevEpochRoot,bytes32 prevAnchorHash,string agentMintCid,bytes32
   payloadHash,bytes32 agentMintRoot)` signatures (domain
   AutonetSubstrate/1, chainId, Substrate address), recovered signers
   strictly ascending, max 64, from registered active agents whose summed
   `RepToken.getPastVotes(agent, t)` STRICTLY EXCEEDS `anchorQuorumBps` of
   `RepToken.getPastTotalSupply(t)`, t = previous anchor's timestamp
   (deploy timestamp for the first anchor). The relayer (msg.sender) has
   no authority. Delegated votes, not balances: REP mints to the agent
   address (AutonetRepClaim), but a holder who delegates elsewhere (or a
   default delegate) moves that voice; any EOA REP holder, including the
   genesis REP wallet, must delegate to a registered agent for its weight
   to sign anchors. Undelegated REP still counts in the denominator.
3. **Genesis bootstrap.** While REP past total supply is 0 (or no
   RepToken is wired), weight is a COUNT of distinct, registered members
   of an immutable deployer-chosen genesis signer set, which must reach
   `genesisQuorum`. Rejected alternative: a quorum over registered agents
   (registration is permissionless, so sybilable). Tradeoff: the genesis
   set is trusted until the first REP exists, then powerless forever.
   `repToken = 0` means genesis-only forever: local dev only.

Constructor: `(treasury, vaultMinter, governor, repToken,
anchorQuorumBps, genesisSigners[], genesisQuorum)`. **USER CALL OPEN:**
`anchorQuorumBps` (proposed 5000 = strict majority of past REP supply;
liveness risk if much REP sits undelegated, safety risk if set low) and
the genesis set/quorum. Daemon: every daemon prepares the identical
anchor, each local agent signs the digest, signatures gossip on
`<event topic>/anchor-sig/1`, the rotating winner waits (default 60 s)
until `Substrate.anchorQuorum` reports reached, then relays; a daemon
whose own agents carry quorum submits at once. A daemon never signs a
close whose voice/pool snapshot was read at a different `anchorCount`.

**FLAG-DAY:** ABI change (constructor, `submitAnchor`, Anchor struct gains
`windowBurned/mintPool/minted`) + mint scaling change. Fresh Substrate and
all daemons on this build before the next close.

## Decision (2026-10-16): governed anchors

Replaces the REP-signature quorum and the genesis signer set of
2026-10-12. An anchor is an ordinary DAO proposal:

- `Substrate.anchor(epochId, epochRoot, prevEpochRoot, prevAnchorHash,
  agentMintCid, payloadHash, agentMintRoot)` is callable only by
  `governor` (the DAO timelock; reverts `NotGovernor`). Constructor is
  `(treasury, vaultMinter, governor)`, governor non-zero. The hash chain,
  one-anchor-per-epoch and the fees-only `mintPool` cap stay in Substrate.
- The DAO Governor decides early once the outcome cannot change: For above
  half of total REP at the snapshot with For >= quorum, or Against at
  half. Otherwise it runs to the deadline under the DAO's ordinary rules.
  Admin's call: no anchor-specific quorum, because the DAO can rally
  against a bad anchor during the voting period and cancel it in the
  timelock.
- Daemon side (`nodes/common/governed_anchor.py`,
  `chain_submission_driver.py`): every daemon derives the identical
  proposal from the bit-identical close (target Substrate, value 0,
  `anchor(...)` calldata, description
  `Anchor epoch <epochId> 0x<epochRoot hex>`), so the proposal id is the
  same everywhere. The rotating winner proposes (other daemons may after
  one tick; a duplicate propose just reverts), every daemon votes For with
  its agents' delegated REP, anyone queues and executes. Agent mints are
  submitted once the anchor lands. `ChainSubmissionConfig.governor_address`
  is the DAO Governor.
- Dev networks may deploy Substrate with an EOA governor; a daemon whose
  key is that governor anchors directly (no proposal).

Latency: one voting delay plus one timelock delay after the close when the
anchoring agents hold a REP majority; up to the full voting period
otherwise. **FLAG-DAY:** constructor and anchor ABI change; fresh Substrate
with the timelock as governor.

## Decision (2026-10-26): epochs the network agrees on

Decided with the owner after the 2026-10-26 survey found that no two honest
daemons could produce the same anchor: epoch ids came from each daemon's
clock (`epoch_<time_ns>`), batch membership from when each daemon's timer
fired, and the proposer from each daemon's own list of senders.

1. **Epoch id = the anchor count.** The epoch being closed is the one after
   anchor N (`anchorCount` = N, read from the chain), named from N and that
   anchor's epoch root. Every daemon derives the same id, so the same
   proposal.
2. **One epoch in flight.** No daemon closes epoch N+1 until anchor N has
   landed. Events stay buffered meanwhile; the floor on an epoch's length is
   the anchor proposal's own lifecycle.
3. **Membership by agreed time.** Epoch N contains every batch whose signed
   timestamp is at or before the cutoff W = (anchor N's on-chain timestamp)
   + (epoch length), not already in an earlier epoch. A daemon closes at
   W + grace. A batch whose timestamp is further than the skew from its
   arrival is refused (no backdating into a closing epoch).
4. **Length set by activity, Bitcoin style.** The length is retargeted from
   the previous epochs, never the current one (its batch set is only agreed
   at its close): the average over the retarget window of the burned fees
   each recent epoch carried, as a share of total ATN supply, scaled so an
   epoch reaches the target share. Bounded below by the anchor lifecycle and
   above by a ceiling (a multiple of the vote duration), so a quiet network
   still pays out. All inputs are anchored, so every daemon computes the same
   length.
5. **The proposer is drawn by REP.** One ticket per household, REP-weighted,
   among households above a REP percentile, seeded from the chain (as the
   robot poll lottery). The first two drawn propose in parallel; each voter
   backs the one matching its own view, so a bad proposal costs no time.
6. **No censorship by the proposer.** The proposal commits to the epoch's
   batch set. A voter missing a batch it names fetches it by hash from peers
   and recomputes. A voter holding an eligible batch the proposal leaves out
   votes Against. A batch in no proposal rolls into the next epoch. An
   omitting proposer sits out the proposer draw for some epochs, and repeat
   omissions become a report to the DAO (which can take REP): "never reached
   me" and "dropped it" look the same, so the automatic penalty is mild.
7. **Every rule is a governed Registry key** (`jurisdiction.<id>.*`), with
   these defaults: target share (set at deployment), retarget window 4
   epochs, ceiling 10x the vote duration, grace 120 s, skew 60 s, the
   proposer percentile and sit-out epochs. Every daemon reads them pinned to
   the same anchor.
8. **The other consensus findings ship in the same flag day:** addresses
   lowercased at the close's entry (checksummed event ids against the
   lowercase voice maps zeroed every mint); the canonical world tracker
   restored at boot, not rebuilt; a failed voice refresh blocks the close
   instead of persisting carry state; the log scan starts at the configured
   deployment block, never inferred from failed reads; the charter hash and
   sanctioned households in the payload encoding. Added with them: the
   kernels behind `world_cid` and the mint's `log1p` use platform-
   independent exp/log (`world_model/generalized/detmath.py`), since C
   libraries round some inputs differently.

   **Item 8 BUILT** (branch `epoch-rules-2`): household lookups in
   `compute_tool_mint` lowercase both sides; the driver persists the
   tracker checkpoint (`canonical_world.json`), restores it at boot and
   otherwise installs the anchored `world_cid` (same rule as the
   carry-over), and a tracker failure blocks; the voice refresh is
   all-or-nothing and a failure blocks; `blockchain.deploy_block` (registry
   key `deploy_block`, 0 = search, which now raises on a failed read)
   starts every log scan; payload schema 5 commits `charter_hash` and
   `sanctioned_households`.

**Items 1-3 BUILT** (same branch), with these specifics:
- The epoch id is `epoch-<n>-<16 hex of anchor n-1's root>-<cutoff>`. The
  cutoff rides in the id, so the next window's start is read from the
  chain (anchor n's `epochId`), never from local state.
- The cutoff is base + `epoch_length`. The base is the last anchor's
  timestamp, or `Substrate.deployedAt` before any anchor. When no pending
  batch is signed by the cutoff, it slides forward one grace step at a
  time to the first slot that holds one, so an empty epoch never stalls
  the chain.
- The rules are Registry keys `jurisdiction.<id>.epoch_length` (default
  0: an epoch lasts one anchor lifecycle), `.epoch_grace` (120) and
  `.epoch_skew` (60), pinned to the last anchor's block. Step 3 replaces
  the fixed length with the retarget.
- One epoch in flight: the driver persists the number it last closed
  (`epoch_in_flight.json`) and closes nothing until the next anchor
  lands.
- The voice refresh reruns only when `anchorCount` changes.
- Gossip refuses batches signed further than the skew from their
  arrival.
- Known gap until item 6: if an epoch's anchor never lands, the network
  waits. Item 6's retry and vote rules resolve it.

**Item 4 BUILT, cutoffs on their own schedule** (same branch, 2026-10-26).
The governance lag is now separate from the epoch schedule:
- Each cutoff is the previous cutoff plus the length, starting from
  `deployedAt`. A slow anchor no longer stretches the next epoch;
  closing epoch n+1 just waits for anchor n.
- The length is retargeted from the last `epoch_retarget_window` anchors
  (default 4): the fees burned per second over that span (the anchors'
  `windowBurned`), against `epoch_target_share` (parts per million of
  the ATN supply, `atnTotalSupplyAt` at the last anchor's block).
  Integer arithmetic.
- The floor is one DAO proposal lifecycle: voting delay + voting period
  + timelock delay, rebuilt from the governor's `VotingDelaySet` /
  `VotingPeriodSet` / `TimelockChange` and the timelock's
  `MinDelayChange` logs up to the anchor block.
- The ceiling is `epoch_ceiling_votes` (default 10) voting periods.
- No fees burned gives the ceiling. An unset target share (0, the
  default until set at deployment) gives the floor.
- The `epoch_length` key is gone.

**Items 5-6 DESIGN (agreed with the owner 2026-10-26, NOT BUILT): anchor
elections, pipelined.** This replaces the "two parallel proposals"
wording of items 5-6 and the superseding re-close. Bundled anchors
(one proposal for several epochs) were reviewed and dropped: three values
come from the anchor transaction itself (the fee pool, `prevAnchorHash`,
the REP snapshot), so every entry after the first in a bundle is unknown
at close time, and a dispute in one entry sinks all the later ones.

OpenZeppelin has no multi-option vote: its counting modules
(Simple/Fractional/Overridable) count Against/For/Abstain only. Agora's
Optimism Governor fork (MIT) has an `ApprovalVotingModule` that can run
single choice (`maxApprovals = 1`, `TopChoices = 1`), but one proposer
writes every option at `propose` time and it needs their modified
Governor. So we write our own, for `HomebaseDAO`:

1. **One election per epoch, single choice among options.** Each epoch
   gets one DAO proposal whose only action is
   `AnchorElection.anchorWinner(proposalId)`, so the proposal id is known
   in advance. It uses a counting module where `support` is an option
   index and tallies are kept per option; it replaces
   `GovernorCountingSimple` for anchor proposals. Quorum, the voting
   period and the timelock stay the Governor's.
2. **Each option comes from an elected proposer's own wallet.** During
   the voting delay, each proposer drawn for the epoch registers one
   option with `AnchorElection`. An option is {epoch number, cutoff,
   parent epoch root, epoch root, payload hash, mint root, cids}. A name
   on every option lets omissions be attributed, and no proposer can
   flood the epoch with variants. The number of proposers is the
   governed key `epoch_proposers` (default 3). Identical content from two
   proposers is one option.
3. **The draw is verified on chain** (Algorand-style threshold
   sortition, so no sorting of all households). A voting account is
   elected when
   `hash(seed, epoch, account) < 2^256 * epoch_proposers * share`,
   where share = the account's REP at the cutoff / total REP, and the
   account is above the governed REP floor `epoch_proposer_floor`. Each
   check is local (`getPastVotes`). The unit is the voting account: a
   household's agents delegate to the owner, as for Governor votes. The
   seed is keccak(epoch number, parent epoch root). The count elected
   varies around `epoch_proposers`.
4. **Voting.** Each voter backs the option matching its own close, after
   fetching by hash any batch an option names that it lacks. Voting
   for no option is the "against" (for example, every option omits a
   batch the voter holds). REP-weighted at the proposal snapshot. The
   winner is the most-voted option above quorum; a tie goes to the
   lowest option hash.
5. **Pipelined.** Epoch n+1's election opens as soon as its cutoff plus
   grace has passed (and its inputs' block is final), without waiting
   for anchor n. Its options name their parent root. At execution,
   `anchorWinner` takes the most-voted option whose parent is the chain
   head. It reverts while anchor n is missing, and the queued proposal is
   simply executed again later (a reverted execute leaves it Queued).
   Failures:
   - A transient failure (missed quorum, absent proposers) is re-posted
     for the same epoch with an attempt number in the description (a
     new proposal id). The later elections queued behind it stay valid.
   - A content dispute voids only the options built on the losing
     parent, which is correct: they were built on a wrong state.
6. **Contract changes** (ordinary code, owner 2026-10-26):
   - `Substrate.anchor` takes an explicit increasing cutoff;
   - `prevAnchorHash` drops block number and sender, so it can be
     computed ahead;
   - each epoch's fee pool = fees burned between its parent's cutoff and
     its own (a cumulative burn checkpoint by timestamp in
     `payForService`, some gas per payment);
   - Substrate accepts anchors from `AnchorElection`, a role the DAO
     grants once by proposal;
   - ToolBond's review epochs key on epoch number, not `anchorCount`.
7. **Off-chain changes:**
   - every close input is pinned to the epoch's cutoff (REP at the
     cutoff timestamp, everything else at the last block before it, once
     final) instead of the previous anchor;
   - daemons keep per-epoch state (carry-over, world, root) for every
     unanchored epoch, to rewind on a content dispute;
   - the retarget reads the anchored epochs' own windows.

REP readings split (BUILT 2026-10-26):
- The substrate (mint weight, drift, robot polls) reads REP **balances**
  per household (`RepToken.getPastBalance`, which existed already), so
  delegation never moves substrate weight.
- Governor votes and the anchor election read delegated votes.
- Only registered agents can be drawn as proposers: members without a
  daemon vote, but cannot compute a close.

BUILT (branch `epoch-rules-2`, 2026-10-26):
- **Contracts.** Substrate anchors take an explicit cutoff, use a
  content-only anchor hash (`anchorHashOf`), keep fee windows between
  cutoffs, and take the anchorer at construction. `AnchorElection` is
  live. HomebaseDAO's options counting and the AutonetSetup wiring are on
  the DAO side (trustless-contracts 6d1a5c6, 368e0ff, 5504969).
- **Daemon election driver** (`nodes/common/anchor_election.py`). It
  proposes, registers this daemon's option from a drawn agent, votes for
  the matching option, then queues and executes. A failed election is
  re-posted as the next attempt.
- **Inputs pinned to the previous cutoff.** `read_voice_state(pin=...)`
  reads REP at the cutoff, logs up to the last block before it, the pool
  as `burnedBetween` of the two cutoffs before it (equal to the on-chain
  `windowBurned`), and the retarget over the cutoff chain.
- **Pipelined closes** (`FederatedCloseDriver.chain_source`). Each epoch
  closes on this daemon's own view of the unanchored ones before it,
  kept in `epoch_chain.json` with their batches. An epoch anchored as
  closed here is pruned. One anchored differently rewinds: its batches go
  back to the buffer, and the anchored carry-over and world are installed
  before the next close. A joiner starts from the chain the same way.
- **Chain submission.** A pipelined option chains onto the parent's
  prepared fields, and elections for epochs settled by another option are
  dropped.

Open inside this design:
- Omission penalties need an on-chain record of which proposer's option
  omitted what. Until then, the sit-out is off-chain reports to the DAO.
- The seed is set by the parent's content; a proposer could grind it
  only by changing its batches, which voters would reject.
- A batch an anchored option left out is dropped once the next window
  opens (it is signed before that window's start). Decision item 6 says
  it rolls into the next epoch; that needs a rule for admitting
  backdated batches named by an omission report.
- Fetching a missing batch by hash before voting is not built: a daemon
  missing a batch votes for no option.

Rollout: these are `epoch_rules` version `"2"` (the existing Registry key).
Daemons on this build refuse `"1"`; the jurisdiction switches by one DAO
proposal, after the owner confirms the whole loop on local hardhat.
**FLAG-DAY:** payload encoding, epoch ids and close timing all change.

**Rollout BUILT (2026-10-26):** this build computes `epoch_rules` `"2"`
(`atn/config.py` `SUPPORTED_EPOCH_RULES`, `CURRENT_EPOCH_RULES`). An
unset key means `"2"`; a jurisdiction still on `"1"` is refused at join.
`tests/test_epoch_rules_v2_e2e.py` runs the loop end to end: two daemon
stacks (gossip, pipelined close driver, election chain submission) on
the real contracts (the DAO's HomebaseDAO with options counting,
AnchorElection, Substrate), several epochs anchored in order, then a
divergent epoch one daemon rewinds out of. Open on the DAO side: werule's
factory default `epochRules` is `"1"` today and must become `"2"`, or new
jurisdictions are refused by this build.

## Decision (2026-10-13): staked tools, flagged audits, REP-settled verdicts

> **Audit half superseded (2026-10-17).** The flagged audits, the vetted
> flag, the auditor share and the off-chain credibility number are replaced
> by staked reviews (docs/multi_jurisdiction.md decision 10, ToolBond
> 448023a): one review type with per-axis scores and harm, committed and
> revealed per agent per epoch, settled on chain by REP weight. The close
> reads ratings and the reviewer share from ToolBond pinned to the previous
> anchor. The author stake, novelty, successor and run-gate (bonded only)
> rules below still stand.

User-ratified 2026-10-13, NOT BUILT. Addresses the open problem of
malicious tools: someone else's code running on your machine. Design
principle: never assume honest actors are the majority; the network may go
through phases of being raided and must hold either way. Supersedes the
v4.1 "no vet gate" stance in one respect: an unvetted tool is still
published and discoverable, but nodes do not run it by default.

1. **Every published tool carries an ATN stake, per tool.** ToolBond
   evolves from the core-replacement gate into the stake contract for ALL
   tools (the core/non-core distinction is dropped: any tool running
   in-process has the daemon's full authority). Publishing stays a local
   gossip action; nodes cross-check `isBonded` on-chain and ignore
   unstaked tools. The stake amount is a jurisdiction parameter set at
   origination and changed by DAO proposal (ToolBond `setParams`). The
   stake persists across epochs, so there is always something behind a
   tool, including malware nobody has run yet.
2. **Novelty claim on every publish.** The author states what is new,
   naming the closest existing tools found via `probe_tools`. Reviewers
   score the claim on `correctness`; a copy with no real difference, or a
   claim that omits an obvious neighbour, fails it. Performance claims
   ("twice as fast") are part of the value proposition and checked the
   same way.
3. **Flags, not inferred audits.** A review does not imply a malware
   check. Auditors raise an explicit flag:
   - **green**: "I checked this for malware and it is clean";
   - **red**: must cite the exact malicious code.
   Flags are commit-reveal: a hash of the flag (the red flag's code
   reference inside it) is committed, and flags reveal once the verdict
   threshold is met, so nobody can copy a running tally. No green flag
   while a red flag is open without explicitly rebutting it. A flag's
   weight is the auditor's REP; auditors stake no ATN, and a zero-REP
   flag weighs nothing. Auditors keep their audit transcript locally and
   commit its hash (bound to the tool hash and pinned dependency hashes)
   with the flag; the transcript is evidence on appeal (whether it reached
   the payload separates an honest miss from complicity).
4. **Verdict settles by REP, not by epoch.** A verdict settles when the
   cumulative REP of the flagging auditors reaches a threshold (DAO
   parameter, e.g. 5% of total REP, retuned as the network grows). No
   arbiter: the REP-weighted crowd is the first court. Unsettled tools
   stay unvetted.
   - **Clean:** the majority auditors share the tool's earnings (DAO
     parameter, default 40% auditors / 60% author), split by REP
     (sybil-proof: splitting identities gains nothing). The auditor share
     flows from mints at the first close after the cooling-off.
   - **Malware:** the finder (first correct red commit) takes a fixed
     share of the stake (DAO parameter, default 50%); the confirmers
     (majority red flaggers) split the rest by REP, plus any slashed REP
     (author's and the wrong green flaggers'), reassigned to them by REP.
     The author's REP is slashed (full by default), per author. No new
     REP is minted.
   - **Minority** flaggers get their REP degraded by one DAO parameter,
     symmetric for wrong red and wrong green.
   - No standing security pool from burned fees ("vigilance always
     pays"). Every payout is bounded by what was slashed, which keeps
     plant-and-flag unprofitable.
5. **Cooling-off and appeal.** Payouts wait out the jurisdiction's appeal
   window (`economy.effectiveAppealPeriod()`, what ToolBond reads today).
   Anyone holding enough REP to submit a DAO proposal can appeal, borrowing
   the `ERC20Project` appeal model: the DAO is the higher court over the
   5% crowd. New evidence can reopen a clean verdict, since the stake is
   still there. Authors may unstake and unpublish behind the unbond delay,
   but never while a red flag is open. Unstaked means `isBonded` false:
   the tool leaves the network's economy (no search, flags or mint);
   local copies run at the owner's own risk.
6. **Versions.** No in-place updates: a new version is a new hash with its
   own stake and verdict. Only the original author can publish a successor
   that carries over the predecessor's reviews and ranking. Vetted status
   never carries: the successor shows as unvetted and nodes keep the last
   vetted version until it settles; auditors can review the diff. No fork
   concept: other authors publish new tools with their own novelty claim,
   and the original keeps its standing alongside them.
7. **What vetted unlocks (launch).** Nodes run vetted tools only by
   default; an unvetted tool needs explicit owner opt-in, per tool.
   Process isolation becomes the default. Manifests declare capabilities
   (network, file paths); reviewers check them on `correctness`; a real
   OS sandbox enforces them later. Until then a missed payload is covered
   economically (stake + slash), not by containment.
8. **Author REP is context, not rank.** `probe_tools` shows author REP,
   their other tools and past slashes next to each result; ranking stays
   review-driven.
9. **Composition credit.** The fixed `COMPOSITE_ROOT_SHARE = 0.7` is
   replaced by a reviewed split: reviewers estimate how much of the value
   is the composite's own logic versus its dependencies, REP-weighted.

**Open / later:**
- Bait attack: a zero-REP author baits honest auditors into green flags,
  then red-flags its own tool from a sybil to harvest their partial
  slashes. Bounded by small honest-miss slashes; needs an appeal path that
  takes the transcript as evidence. No existing contract covers it
  (`Economy`/`ERC20Project` is project escrow; ToolBond is arbiter + DAO
  overrule).
- Whether an unvetted tool mints for its author before its verdict
  settles.
- **Cross-repo:** REP slashing and reassignment need burn/move authority on
  RepToken (trustless-contracts).
- Parameter values: stake, threshold, split, finder share, degradation.

**BUILT 2026-10-13 (the close side: §4 Clean payout, §6 successor carry,
§9 composition split).** `federated_epoch_close` / `compute_tool_mint`
(`nodes/common/federated_reconcile.py`):

- **Auditor share.** New close input `vetting_snapshot`: {auditor_share_bps,
  tools: {digest: {round, auditors: {address: flag weight}}}}, listing only
  vetted tools (bonded, latest finalized verdict Clean, no malware verdict
  pending) with the revealed green flaggers of the round that vetted them.
  Of such a tool's mint, `auditorShareBps` (ToolBond param, default 4000)
  goes to those auditors split by flag weight; the author takes the exact
  remainder. Redistribution only: the pool normalization and the base-unit
  floor (`floor_mint_to_pool`) run on the redistributed map, so the sum
  stays <= the on-chain pool (dust: at most one base unit per recipient).
  Auditor addresses map onto an agent id the close already knows
  (case-insensitive), else the lowercased address. Unvetted tools mint
  exactly as before. `per_digest` gains `vetted` and `auditor_mint`.
- **Pinned read.** `voice_state.read_tool_vetting_snapshot` rebuilds the
  snapshot from ToolBond event logs up to the snapshot anchor's block (the
  same block as the owner map), so no archive node is needed and every
  daemon reads the same history; `auditorShareBps` comes from the last
  `ParamsSet` (the constructor emits one), else 4000. Wired through
  `read_voice_state(tool_bond_address=...)`, the driver's voice refresh,
  and `blockchain.tool_bond_address` (env `AUTONET_TOOL_BOND_ADDRESS`,
  seeded from registry.json `contracts.tool_bond`).
- **Reviewed composition split.** `attest_tools` judgments take an optional
  `own_share` in [0,1], gossiped on the `tool_used` event. A composite's
  root share is the drift-weighted (log1p(n) x rep_share x credibility, no
  ε floor, author and co-hosted reviewers excluded) running mean of those
  estimates, carried as `tool_positions[digest].own_share = {sum, mass}`
  (no new carry key; the bundle shape is unchanged). Updated before the
  fan-out, so this close's reviews count; damp-then-split is unchanged.
  No estimates: `COMPOSITE_ROOT_SHARE` (0.7).
- **Successor merit carry.** `manifest_meta` carries `version_of` when set.
  A manifest first appearing with `version_of` naming a registered
  predecessor by the same author starts from the predecessor's carried
  position (head, mass, own_share) and review book. A different author
  inherits nothing. Vetted status never carries (it is per digest on
  chain).

**FLAG-DAY:** close outputs change (auditor leaves in `agent_mint`, root
shares, inherited positions) and `manifest_meta` / `tool_used` gain
optional fields. Every daemon must run this build and read the same
ToolBond before the next close, or closes fork.

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

## Decision (2026-10-27): the owner wallet claims, the daemon operates

Ratified in discussion 2026-10-27. Supersedes item 5 above (the deferred
owner claim surface) and the per-agent claim it worked around.

1. **The household is the owner wallet, and it is the only claimant.**
   The close keys every mint leaf by the proven owner of the earning
   identity (agent → `agentOwner`, pinned like the household damping
   reads), summed per owner. Owner-authored tools (author `user` → the
   owner wallet) are no longer stranded. ATN lands on the owner; agents
   are funded by their owner, as "the human owns the wallet" says.
2. **Claims are permissionless.** `claimFor(owner, amount, epochIdHash,
   proof)` replaces the agent-only `recordTrainingForEpoch`: anyone may
   submit it, the merkle proof fixes the amount and the ATN can only go
   to `owner`. The daemon submits it and pays the gas, so automatic
   claiming survives without the daemon holding the owner key. Still
   idempotent per (claimant, epoch).
3. **REP follows the ATN.** REP is claimed 1:1 on an address's ratified
   earnings, so the owner holds the household's REP. The REP claim's
   agent-only gate is dropped; an agent's REP is minted to its owner.
   (Service earnings later left the REP base: see "REP comes from the
   tool pool only" below.) Agents still receive service ATN to spend.
4. **The daemon operator key.** Once an owner wallet is associated, the
   daemon generates its own key and registers it on Substrate, bound to
   the owner. It is not an agent: no fleet entry, no model, no earnings.
   It proposes the daemon's close when drawn, votes, and submits claims.
   AnchorElection keeps its `isLive` gate, so only addresses with a
   registered daemon are drawn as proposers.
5. **Voting power by delegation.** The owner delegates REP votes to the
   operator key once, in werule's user section (revocable there). The
   draw and the governor both count delegated votes, so the operator is
   drawn and votes with the household's REP. Owners may instead give the
   daemon their own key (server setups). An undelegated owner can still
   vote directly in werule but is never drawn as a proposer. Shares stay
   with the owner; the daemon is the proxy voter.
6. **Wallet before registration is a daemon rule, not a chain rule.** The
   chain only checks that some address signed as owner; a throwaway key
   works, so an agent bound to a fake owner is a household of one. That
   is priced by the per-household damping and REP weighting, not
   prevented. The daemon and app do not offer on-chain registration
   before an owner wallet is configured.

**FLAG-DAY:** the mint leaves change (keyed by owner), the claim ABI
changes (`claimFor`), and the REP claim rail changes. Every daemon must
run the new build on a fresh Substrate before the next close.

## Decision (2026-10-27): an epoch ends on volume, not on a retargeted time

Ratified in discussion 2026-10-27; BUILT same day
(`voice_state.volume_cutoff`, `FederatedCloseDriver`). Replaces item 4 of
Decision (2026-10-26) ("length set by activity, Bitcoin style").

- The retarget estimated a length from the previous epochs' fee rate.
  Substrate already meters fees on chain (`burnedBetween`), so the epoch
  now simply ends at the first second after the previous cutoff when the
  fees burned since it reach `epoch_target_share` of the ATN supply (the
  supply pinned at the last anchor). Every daemon reads the same on-chain
  checkpoints and lands on the same second; only time one grace old is
  read, so the burns it sees are final.
- Floor (one anchor lifecycle) and ceiling (`epoch_ceiling_votes` voting
  periods, so a quiet network still pays out) stay. Unset target: every
  epoch lasts the floor.
- No contract change. `epoch_retarget_window` is retired (still read, so
  an old key is harmless). The first epoch needs no time parameter: it
  ends on volume like any other.
- Tradeoffs accepted: burns are lumpy (a channel close settles many
  payments at once), so epochs can end in bursts; and anyone can end an
  epoch early by burning fees, which costs real fees that join the pool.

## Decision (2026-10-27): containment follows the network's judgement

Ratified in discussion 2026-10-27; BUILT same day (`atn/tool_security.py`
`containment_tier`, `ToolStore.call`). Amends "numbers are exposed, never
gated on" (decision 10 of `docs/multi_jurisdiction.md`) for containment:
the network's settled judgement of a tool's exact code now decides how
much that code may do on every node.

1. **Whose code.** Your own household's tools run on the authored path.
   Containment is for network tools (adopted, authored elsewhere). An
   external harness over MCP is the owner's own agent, not a stranger.
2. **Tiers, per digest,** from the anchor-pinned ToolBond review snapshot
   (settled harm on the 0..100 scale and the REP mass behind it, against
   REP supply), so every node lands on the same tier:
   - **denied:** harm at or above `deny_harm` with at least
     `trust_mass_bps` of supply behind it. Refused at adoption and at run.
   - **declared:** harm below `trust_harm` with that same REP behind it.
     The manifest's declared net/fs/spawn apply.
   - **strict:** everything else (new, thinly reviewed, contested): no net,
     no files outside its sandbox, no spawn, whatever the manifest says.
   One reviewer can neither deny nor free a tool: both ends need REP mass.
   Content addressing makes this safe: an update is a new digest and
   starts strict.
3. **Never the vault.** Even declared fs never reaches the daemon's data
   dir or the keystore. With fs off, deletes, renames and listings are
   refused too, not only opens.
4. **Refusals are evidence.** Every network-tool run is traced. The guard
   records a refusal before raising, so a tool that swallows the error is
   still caught: the refusal comes back in the `use_tool` result
   (`containment.denied`) and raises the security alarm. Reviewing agents
   see it, which is how harm reviews (and so the tier) catch up.
5. **Parameters** (`jurisdiction.<id>.containment_deny_harm`,
   `containment_trust_harm`, `containment_trust_mass_bps`) are DAO-set.
   Defaults 50, 10 and 200 bps are PROVISIONAL.
6. **Tested live:** the WAN run publishes a canary (a useful tool that
   quietly reads a planted secret and beacons out) next to an honest
   control, without telling the fleet, and measures how fast consensus
   denies it.

## Decision (2026-10-27): REP comes from the tool pool only; reviews weigh the household's REP

Ratified in discussion 2026-10-27 after the S0 local run
(`scripts/s0_local_run.py`). Two consequences of the household model
(the owner wallet holds the household's REP, agents hold none):

1. **Service revenue earns money, not voice.** `AutonetRepClaim`'s
   earnings base is `agentMintTotal` alone; `serviceEarnings` stays on
   Substrate as money and as the provider's track record, and mints no
   REP. S0 measured the old base: households paying each other bought REP
   at the fee rate (2.5 ATN of fee for 97.5 REP; owners went from 250 to
   about 1,700 REP in 35 minutes). REP from the tool pool costs real
   burned fees, so the cheap path is closed without raising the fee on
   honest volume. This retires the "known residual exposure" of item 8
   in the v4.1 section and the matching entry in `docs/resilience.md`.
   What remains is fake usage farming the tool pool, bounded by the
   usage dampers and by reviews, not priced by fees; S0 is the place to
   measure it.
2. **A ToolBond review weighs its household's REP.** The weight is the
   reviewer's bound owner's REP at the epoch's snapshot (trustless-contracts
   d2f32ac). One weight per household per tool per epoch, carried by the
   household's earliest commit (commits are hidden, so their order carries
   no information); REP slashes land on the household. Before this, every
   review weighed nothing, so ratings, harm and containment never settled.

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

- Tools crystallize **procedures**. Coordination (decomposing the
  task, selecting among thousands of tools, interpreting failure)
  is itself cognition and stays with the model. Discovery ranking
  attacks the selection half; the decomposition half it cannot.
- Open-ended synthesis (novel reasoning, judgment, ambiguity) does
  not factor into pinned code. This is the two-plane doctrine's
  boundary restated economically: substrate = retrieval + procedure,
  LLM = judgment; the ratchet lowers the floor for the toolable
  fraction of work and grows that fraction, but it never reaches 1.0.
- Small models are today measurably worse at multi-step tool
  coordination; the ratchet bets that shrinking per-step depth
  outpaces the coordination burden of a bigger library. That is an
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

**[Deciding factor and the three preconditions SUPERSEDED by Decision
(2026-10-12) below: swaps are live behind core-replacement approval, a
ToolBond bond and a per-agent loadout. The containment analysis stands.]**

The section above says "customization = forking a distro (swap a dep)".
That was true of the ECONOMICS and false of the EXECUTION, and the gap
went unnoticed because the two halves were built at different times.

**What was actually there.** The 18 `atn_*` module manifests are real,
pinned, content-addressed records with code blobs, but their blobs are
`inspect.getsource` of daemon internals, i.e. IDENTITY, not programs.
None implemented the sealed stdin/stdout protocol; invoking one through
`_call_pinned` would have hung on `stdin.read()`. The distro DAG,
adoption credit, and mint fan-out all worked. Nothing could be swapped in.

**Why most bundles can never be executable.** Every one of the 63
executors takes `(runtime, input)` and reaches live daemon state:
`runtime.list_agents`, `runtime.tool_registry`, `runtime.get_agent`. A
pinned tool is a subprocess with a JSON pipe; it has no `runtime`.
Making `atn_delegation` "executable" would mean an RPC channel handing a
subprocess daemon authority, a privilege-escalation surface, not a
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

- `tool_guard.py` has exactly three checks (`socket.*`, open-outside-
  prefix, and a spawn-event tuple). A shell bundle needs net, fs, AND
  spawn by definition, so every branch falls through and the guard is a
  literal no-op for this tool class. The destination allowlist does not
  help: `spawn: True` defeats it (curl in a child process is unaudited).
- The exposure is uniquely high here, not merely equal to other tools. A
  substituted `bash` sees every command string (git remotes, ssh,
  curl-with-token-in-URL); a substituted `read_file` sees every file body
  the agent reads, the plane where credentials actually live, and
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

### Decision (2026-10-12): swappable harness parts; code swaps allowed behind approval + bond

**Supersedes the 07-29 deciding factor** ("core logic upgrades only via
daemon release"). The goal (user-approved): every part of the harness is
swappable through the tool library and the market (reviews, ratings)
selects the best; the daemon keeps only a minimal safety kernel (budget
gate, tool dispatch, sandbox, governance heartbeat, history storage). The
07-29 containment analysis is NOT retracted: a substituted `bash` /
`read_file` still sees every command and file body, and the audit hook
still cannot contain a net+fs+spawn tool. What changed is who carries
that risk: an explicit owner act, an on-chain stake, and signed evidence.

1. **Loop policy is a distro part.** `atn/harness_policy.py`
   (`HarnessPolicy`): system prompt templates, compaction instruction +
   reduction ladder, verify/review closing turns, loop-guard limits,
   headroom thresholds, parallel-safe list, tool bundle selection
   (narrow only). The distro manifest's code blob is now JSON
   (`kind: atn_harness_distro`, `modules`, `policy`); the reference
   distro carries the reference values, built from the live constants,
   so the default is byte-identical. Forking = copy the blob, edit the
   policy, register with the same (or swapped) dependencies. Overlay
   semantics, validation and the kernel boundary: docs/agentic_loop.md
   §25.
2. **Per-agent loadout binding.** `AgentDefinition.loadout` (distro
   digest, "" = reference) is set by the OWNER only (`update_agent
   {loadout}` from the owner surface; any agent caller is refused; the
   atn_web agent settings show a distro picker fed by the owner-only WS
   `list_distros`). `active_loadout` is now just the daemon's reference
   distro; attestations carry `ToolStore.loadout_for(caller)`, the
   agent's actual loadout.
3. **Code swaps are allowed** (`runtime/shell_provider.py`,
   `SHELL_SWAP_ENABLED = True`, kill switch `ATN_CORE_SWAP_DISABLED=1`).
   A pinned tool whose `capabilities.provides` claims core built-in
   names replaces them for an agent only when ALL hold:
   (a) **core-replacement approval:** it was adopted with an explicit
   owner acknowledgment; `approve_adoption` rejects such a tool unless
   the approval carries `core_replacement_ack` (WS field, CLI
   `approve <digest> --core-replacement`), and the UI shows a second
   warning modal; proposals carry `core_replacement: [names]`. Locally
   authored records cannot replace core tools.
   (b) **bond:** `ToolBond.isBonded(manifestDigest)` is true
   (`runtime/tool_bond.py`, address from registry key `tool_bond` or
   `ATN_TOOL_BOND_ADDRESS`, 60s/15s cache). No address or any RPC fault
   = refused (fail closed); `ATN_TOOL_BOND_DEV_OVERRIDE=1` is the local
   testing escape hatch only.
   (c) **per-agent binding:** the provider is a dependency of the
   agent's owner-bound loadout (the reference distro never swaps).
   Providers must be sealed (no dependencies). A refused gate falls back
   to the built-in, never to an error. In worker isolation, swapped names
   route to the daemon (`framework_tool` RPC) so gates and evidence stay
   daemon-side. Bridge (SDK-native) tools are not swappable.
4. **Policy as code**: deferred. The hook names (`policy:compaction`,
   `policy:system_prompt`, `policy:review_step`) are declarable in
   `capabilities.provides` and kept by registration, but nothing
   dispatches them yet (`POLICY_HOOKS` documents the intended contract).
5. **Evidence for disputes.** Every swapped call records a trace: tool
   digest, input, output (64k cap), timestamps, and what the guard
   observed in the provider process (file opens outside the interpreter
   tree, socket destinations, spawn argv; 5000-line cap). It is signed by
   the agent key (EIP-191 over canonical JSON; unsigned when the agent
   has no key), stored as a blob, logged, stamped on the record
   (`last_evidence_digest`) and indexed (`evidence_index.jsonl`, WS
   `core_replacement_evidence`). That digest is the `evidenceCid` for
   `ToolBond.accuse`. NOT captured: what spawned children do beyond
   their argv, native-code bypasses of the audit hook, kernel-level
   activity. The trace proves what the provider was given and what it
   did in Python, which is the claim/behavior surface a CON needs.

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

## Decision (2026-10-26): the venture vault is scrapped

The venture vault (`contracts/core/VentureVault.sol`), its verifier-trial
rail (`atn/trial_runner.py`, the `run_trial` tool, `attestTrial`) and the
venture-loop E2E are removed. Reason: no real use case. Capital in ATN is
not anyone's bottleneck (an agent's costs are mostly fiat inference bills),
no market of backers for small agent businesses exists, and the
jurisdiction's trustless projects already fund work. The design history is
in git (the 2026-07-05 "Verifier trials" section and the contract NatSpec).

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
  2026-07-10; the mint leaf is the 2-field `(agent, amount)` leaf (the
  authoritative payload is schema 4 since the carry-over commitment,
  below). `rep_shares` (the un-ε-floored raw reputation share, read from
  RepToken) survives as an input, but drives POSITION DRIFT weight only.
- Carry-over commitment (authoritative payload schema 4): the five
  carry-over maps derive from raw canonical events that are never
  published, so a daemon that first boots after anchors exist cannot
  replay them. Each close therefore serializes the maps it produced (the
  next close's inputs) into one canonical bundle blob
  (`encode_carry_bundle`: compact JSON, sorted keys, shortest-repr
  floats; key order is not semantic here because every consumer iterates
  sorted) and commits its sha256 as the payload's `carry_cid`. Every
  daemon publishes the bundle like the world checkpoint. A joiner whose
  carry-over files are missing reads the latest anchor, verifies the
  payload against the on-chain `payloadHash`, fetches the bundle from
  any peer, verifies sha256 == `carry_cid`, installs the five files and
  closes normally (`state_sync.fetch_carry_over_from_chain`). It installs
  only a bundle whose anchored `epoch_root` equals the canonical root of
  the epoch it observed immediately before the close it is about to run
  (epoch ids are per-daemon, roots are shared). So a fresh joiner always
  observes one full epoch first, and a lagging anchor (the last winner
  failed to anchor), a partially observed epoch, or an anchor for the
  epoch being closed all keep it observing instead of installing a stale
  or double-applied carry-over. It also keeps refusing with a clear log
  line when the latest anchor predates schema 4. This replaced the
  operator seed workaround (`ATN_CARRY_OVER_SEED_DIR`). Blobs are served
  by ONE host-level `/autonet/blob/1.0.0` handler (`HostBlobServer`):
  every `LibP2PBlobResolver` and the `BlobStore` share it instead of each
  replacing the host's handler.
  **FLAG-DAY:** the payload schema change (4, new `carry_cid` field)
  changes the anchored payload bytes and hash, so every daemon must run
  the new build before the next close; mixed builds fork on payloadHash.
  No contract change or redeploy: Substrate stores only the payload hash
  and never parses the payload.
  **Known gap (keyless daemons):** the bundle (like the world checkpoint)
  is published, and the joiner's blob resolver is obtained, through the
  chain-submission driver, which only attaches when the daemon has a
  `private_key`. A read-only daemon (chain RPC but no key) therefore
  neither serves the bundle nor can fetch it to join; it stays in
  observe mode. Left as is for now.
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
