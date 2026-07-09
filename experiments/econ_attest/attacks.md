# Red-team: Substrate v3 tool-economy incentive design

Scope: the tool-mint / position-drift / vetting rails as built, not as
documented. Where doc and code disagree I trust the code. All line cites
are to the files below at the time of writing.

- `nodes/common/federated_reconcile.py` — `compute_tool_mint` (mint,
  fan-out, vetting greenlight, position drift), `federated_epoch_close`.
- `nodes/common/world_model_substrate/tool_usage.py` — usage / vet / axis
  aggregation from events.
- `nodes/common/world_model_substrate/infer.py` — `_infer_artifacts`
  discovery ranking.
- `nodes/common/voice_state.py` — the household voice map.
- `atn/tool_store.py` — `vet_tool` (local self-vet reject).

## The single fact that changes everything: voice = reputation, not ATN

The `tool_substrate.md` addendum (lines 89-153) says voice weight =
`ε + household_ATN / supply`. **The code does not do this.** `voice_state.py`
lines 1-52, 218-343 compute:

```
weight(house) = ε + household_REPUTATION / reputationTotalSupply
```

and reputation is **soulbound, minted ONLY by `recordTrainingForEpoch`**
(voice_state.py:24-30, 300-327). Reputation is minted in lockstep with ATN
tool-mint (CLAUDE.md dual-token; Substrate.sol). So the voice denominator
is *earned network standing*, and the ONLY way to earn it is to have
previously won tool-mint through anchored consensus.

This makes the central sybil defense a **bootstrapping problem for the
attacker, not a runtime cost**:

- A fresh identity (new agent, new owner wallet) has zero reputation →
  voice weight = `VOICE_EPSILON = 0.05` (federated_reconcile.py:106,
  479-481).
- `voice_weights` multiplies BOTH the mint term (line 482) and the
  position-drift mass (line 603). A voice that can't mint can't drift.
- Therefore every attack below that relies on fabricated identities is
  capped at ε per household, and ε-households can only *earn* their way
  above ε by first landing real mint — which requires being reviewed/used
  by *non-ε* households. This is a genuine, load-bearing defense. The
  attacks that beat it are the ones that (a) exploit the ε floor at scale,
  (b) exploit the vetting rail which is NOT voice-weighted, or (c) don't
  need fabricated identities at all.

`voice_weights=None` (no chain, e.g. epoch 1 or a private/offline swarm)
makes every household weigh 1.0 (line 478). **Every attack below is
dramatically more profitable in the pre-chain / weights=None regime** —
that is the real soft underbelly, called out per-attack.

---

## Attack 1 — Sybil review ring (pump position + mint via mutual use/review)

**Mechanism.**
1. Operator stands up N agents. Each publishes a tool, gets it greenlit
   (see Attack 3 for how greenlight is beaten), then all N agents call and
   `attest_tools` (attested-ok receipts, `axes` all +1 on correctness /
   simplicity) each other's tools.
2. At close, each attesting agent is a "caller"; `usage_term` sums their
   damped counts (federated_reconcile.py:395-409, 471-487); axis reviews
   drift the head toward +1 on the usefulness axes (lines 550-610).
3. Drifted head lifts discovery: `final = base·(1+tanh(rating))`,
   `rating = mean(correctness, simplicity)` of the drifted head
   (infer.py:321-327). Ring tools out-rank honest tools → real callers
   find them → real mint.

**Code-grounded feasibility.** Two independent defenses must both be beaten.

- *Owner-map household collapse* (lines 387-409, 473-476). If all N agents
  are bound to ONE owner wallet, `_household()` collapses them to one key,
  counts pool BEFORE log1p, and the author's own household is excluded
  (line 474). A same-owner ring nets **exactly zero** self-mint. So the
  ring must use *distinct owner wallets* — free to create, but see next.
- *Voice weighting* (lines 478-482). Distinct throwaway owners each weigh
  ε=0.05 and have zero reputation. N sybil reviewers contribute
  `Σ log1p(count_i)·0.05`. To move a tool's mint by 1.0 unit of
  voice-weighted usage you need `Σ log1p(counts)·0.05 ≥ 1`, i.e.
  ~20 log1p-units spread across ε-households. Because log1p saturates,
  per-household volume is near-useless (log1p(1000)=6.9), so you need
  **breadth: ~3 fresh ε-households each attesting ~a few times** gets you
  ≈ 0.05·3·log1p(3) ≈ 0.2 mint — small but nonzero, and it is REAL ATN +
  REAL reputation. That reputation then raises those households above ε
  next epoch: **the ring bootstraps itself.** This is the accepted
  "known-open risk" (tool_substrate.md:79-87) but the code shows it is
  *self-amplifying*, not merely present: epoch k's ε-mint becomes epoch
  k+1's super-ε voice.

Position drift is even cheaper than mint because drift is a *weighted
average*, not a sum: `head' = (mass·head + Σw·score)/(mass+Σw)`
(lines 606-610), author prior mass 1.0 (line 557). A handful of +1
reviews from ε-households with `w=log1p(n)·0.05` still pulls a
low-mass (new) tool's head measurably toward +1, because the prior mass is
only 1.0. **Ranking is corruptible by ε-voices even when mint is not** —
the drift denominator has no ε-scaled floor protecting it beyond the
author prior. Discovery pollution is the cheaper, higher-leverage half of
this attack.

- Which defense it must beat: voice-weighting (mint) + author-prior-mass
  (drift). It *partially* beats drift (low-mass tools), does NOT
  meaningfully beat mint in the reputation regime, and **fully beats both
  when `weights=None`**.

**Payoff sketch.** Cost: N wallets (free) + N greenlights (real vetting
cognition, ~the expensive part) + real inference for each fabricated
attestation (the "anti-wash floor price", tool_substrate.md:304). Gain in
the reputation regime: O(ε·N·log1p) mint per epoch, compounding as sybil
reputation accrues; discovery lift is larger and near-immediate. Gain in
weights=None regime: unbounded (each sybil weighs 1.0).

**Sim-testable prediction.** Sweep N distinct-owner sybils, each attesting
its ring's tools m times per epoch, over T epochs, in BOTH regimes.
Measure: (i) sybil-tool mint share vs an honest control tool of equal true
quality; (ii) sybil-tool discovery rank; (iii) the epoch-over-epoch
trajectory of sybil household voice weight. Prediction: in reputation
regime, mint share stays < a few % and grows *slowly* (bootstrapping), but
discovery rank crosses the honest control within 1-2 epochs; in
weights=None, mint share tracks N linearly. Refuted if drift rank does NOT
cross the control at low N.

---

## Attack 2 — Self-composition depth farming

**Mechanism.** Author declares a chain of own tools as deps
(`root → dep1 → dep2 → …`) hoping each attestation of the root fans mint
out across all of them, multiplying the author's take.

**Code-grounded feasibility. This is closed by construction — damp-then-split.**
`_composition_shares` (lines 151-182) splits ONE unit of weight over the
DAG: root keeps 0.7, remainder splits among deps, recursing to depth 4,
sum ≤ 1.0. Critically the per-caller count is damped FIRST (line 406
`damped = math.log1p(counts[house])`) and THEN multiplied by shares
(lines 407-409). Since Σshares ≤ 1, total credit = `damped·Σshares ≤
damped`. Padding with your own deps just *moves* your own damped credit
around within your own household — and since all the deps share the SAME
author household, they all land back on the same `author_house` and are
summed (lines 516-524). **Net for a same-author chain: identical to a
single tool, minus any forfeited cyclic/unregistered-dep share** (lines
176-179 forfeit, never redistribute). So self-composition is neutral at
best, self-harming if any dep is cyclic/unregistered.

The concavity exploit the doc warns about (`log1p(0.7)+log1p(0.3) >
log1p(1)`) is genuinely prevented by the operation order. I checked: there
is no path where log1p is applied post-split.

- *Real* (small) opening: composition pays a **royalty to OTHER authors'
  imported deps** (lines 516-524, the dep-author gets their share on their
  own node). This is intended. The only farm is: if you can get honest
  callers to attest a *composite* you built on your own deps, your deps
  earn their fan-out share too — but that share is carved OUT of the 0.7
  you'd otherwise keep, so it's still conserved. No amplification.

**Payoff sketch.** Attacker cost: authoring + greenlighting extra tools
(real). Gain: zero to negative. This attack does not work as designed.

**Sim-testable prediction.** Construct a same-owner author with a
depth-4 self-chain vs a single flat tool, feed identical attestation
streams. Measure total author mint. Prediction: equal to within rounding
(OUTPUT_DECIMALS), and strictly less if any declared dep is
unregistered/cyclic. Refuted if the chain ever earns more than the flat
tool.

---

## Attack 3 — Vetting collusion (fake "distinct fleets" for greenlight)

**Mechanism.** Greenlight requires `Σ over distinct fleets of best vet
weight ≥ VET_QUORUM=2.0` (lines 350-364). Operator wants to greenlight
its own (possibly malicious) tool cheaply.

**Code-grounded feasibility. This is the softest spot in the whole rail.**
"Distinct fleet" is computed as (line 356):

```python
fleet_key = owner_map_all.get(vetter, "") or vetter
```

i.e. a vetter's fleet is its **owner wallet if bound, else its own agent
id**. Two escapes:

1. **Unbound vetters are each their own fleet.** If the operator's vetting
   agents are simply *not owner-bound*, each one is a distinct fleet with
   `owner_map.get = ""` → falls back to distinct `vetter` id. Two unbound
   throwaway agents = two distinct fleets = quorum met. The self-vet /
   same-owner exclusions (lines 329-341) only fire when the vetter equals
   the author or shares the author's *bound* owner — an *unbound* sybil
   vetting a *bound* author's tool passes all three checks.
2. **Vet weight is NOT voice-weighted.** Unlike mint (line 482) and drift
   (line 603), the greenlight sum uses `weight = 1/(1+busts)` (line 357),
   which is **1.0 for any never-busted agent regardless of reputation or
   balance.** ε does not touch vetting. So a zero-reputation sybil casts a
   *full-weight* vet. Two fresh sybils clear VET_QUORUM=2.0 for free.

The only cost is the local self-vet reject in `vet_tool`
(tool_store.py:1430-1434) — trivially sidestepped because it only blocks
vetting through the *same daemon/agent* as the author; a separate agent
process vetting by digest passes.

**This is the load-bearing failure.** Greenlight is the gate the doctrine
leans on hardest — tool_substrate.md:85-87: "Covert harm invisible to
satisfied users has ONLY the vetting entry gate as its dedicated defense."
And that gate is beatable by two unbound sybils at zero economic cost. The
bust/claw-back that was supposed to punish bad vets is **DORMANT** in v3
(federated_reconcile.py:343-348 — no CON signal exists to trigger it), so
there is no downside to greenlighting your own malware.

- Which defense it must beat: distinct-fleet collapse + self/same-owner
  exclusion + bust deterrent. It beats all three: unbound → distinct;
  unbound → not same-owner as bound author; bust → dormant.

**Payoff sketch.** Cost: 2 throwaway agents, 2 vet reports (cognition, but
cheap — the report is unvalidated free text, tool_store.py:1427-1429 only
checks non-empty). Gain: greenlit status = mint-eligibility + adoption
recommendation for an arbitrarily malicious pinned tool. This is the
highest-severity, lowest-cost attack in the set.

**Sim-testable prediction.** Register a tool authored by a bound agent;
greenlight it with K unbound zero-reputation sybil vetters. Measure the
minimum K to cross VET_QUORUM and whether any exclusion fires. Prediction:
K=2 greenlights with no exclusion triggered and no bust. Then measure
whether making the vetters *owner-bound to a shared wallet* changes it:
prediction — collapses to one fleet, greenlight fails. That delta is the
fix surface (require bound owners for vet weight, or voice-weight vets).

---

## Attack 4 — Flood-and-bury (dilute rivals / spam discovery)

**Mechanism.** Publish many near-duplicate tools to (a) steal a rival's
pool share, or (b) crowd the rival out of `probe_tools` results.

**Code-grounded feasibility.**
- *Is mint zero-sum within an epoch?* **No, not at the compute_tool_mint
  stage** — each digest's `usage_term` is computed independently from its
  own attestations (lines 442-524); there is no per-epoch normalization
  there. BUT `federated_epoch_close` applies an `emission_pool`
  (federated_reconcile.py:810-821, `apply_emission_pool`) that normalizes
  all agent mints to shares of a **fixed pool** (`BASE_EMISSION_PER_EPOCH
  = 100` + recycled fees, voice_state.py:340). **So mint IS zero-sum at
  the pool stage.** Flooding with tools that attract *real attestations*
  dilutes everyone's per-unit payout. But near-DUPLICATE tools with NO
  real usage earn nothing (usage_term=0 → skipped, line 488), so flooding
  the *registry* alone does not dilute the pool — you must also generate
  real (voice-weighted) attestations, which returns to Attack 1's ε cost.
  Flood-to-dilute-pool is therefore weak in the reputation regime.
- *Discovery spam is the real payoff.* `_infer_artifacts` returns the
  top-k by `final` (infer.py:354-356). Candidates come from
  `artifact_index.search(query, k=k*3)` (line 275) — pure embedding
  cosine, NO voice weighting, NO greenlight filter at the candidate stage.
  So an attacker who registers many manifests whose *claimed embedding*
  sits near a hot query occupies candidate slots for free. Greenlight only
  gates MINT (line 460), not RETRIEVAL — un-greenlit spam still appears in
  `probe_tools` output. The drifted-head lift (line 327) can't save honest
  tools here because spam tools with zero reviews get `rating=0 →
  final=base·(1+tanh(0))=base·1`, i.e. they rank on raw cosine, and a
  keyword-stuffed manifest can win raw cosine.

- Which defense it must beat: emission-pool zero-sum (for dilution — holds,
  attack weak) vs candidate-retrieval (for spam — NO defense, attack
  works). The density blend (infer.py:306-313) is *floored at claimed
  cosine* (line 310, `max(cosine, …)`) so it can only LIFT, never sink a
  SEO manifest — the doc explicitly flags this as unresolved
  (infer.py:298-302).

**Payoff sketch.** Dilution: cost = real attestations (ε-expensive), gain
= marginal pool shift; not worth it. Spam: cost = registering M manifests
(cheap, content-addressed but distinct text = distinct digest), gain =
occupying M of the 3k candidate slots for a hot query, pushing the honest
tool below the top-k cut. High leverage, low cost.

**Sim-testable prediction.** Seed an index with one honest tool and M
SEO-manifest duplicates near a query; run `_infer_artifacts(k=5)`.
Measure the honest tool's rank as M grows. Prediction: honest rank
degrades monotonically with M and falls out of top-k with modest M,
independent of voice/greenlight. Refuted if greenlight or drift filtered
the spam from candidates (it does not, per code).

---

## Attack 5 — Competitive review nuking (down-review a rival)

**Mechanism.** Attacker uses a rival's tool once, `attest_tools` with
`axes` all −1, to drag the rival's drifted head negative → sink its
discovery rank.

**Code-grounded feasibility.** Reviews ride attested-ok receipts
(tool_usage.py:105-125); a negative-axis review still requires `ok=True`
(line 105) and an actual invocation, so the attacker must genuinely call
the tool. Drift is voice-weighted (federated_reconcile.py:591-603): a
throwaway ε-household's −1 review moves the head by
`w = log1p(n)·0.05` against a prior mass 1.0 + accumulated honest mass.
Against a *popular* tool (high mass from many honest reviews) an
ε-nuke is negligible (mass inertia, lines 606-610 — the doc's "proportional
inertia", tool_substrate.md:64-65). Against a *new* rival (mass ≈ 1.0) a
single −1 from even an ε-household shifts `head` to
`(1·0 + 0.05·log1p(1)·(−1))/(1+0.05·0.69) ≈ −0.033` — small but it drops
`rating` below zero, and `1+tanh(rating) < 1` *penalizes* the rival's
cosine (infer.py:329 path is for non-manifests; line 327 for manifests
uses `rating`). So **early-stage tools are nukeable, mature ones are not.**

Asymmetry worth noting: to PUMP your own tool you must beat the same-owner
exclusion (can't self-review, lines 474/586), but to NUKE a rival there is
NO exclusion — the attacker is by definition a different household. Nuke
is *structurally easier than pump* per unit of voice. The only brake is ε.

- Which defense it must beat: voice-weighting + mass inertia. Beats them
  for low-mass (new) tools, loses to them for high-mass tools. Fully wins
  under weights=None.

**Payoff sketch.** Cost: one real invocation + one attestation per nuking
household (ε-cheap). Gain: suppress a nascent competitor before it
accumulates protective mass — a *timing* attack on new entrants. Combined
with Attack 3 (greenlight your replacement) this is a viable
"kill-the-newcomer, ship-your-clone" play.

**Sim-testable prediction.** Two equal-quality new tools; attacker directs
J ε-households to −1-review one of them each epoch. Measure the divergence
in discovery rank and the mass at which nuking stops moving the head
appreciably (define "immune mass"). Prediction: rank divergence is large
while mass < ~5, negligible after; immune mass is reached faster with more
honest usage. Refuted if ε-weighting flattens the divergence even at
mass ≈ 1.

---

## Attack 6 (novel) — ε-floor mint faucet at scale ("dust minting")

**Mechanism.** The ε floor (0.05) is a *per-household* guaranteed voice
for ANY identity, funded from a *fixed* emission pool. Create K distinct
unbound owner-wallet households, each authoring one cheap real tool and
cross-attesting the OTHERS' tools (not their own — dodging the same-owner
exclusion). Each honest-looking attestation carries voice ε. With K large,
Σ ε·log1p across the fabricated crowd becomes a material, *permanent*
share of the fixed 100-ATN pool — skimmed from honest authors every epoch.

**Code-grounded feasibility.** The ε default for unknown households is
hard-wired (line 481, `voice_weights.get(house, VOICE_EPSILON)`). Nothing
caps the NUMBER of ε-households, and the emission pool is fixed and
zero-sum (Attack 4). So the attack is: convert "many free identities" into
"many ε-voices" into "a fixed fraction of a fixed pool." Unlike Attack 1
this doesn't need the ring tools to out-rank anything — it just needs them
greenlit (Attack 3, free) and mutually attested. Each ε-household's take is
tiny, but it is `K·ε·log1p(m)` summed, and it *recurs every epoch* and
*compounds* as the dust reputation lifts households above ε. The doc
acknowledges "damage bounded at ε per fabricated identity" (line 121) —
but ε·K is unbounded in K, and the "real economy outgrows it" claim is an
assumption the fixed pool does not enforce.

- Which defense it must beat: ε floor itself (by definition it exploits,
  not beats, the floor) + greenlight (Attack 3) + same-owner exclusion
  (dodged by cross- not self-attestation). Voice-weighting is the intended
  defense and it *bounds per-identity* but not *per-operator-with-K-identities*.

**Payoff sketch.** Cost: K wallets + K greenlights (2 unbound sybil vets
each, free) + K real cheap attestations/epoch. Gain: a persistent
`≈ K·ε·log1p(m) / poolTotal` slice of every epoch's 100 ATN, growing as
dust reputation accrues. The break-even is where the inference cost of K
fabricated attestations < the ε-slice value — solvable analytically once
emission ATN has a price.

**Sim-testable prediction.** Fix the honest economy; inject K ε-only
households cross-attesting a greenlit ring, sweep K and epochs. Measure the
sybil share of the fixed emission pool and its epoch-trajectory.
Prediction: sybil pool-share grows ~linearly in K (not bounded by a single
ε) and *accelerates* once dust households cross above ε from their own
accrued reputation. The policy question the sim answers: at what K does
ε·K exceed a tolerable fraction of BASE_EMISSION, i.e. how low must ε be
(or must vetting/binding cost rise) to keep the faucet closed.

---

## Attack 7 (novel) — Greenlight-then-mutate is blocked; note why it fails

Worth recording because it's the obvious next idea and the code stops it.
Greenlight is keyed to the manifest **digest** (lines 317, 458-460), and
the digest content-addresses the manifest incl. `code_digest`. Changing
the code changes the digest → a NEW candidate that must be re-greenlit. So
"get benign code greenlit, swap in malware" is impossible without a fresh
greenlight. The residual is Attack 3 (greenlight the malware directly,
cheaply) — the mutation path adds nothing. No sim needed; this is a
static-analysis conclusion.

---

## Summary ranking (severity × ease, as-built)

| # | Attack | Beats its defense? | Regime where it wins |
|---|--------|--------------------|----------------------|
| 3 | Vetting collusion (unbound distinct fleets, vets not voice-weighted) | **Yes, fully** | ALL regimes |
| 4 | Discovery spam (candidate stage unfiltered) | **Yes** (spam half) | ALL regimes |
| 6 | ε-floor dust minting at scale | Exploits floor by design | reputation + weights=None |
| 5 | Review-nuke new entrants | Partial (low-mass only) | new tools; all under weights=None |
| 1 | Sybil review ring (mint) | Partial / self-bootstrapping | weak in reputation, total under weights=None |
| 2 | Self-composition depth farm | **No — closed by damp-then-split** | none |
| 7 | Greenlight-then-mutate | **No — digest-bound** | none |

**Two structural fixes the sims should price:** (1) make vet weight
voice-weighted and/or require an owner-bound wallet to cast a
greenlight-counting vet — this kills Attack 3 and raises the cost floor
under 6; (2) filter `probe_tools` candidates by greenlight and/or sink
(not just floor) SEO manifests whose claimed position diverges from
demonstrated coverage — this kills Attack 4's spam half. The reputation-as-
voice switch is the strongest thing in the design and should be defended by
never letting any rail (vetting, candidate retrieval) run *un*-voice-weighted.
