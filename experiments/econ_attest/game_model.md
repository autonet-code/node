# Game-theoretic model of the Substrate v3 tool economy

Status: analysis / pre-registration scaffold for a sim battery. Code-grounded
against `master` as of 2026-07-09. Every payoff term cites the exact line it
is transcribed from. This document does NOT change any code; it is the formal
object the sims in `experiments/econ_attest/` will falsify.

Authoritative sources read:
- `docs/tool_substrate.md` — the ratified v3 spec (Decision 2026-07-08).
- `nodes/common/federated_reconcile.py` — `compute_tool_mint`, the close.
- `nodes/common/world_model_substrate/tool_usage.py` — usage/review/vet aggregation.
- `nodes/common/world_model_substrate/infer.py` — `_infer_artifacts` discovery ranking.
- `nodes/common/world_model_substrate/reconcile.py` — `apply_emission_pool`.
- `nodes/common/voice_state.py` — voice weights + emission pool derivation.
- `nodes/common/federated_close_driver.py` — how the close is actually called.
- `nodes/common/world_model_substrate/adapter.py` — the 6-axis CHARTER.
- `contracts/core/Substrate.sol` — `payForService` fee split.

---

## 0. The one structural fact that governs everything: share-of-pool

**The per-tool mint is a SHARE of a fixed epoch pool, not an absolute amount** —
whenever the pool is set (which it is in production; see §6). This makes the
game **zero-sum among all minting agents within an epoch.**

Chain of evidence:

1. `compute_tool_mint` computes a **raw** per-author mint,
   `mint = usage_term` (`federated_reconcile.py:487`), and returns it in
   `node_agent` (`:516-524`).
2. `federated_epoch_close` passes that map as `extra_node_agent_mint` into
   `federated_reconcile_epoch` (`:1057`). That is the ONLY earning rail —
   score-movement mint is retired (`:664-676`), so `agent_mint` is built
   solely from tool mint (`:724-737`).
3. When `emission_pool is not None`, `apply_emission_pool` runs
   (`federated_reconcile.py:810-812`), and inside it:
   ```
   raw_total = Σ agent_mint            # reconcile.py:489
   scale     = emission_pool / raw_total   # reconcile.py:494
   agent_mint[a] = agent_mint[a] * scale   # reconcile.py:495
   ```
   So each agent's payout is `raw_mint_a / Σ raw_mint · pool`. The per-digest
   display `mint` is scaled by the identical factor (`federated_reconcile.py:1067-1072`).

**Consequence (the crux of C1):** an author's absolute ATN in an epoch is
`pool · (my_raw_usage_term / total_raw_usage_term)`. The pool is fixed for the
epoch (§6). Therefore:
- Publishing the marginal tool that draws real usage **dilutes** every other
  author's share (the "pool dilution" tension — see C6).
- The competition an author faces is not "did I clear a bar" but "what fraction
  of this epoch's total damped-usage mass did I capture."
- Reputation (voice = soulbound) mints 1:1 with ATN (dual-token model,
  `Substrate.sol` `recordTrainingForEpoch`), so the SAME share formula sets
  future voice weight — a slow positive-feedback term (see C7).

If the driver ever runs with `emission_pool=None` (no chain access,
`federated_close_driver.py:167,189-191`), mint is raw/absolute and NOT zero-sum
— publishing then never dilutes anyone. The sims must run BOTH regimes; the
production regime is pooled.

---

## 1. Players and strategy spaces

The **agent** is the only economic entity on the web3 layer
(`federated_reconcile.py:1116-1128`); a "daemon" is transport. A **household**
is the collapse of all agents under one proven owner wallet (or the agent
itself when unbound) — `_household` (`federated_reconcile.py:387-388`). All
damping and exclusions happen at household granularity (voice addendum,
`docs/tool_substrate.md:97-153`).

| Player | Choice variables (strategy space) |
|---|---|
| **Tool author** | whether to publish (`register_tool(publish=true)`); the manifest embedding (topical position, tail) — but NOT the charter head (enters at ZERO, `docs/tool_substrate.md:44-46`); declared `dependencies` DAG; trust_class (only `pinned` mints, `federated_reconcile.py:451`); whether to clone a demonstrated Service into a free tool (C1). |
| **Service operator** | ask price; whether to keep a capability remote (Service, earns fees, no mint, no standing — `docs/tool_substrate.md:155-161`) or let it be absorbed. |
| **User agent (consumer + reviewer)** | which tool to call (steered by discovery ranking, §4); whether to attest at all (cognitive cost); ok/score and optional per-axis review scores `axes ∈ [-1,1]` (`tool_usage.py:113-125`); honest vs pumped reviews; which loadout/distro to run. Reviews come ONLY from the post-use `attest_tools` step — no usage, no review (§5). |
| **Validator / vetter** | which candidate tools to vet (reads pinned code); ok/fail vet (only ok counts toward greenlight, `tool_usage.py:88-96`); this stakes future royalty and vet-weight (C3). |
| **Sybil operator** | number of wallets/agents to spin up; whether to bind them to one owner or leave unbound; co-host on one wire key or spread keys; fabricate attestations/reviews/vets. |
| **Distro / adopter** | curate a loadout (composite manifest), compete on policy (retrieval, compaction, delegation); adopt foreign tools (`adopt_tool`, owner-approved). |

---

## 2. Payoff functions — transcribed from code

### 2.1 Usage aggregation (the raw signal)

`tool_usage_from_events` (`tool_usage.py:21-148`). Per manifest digest it emits,
among others:
- `attested_ok_by_caller: {caller: count}` — successful invocations that carried
  BOTH `ok` and `attested` (`tool_usage.py:99-111`). **This is the only mint
  input.** Mechanical (`attested`-absent) receipts increment `count`/`ok_count`
  only and mint nothing (`tool_usage.py:98-104`).
- `attester_senders: {caller: [sender_hex,...]}` — wire keys for the co-host dedup.
- `axis_reviews_by_caller: {caller: {axis_id: {sum, n}}}` — per-axis signed
  scores, clamped to `[-1,1]` (`tool_usage.py:118-125`), ride ONLY attested-ok
  receipts (`tool_usage.py:112-113`). Axis-less usage mints but moves no position.
- `vets_by_caller` / `vet_senders` — vets are a THIRD flavor; a `vet:true`
  receipt short-circuits before any usage count (`tool_usage.py:85-97`) so a
  vet NEVER inflates usage.

### 2.2 The mint (the payoff), step by step

Inside `compute_tool_mint`:

**(a) Household collapse + log1p damping, at composition fan-out time.**
For each `(digest, caller)`, the caller's attested-ok count is added into its
household bucket, THEN `log1p` is applied once per household
(`federated_reconcile.py:399-409`):
```
counts[house] += entry["attested_ok_by_caller"][caller]      # :401-402
damped = math.log1p(counts[house])                            # :406
bucket[target] += damped * shares[target]                     # :409
```
`log1p` is applied to the POOLED per-household count — N co-owned agents are one
voice, not N (`docs/tool_substrate.md:105-108`). This is the anti-amplification
order-of-operations: **damp first, then split** over the DAG
(`federated_reconcile.py:371-379`).

**(b) Composition fan-out** (`_composition_shares`, `:151-182`, called at `:396`).
One unit of a caller's damped attestation weight splits over the declared-dep DAG:
root keeps `COMPOSITE_ROOT_SHARE = 0.7` (`:74,170`), the remaining 0.3 splits
equally among declared deps, recursing to `COMPOSITE_MAX_DEPTH = 4` (`:75,167`).
Cyclic or unregistered deps FORFEIT their share (`:176-179`) — total ≤ 1, never
> 1. Because damping precedes splitting, self-padding is exactly neutral
(`:371-378`).

**(c) Exclusions + voice weighting** (`:471-482`):
```
for house in sorted(attested_by):
    if house == author_house: continue                        # :474-475 owner-rooted excl.
    if reg_sender in house_senders[house]: continue           # :476-477 wire co-host dedup
    w_voice = 1.0
    if voice_weights is not None:
        w_voice = round(voice_weights.get(house, VOICE_EPSILON), 9)   # :479-481
    usage_term += attested_by[house] * w_voice                # :482
```
`VOICE_EPSILON = 0.05` (`:106`) floors unknown/zero-balance households.
`voice_weights[house] = ε + household_rep / rep_supply` — LINEAR in balance so
splitting is weight-neutral (`voice_state.py:329-332`; **see divergence D1: the
denominator is REPUTATION, not ATN**).

**(d) Mint = usage_term** (`:487`). No standing multiplier, no gate
(v3 Decision, `:484-487`). Tools that aren't `pinned` are skipped (`:451`);
tools not greenlit are skipped when `vet_quorum>0` (`:460-461`).

**(e) Vetting gate + royalty split** (`:496-524`):
- A published tool is a CANDIDATE until greenlit. Greenlight = Σ over distinct
  fleets of `1/(1+busts)` weight ≥ `VET_QUORUM = 2.0` (`:350-364`, `:88`).
  Validators are FROZEN at greenlight (`:364`).
- While `royalty_left>0`, `VET_ROYALTY_SHARE = 0.1` (`:89`) of the tool's mint
  splits equally among frozen validators, taken FROM the author's share
  (conserved, `:517-522`). Window ticks once per close, minted or not
  (`:529-532`).

**(f) Adoption / loadout credit** (`:422-440`). Distinct fleets per loadout each
contribute `log1p(1)` once (volume-blind, `:436`), injected at the distro root
and fanned over its dep DAG. Modules earn through distro deps.

### 2.3 The chain: pool, fee recycling, dual mint

- `payForService` charges `SERVICE_FEE_BPS = 250` (2.5%) of `amount`
  (`Substrate.sol:1021,1107`); half to treasury (`FEE_TREASURY_BPS = 5000`,
  `:1023,1109`), the other half `burned` — supply decreases now
  (`Substrate.sol:1123-1124`), re-minted into next epoch's pool.
- `emission_pool = BASE_EMISSION_PER_EPOCH (100.0) + recycled`
  (`voice_state.py:340`; `federated_reconcile.py:122`), where `recycled` = Σ
  burned in the anchor window (`voice_state.py:276-281`). Wash-proof by
  conservation.
- Reputation (voice) mints 1:1 with ATN per training event (dual-token model,
  CLAUDE.md; `recordTrainingForEpoch`). So each epoch's share-of-pool sets BOTH
  the author's ATN income and the increment to their (soulbound) voice.

### 2.4 Discovery ranking — the feedback loop (the heart of the model)

`_infer_artifacts` (`infer.py:246-365`) scores each candidate manifest:
```
base = cosine                                                 # infer.py:303
if coverage available and manifest has coverage:
    base = max(cosine, 0.5·cosine + 0.5·min(1,density))       # infer.py:306-313
head = drifted charter head of the tool                       # infer.py:324
rating = mean(head[correctness], head[simplicity])            # infer.py:325-326
final  = base · (1 + tanh(rating))                            # infer.py:327
scored.sort(key=(-final, digest))                             # infer.py:355
```
Non-manifest artifacts keep `final = base·(1+tanh(standing))` (`:329`); with no
claim nodes that's ×1.0 (pure cosine). **`rating` is read off the DRIFTED head**,
which comes from `tool_positions` written back onto the world by
`apply_tool_positions` (`federated_reconcile.py:834-875`, called at close `:1049`).

The loop: reviews → per-axis position drift (§2.5) → higher `rating` → higher
`final` → served first by `probe_tools` → more calls → more attested usage →
more mint. **Reviews never enter the mint formula directly** (`docs/tool_substrate.md:53-56`);
they pay only by steering future usage.

### 2.5 Position drift (what reviews buy)

Per-axis mint-weighted running centroid (`federated_reconcile.py:534-614`).
Household-collapsed, log1p-damped, voice-weighted review mass — SAME exclusions
as mint (`:585-594`):
```
w      = log1p(n_house_axis) · w_voice                        # :603
add_mass[axis] += w                                            # :604
add_val[axis]  += w · (cell.sum / n)                           # :605
head[axis] = (mass·head + add_val) / (mass + add_mass)        # :608-609
mass[axis] += add_mass                                         # :610
```
Prior: `head = 0`, `mass = 1.0` per axis (`:556-557`) — "the author counts as
one damped attestation." Heavily-reviewed tools have proportional inertia (no
free drift-rate knob). Only axes-bearing attestations move position; axis-less
usage mints but leaves the head at 0.

---

## 3. Known structural facts the sims MUST respect

1. **Reviews come only from real usage.** `axes` ride attested-ok receipts
   (`tool_usage.py:112-125`), which come from the agentic loop's post-use
   `attest_tools`. No call → no attestation → no review → no drift and no mint.
2. **Axis-less usage mints but does not move position** (`:558-560` folds review
   only when `axis_reviews_by_caller` present; usage_term at `:482` needs no axes).
3. **Entry head = ZERO on all 6 charter axes** — authors never self-claim
   alignment/usefulness (`:556`, `docs/tool_substrate.md:44-46`). A tool's
   `rating` starts at 0, so `final = base·(1+tanh 0) = base` (pure cosine) at
   birth — cold-start tools rank on claimed cosine alone (see C8).
4. **No pruning.** The carry-over maps grow monotonically; low-rated tools die
   only by ranking burial (`docs/tool_substrate.md:66-69`). `positions_next` and
   `registrations_next` accumulate every digest ever seen (`:550`, `:620`).
5. **Fixed pool** (§0/§6). Zero-sum among minters when set.
6. **CHARTER is 6-root**: `life_precious, self_preservation,
   promotion_of_intelligence, evolution` (alignment axes 0-3) + `correctness,
   simplicity` (usefulness axes 4-5) (`adapter.py:58-93`). Only the two
   usefulness axes lift discovery (`infer.py:285-287`).
7. **Violator-pays gate and CON-bust are DORMANT** (`apply_gate=False` default,
   `federated_reconcile.py:888`; bust code present but no trigger, `:343-348`).
   Sims model the launch config: vetting entry + damper + voice + burial only.

---

## 4. Claims to test (falsifiable)

Each claim names the sim-measurable quantity and the decision rule.

### C1 — Service → tool commoditization (headline)
**Claim (refined 2026-07-09).** When a paid Service demonstrates demand
(revenue `R` ATN/epoch), the EXPRESSIBLE fraction of its capability gets cloned
into a free `pinned` tool whose pooled mint is competitive with the revenue
that fraction carried — so market-validated expressible capability migrates
into the commons and Service prices compress toward moat rent.

**Value decomposition.** Cloning is NOT free entry by "any author". All agents
share the same baseline (same models, same harness), and a tool is a LOCAL
artifact: its cognition must come from the CALLER's own model, and its
execution cost lands on the caller's daemon, never the author's
(`docs/tool_substrate.md:155-161` — tools run local; anything needing foreign
credentials/counterparty is a Service or an `attested` tool, which mints
nothing, `federated_reconcile.py:451`). So decompose:
`R = R_moat + R_expr`, with `R_expr = φ·R`, `φ ∈ [0,1]` — where `R_moat` is
rent on what the baseline cannot have (private data, credentials, hardware,
human labor, proprietary server-side cognition) and `R_expr` is the fraction
re-expressible as caller-side pinned code + caller-side cognition. **Only
`R_expr` is cloneable.** Three frictions bound replication:

1. **Expressibility (φ).** Moat-backed value has φ→0 by construction — the
   same line that denies Services substrate standing (remote execution
   unknowable in principle, `docs/tool_substrate.md:17-20`) is the line the
   cloner cannot cross.
2. **Mechanism opacity (rediscovery cost `K`).** A Service is a remote black
   box (behavioral trust only); the cloner must REDISCOVER the mechanism from
   observed behavior, not copy it. `K` scales with how much of `R_expr` is
   secret workflow vs obvious-once-seen.
3. **Cost shift, not price zero.** The caller's switch condition is not
   free-vs-`p` but `c ≤ p`: local execution cost `c` (the caller's own
   compute/inference — the tool externalizes cost onto the caller's daemon) vs
   service price `p`. Clone call volume ≈ service volume only over the caller
   set with `c < p`.

**Model terms.** The cloner's per-epoch ATN = `pool · (u_clone / U_total)`,
where `u_clone = Σ_house log1p(calls_house)·w_voice` is the clone's damped
usage term (`:482`) with calls drawn from the service's demonstrated demand
restricted to the `c < p` caller set, and `U_total` is the epoch's total raw
usage term across all minting authors. The Service earns `R` minus the `2.5%`
fee (`Substrate.sol:1107`), of which the burned half re-enters `pool`
(`voice_state.py:281`) — the Service part-finances the pool that pays its
replacement.

**Break-even condition (revised).** Cloning pays iff, over the horizon:
```
Σ_epochs  pool · u_clone(φ, c<p) / U_total   ≥   K + authoring/vetting cost
```
The prize is bounded by what `R_expr` demonstrated (expected `u_clone` derives
from validated demand × the `c<p` share of callers), and the payout is a POOL
SHARE, so it falls as `U_total` (tool count/activity) grows. **Equilibrium
prediction:** prices compress toward moat rent — `p·calls → R_moat` as tooled
substitutes strip `R_expr`; services retreat to what only they can do. That IS
the commons-growth mechanism, self-reinforcing via fee recycling only
fractionally (dilution point below stands). **Measure:** ratio
`clone_mint_ATN / (φ·R_net)` as a function of (a) φ, (b) rediscovery cost `K`,
(c) the `c/p` distribution across callers, (d) commons size `M`. **Falsified
if** for high-φ, low-`K`, `c ≪ p` services the clone still cannot recoup
`K` + authoring cost even at call parity in a small commons (mint can't fund
absorption in the easiest case), OR if prices do NOT compress toward `R_moat`
as clones enter.

**Knife edges to log:** (i) burned-fee recycling means Service success *raises*
`pool`, but that pool is split among ALL authors, not just the cloner — the
Service subsidizes the whole commons, only fractionally its own replacement;
quantify the leakage. (ii) `c` is not static: the capability ratchet
(`docs/tool_substrate.md:163-202`) lowers caller-side cognition cost as the
corpus grows, expanding the `c < p` set over time — absorption accelerates
even at fixed φ. (iii) **[NEW GAP — G1] fee recycling is BYPASSED on the default
service rail.** The 2.5% burn/recycle exists ONLY on `Substrate.payForService`
(`Substrate.sol:1107-1128`). But the ratified default settlement is the
`ServiceMarket.PaymentChannel`, which pays the provider by raw
`IERC20.safeTransfer` in ANY ERC20 (`ServiceMarket.sol:328`) — no fee, no burn,
no `ServiceFee` event. So for channel-settled services `recycled = 0` and the
pool stays pinned at the `BASE = 100` floor. The "Service part-finances the pool
that pays its replacement" claim (and the `docs/services_market.md:161-163` "two
grow each other" doctrine) has NO on-chain wiring on the canonical service path.
This strengthens the C6/C1 dilution problem: `pool` does not grow with service
volume, so per-clone pool share `pool/U_total` decays purely as the commons
grows, with no service-revenue counterweight. Sims must run C1 with recycling
OFF (channel default) as the primary case and ON (`payForService`) as the
optimistic case; the delta between them is the value of wiring the fee into the
channel.

### C2 — Capable tools out-earn junk
**Claim.** The loop review-drift → ranking → usage → mint converges so that,
holding topical relevance fixed, tools with higher true (correctness,simplicity)
earn strictly more over time.
**Measure.** Rank correlation between hidden true-quality parameter and realized
cumulative mint across epochs. **Rule:** Spearman ρ → significantly positive and
increasing with epoch count. **Falsified if** ρ stays ≈0 (ranking doesn't
translate quality into earnings) — which would happen if `tanh(rating)` lift is
swamped by cosine noise on a crowded topic.

### C3 — Review pumping / sybil rings don't pay
**Claim.** A ring fabricating attestations + reviews cannot profitably pump a
tool's mint or position after: household collapse + log1p (`:406`), owner-house
exclusion (`:474-475`), wire co-host dedup (`:476-477`), voice weighting to
`ε=0.05` for unfunded households (`:481`), and distinct-fleet vetting
(`:350-364`).
**Model.** A ring of `k` throwaway (unbound, unfunded) identities contributes at
most `k · log1p(1) · ε` to `usage_term` if it manages distinct wire keys and
distinct households — i.e. bounded by `ε` per fabricated household
(`docs/tool_substrate.md:116-121`). If co-hosted, the wire dedup collapses them.
If co-owned, household collapse makes them ONE `log1p(Σ)` term.
**Measure.** Pumped-tool mint share vs honest baseline as a function of ring size
`k` and ring funding. **Rule:** with unfunded rings, pumped share ≤ `ε`-bounded
and does not scale with `k`. **Falsified if** an unfunded ring of growing `k`
grows its mint share super-linearly, OR if a funded ring earns more than its own
staked capital would justify (that's the accepted `ε`-surface + capital=voice
tradeoff — measure where it breaks).

### C4 — Ranking burial suppresses junk earnings
**Claim.** With no pruning, burial is the only GC: a tool that drifts to low
(correctness,simplicity) is served later by `_infer_artifacts` (`final` lower,
`:327`), pulled less, used less, and its mint share decays toward zero.
**Measure.** For a tool whose reviews turn negative at epoch `t`, track its
retrieval rank and mint share for `t+1..t+n`. **Rule:** monotone decay of both;
mint share → below noise floor. **Falsified if** a buried tool retains nonzero
mint share indefinitely (e.g. because incumbents keep calling it out of habit —
the sim must model discovery-driven, not habitual, calling to test this fairly),
or if `tanh` saturation makes rank differences too flat to bury.

### C5 — Composition rewards reuse, not depth farming
**Claim.** Genuine reuse (many distinct callers/composites depending on a shared
tool) earns the dependency more than self-composition depth farming.
**Model.** Damp-then-split (`:371-378`, `:406-409`) makes self-padding exactly
neutral: `log1p(1)` split as `0.7 + 0.3` over your own deps equals what you'd get
undivided, and forfeiture of unregistered/cyclic deps (`:176-179`) means padding
with dead deps only loses credit. A genuinely-reused dep collects a
`0.3/len(deps)` slice from EVERY composite built on it, across distinct authors.
**Measure.** Compare cumulative mint of (a) a shared library tool with `d`
distinct dependent composites vs (b) an author who wraps their own tool in a
depth-4 self-chain. **Rule:** (a) grows with `d`; (b) ≤ the un-composed baseline.
**Falsified if** any self-composition arrangement yields > the caller's genuinely
attested credit (would contradict `:371-378`).

### C6 — Marginal-tool publication (pool dilution vs incentive)
**Claim (new, load-bearing).** Because payout is share-of-pool (§0), publishing
the marginal tool is individually rational (captures share) but dilutes
incumbents — yet the system does NOT collapse into spam, because un-used tools
draw `usage_term = 0` and mint nothing (`:487-489`), and greenlight vetting gates
entry (`:460-461`).
**Measure.** Equilibrium tool count `M*` as a function of pool size and per-tool
vetting cost. **Rule:** `M*` stabilizes where marginal author's expected pooled
share ≈ vetting+authoring cost; spam (zero-usage tools) has no effect on the pool
because raw_total ignores them. **Falsified if** publishing zero-usage tools
dilutes honest earners (it must not: they contribute 0 to `Σ raw_mint`).

### C7 — Voice positive-feedback (rich-get-richer bound)
**Claim (new).** Since reputation mints 1:1 with ATN and voice weight is LINEAR
in reputation share (`voice_state.py:331`), high earners get higher `w_voice`,
which multiplies their next-epoch usage term (`:482`) — a positive-feedback loop.
The linearity (splitting-invariance) is deliberate; the question is whether it
runs away.
**Measure.** Gini of reputation over epochs under honest play. **Rule:** Gini
converges (bounded) rather than → 1, because `w_voice` multiplies a household's
OWN damped usage — it cannot manufacture usage it didn't receive, and `ε` floors
newcomers. **Falsified if** a single early household captures a dominant,
non-decaying pool share purely from the voice multiplier without commensurate
usage.

### C8 — Cold-start vs drifted incumbents
**Claim (new).** A new correct tool enters with head=0 → `rating=0` →
`final=base` (pure cosine, `:327` with `tanh 0`), competing against incumbents
whose drifted heads give them `tanh(rating)>0` lift. Can a superior newcomer
overtake?
**Measure.** Epochs-to-overtake for a newcomer with true quality above an
incumbent, as a function of the incumbent's accumulated `mass` (inertia, `:610`).
**Rule:** overtaking is possible but slowed by incumbent mass; high-mass
incumbents that turn bad are sticky (relates to C4). **Falsified if** cold-start
is insurmountable (newcomer never gets enough initial cosine-driven calls to
accumulate the reviews it needs) — a genuine cold-start-monopoly risk to flag.

### C9 — Attested-only mint integrity
**Claim.** Mechanical receipts (exit codes) cannot mint; only cognitive
attestations do (`tool_usage.py:99-111` gate on `attested`). The cognitive cost
of attestation is the anti-wash floor price (`docs/tool_substrate.md:303-305`).
**Measure.** Attempt to mint from a flood of mechanical (non-attested) receipts.
**Rule:** `usage_term = 0`. **Falsified if** any non-attested path contributes to
mint (pure code check + sim assertion).

---

## 5. Equilibrium sketches (informal)

**Honest steady state.** With the pool fixed and zero-sum, honest authors reach a
Cournot-like split where each tool's pool share ≈ its share of total damped,
voice-weighted attested usage. Quality sorts earnings through the ranking loop
(C2); the marginal published tool earns just above its vetting+authoring cost
(C6). Services survive only on the truly-remote kernel (proprietary
data/credentials/hardware); everything replicable is absorbed at a rate set by
`pool/U_total` (C1) — strong early, self-limiting as the commons grows. Voice
concentrates toward productive households but is bounded by the fact that
`w_voice` scales a household's OWN usage and newcomers are floored at `ε` (C7).

**Knife edges (where the model predicts fragility):**
1. **The `ε`-surface (accepted).** Each fabricated unfunded household can inject
   up to `ε=0.05` of unearned weight into both mint and drift (`:481`, `:594`).
   Bounded per identity, but the aggregate over many identities is the residual
   sybil hole the design explicitly accepts (`docs/tool_substrate.md:116-121`).
   Sims should quantify aggregate `ε`-leakage vs honest supply growth.
2. **Vetting is the ONLY defense against covert harm** invisible to satisfied
   users (`docs/tool_substrate.md:85-87`). Since the CON-bust is dormant, a
   greenlit-then-malicious tool has no automated clawback in the launch config —
   the whole covert-harm story rests on `VET_QUORUM=2.0` distinct fleets reading
   code. This is a structural single-point-of-failure to surface, not a claim to
   pass/fail.
3. **Cold-start monopoly (C8).** If incumbent `mass` inertia + `tanh` lift make
   ranking too sticky, a better newcomer may never accumulate the initial usage
   to drift its head up — burial's mirror image.
4. **Pool-share vs absolute-value confusion for Services (C1).** Absorption
   incentive depends on `pool/U_total`, which shrinks as the commons succeeds —
   so the pump weakens exactly when there's most to absorb. Whether this leaves a
   durable paid frontier or a slow commodification is the empirical question.
5. **Voice runaway (C7).** Linearity is splitting-invariant by design but is also
   rich-get-richer; the bound rests entirely on `w_voice` multiplying only
   received usage. If sims let voice buy visibility (it does not in current code —
   `w_voice` is not in `_infer_artifacts`), the loop would close and run away.
   Confirm the ranking path is voice-free (it is: `infer.py` has no voice term).

---

## 6. Emission-pool normalization (precise answer)

- **Is per-tool mint share-of-pool or absolute?** **Share-of-pool in
  production.** `compute_tool_mint` produces raw `usage_term`; the close scales
  every agent's mint by `pool / Σ raw_mint` (`reconcile.py:489-495`), and scales
  each per-digest display `mint` by the same factor
  (`federated_reconcile.py:1067-1072`).
- **How the pool is divided among tools per close:** pool ×
  `(tool_raw_usage_term / Σ_all tool_raw_usage_terms)`, with the author's slice
  reduced by any live validator royalty (`:517-522`).
- **Is the game zero-sum among authors within an epoch?** **YES when
  `emission_pool` is set** (the production path — `federated_close_driver.py:230`
  feeds `self.emission_pool` from `voice_state.read_voice_state`, which always
  returns at least the `BASE_EMISSION_PER_EPOCH=100` floor,
  `voice_state.py:261,340`). **NO when `emission_pool=None`** (no-chain/local
  path, `federated_close_driver.py:167` default) — then mint is raw/absolute and
  publishing dilutes no one. Sims must label which regime they run.
- **Pool value:** `100.0 + recycled` where recycled = burned service-fee shares
  in the anchor window (`voice_state.py:340`, `Substrate.sol:1123-1124`).

---

## 7. Spec / code divergences (the point of this audit)

**D1 — Voice denominator: REPUTATION, not ATN. [MATERIAL]**
`docs/tool_substrate.md:109-115` and the module docstring
(`federated_reconcile.py:96-106`) both say voice weight = `ε + household_ATN /
total_ATN_supply`, and the addendum's whole "capital = voice" framing
(`:148-153`) is about ATN holdings. But the actual implementation reads
`reputationOfAt` / `reputationTotalSupplyAt` — **reputation supply**, not ATN
(`voice_state.py:301-306,329-332,337-342`, `"weight_source":"reputation"`).
This matches a LATER decision in MEMORY.md ("ATN=money, rep=voice", 2026-07-08,
same day, superseding balance-weighted voice), so it is intentional — but
`docs/tool_substrate.md` and the `federated_reconcile.py` docstring were NOT
updated and still describe ATN-weighted voice. **Game-theoretic consequence:**
voice cannot be bought on the open market (ATN is transferable; reputation is
soulbound and only mints via training/usage). This CLOSES a sybil/whale vector
the spec text leaves open — a whale buying ATN gains no voice. Sims must weight
by reputation. Fix: update the doc + docstring to say reputation.

**D2 — `apply_gate` default flips between call sites. [BENIGN but confusing]**
`federated_epoch_close` defaults `apply_gate=False` (v3 dormant gate,
`:888`), but the inner `federated_reconcile_epoch` still defaults
`apply_gate=True` (`:643`). Only safe because the outer function always passes
its own value through (`:1053`). Anyone calling `federated_reconcile_epoch`
directly gets the dormant gate REACTIVATED by default. Flag as a latent footgun.

**D3 — `tool_usage.py` module docstring is stale (v2). [DOC]**
Its header still says mint is "author attribution ∝ standing × usage"
(`tool_usage.py:8-11`) — the retired v2 formula. v3 mint is usage-only
(`federated_reconcile.py:484-487`). Cosmetic but misleading to a reader.

**D4 — Retrieval `standing` term is dead for tools but still computed.
[MINOR]** `_infer_artifacts` computes `standing = _standing_of(claim_nodes)`
(`infer.py:292`) and still uses it for non-manifests (`:329`), but for manifests
the branch overrides with the `rating` lift (`:327`). For a tool, `standing` is
computed and discarded — harmless, but a reader may think debate standing still
ranks tools (it does not, per v3). The spec is consistent (`docs/tool_substrate.md:315-320`);
only the dangling computation is noise.

**D5 — `docs/tool_substrate.md:8-9` claims "vetting … stands [unchanged]" but
the CON-bust that vetting's royalty-slashing depends on is DORMANT.** The bust
that claws back validator royalties requires a charter-violation signal
(`federated_reconcile.py:343-348` explicitly notes there is NO such signal in
v3). So `VET_ROYALTY` is stake-that-can't-be-slashed in the launch config: the
"forfeitable / slashing without a staking contract" property
(`docs/tool_substrate.md:490-492`) is inoperative until the CON rail
reactivates. Vetting still gates ENTRY, but its economic-alignment ("green-
lighting malware costs you money") is currently toothless. **This is the most
game-theoretically load-bearing divergence:** it means a validator faces no
downside for greenlighting a covertly-harmful-but-user-satisfying tool, exactly
the covert-harm case the spec names as vetting-only-defended (`:85-87`). Flag
prominently.

---

## 7b. C10 — Monotone capability accumulation vs absorbing bad states

**Claim (composes C1+C2).** Does the commons monotonically accumulate capability,
or are there absorbing states? Positive flywheel: honest usage (C2) → higher
honest reputation → wider ε-gap starving sybils (C3) → healthy discovery
surfaces quality → clones absorb revealed demand (C1) → more capability. Every
honest mint compounds honest voice (soulbound), monotonically raising the sybil
cost floor. But three absorbing (or stagnating) states, each mapped to an attack
in `attacks.md`:

1. **Spam-flooded discovery** (attacks.md Attack 4). Candidate retrieval is pure
   unfiltered cosine over `k*3` (`infer.py:275`) — NO greenlight or rating gate at
   the candidacy stage. SEO manifests with zero reviews rank on raw cosine
   (`rating=0 → lift=1.0`) and can crowd honest tools out of top-k before C2's
   review loop can even start (no calls → no reviews → no drift). **Genuinely
   absorbing:** cheap, self-sustaining, starves the quality signal. Burial (C4)
   sinks ranking WITHIN candidates but cannot revoke candidacy — it has nothing to
   grip on an un-reviewed spam manifest.
2. **ε-faucet pool drain** (attacks.md Attack 6). K ε-households cross-attesting a
   greenlit ring skim `≈ K·ε·log1p(m)/U_total` of the fixed 100 pool per epoch,
   ~linear in K and compounding as dust reputation accrues. Bounded per-identity
   (ε), unbounded in K. Absorbing iff K scales faster than honest reputation
   supply.
3. **Vetting-collusion malware** (attacks.md Attack 3). Two UNBOUND sybils clear
   `VET_QUORUM=2.0` at zero cost — vet weight is NOT voice-weighted
   (`:357`, `weight=1/(1+busts)=1.0` for any never-busted agent) and the bust
   deterrent is DORMANT (`:343-348`, D5). Greenlit malware mints and gets
   adoption-recommended: a capability-DECREASING absorbing state the review-only
   rail cannot exit (covert harm invisible to satisfied users).

**Verdict.** Capability accumulation is monotone ONLY if (a) discovery candidacy
is spam-resistant (currently NOT — Attack 4), (b) vetting is collusion-resistant
(currently NOT — Attack 3), and (c) the pool grows with the commons (currently
NOT on the channel rail — G1). Absent these, C1/C2 hold LOCALLY but C10's
monotonicity fails at scale. **Sim:** long horizon, honest economy + all attack
vectors, sweeping three defense toggles (candidate greenlight-filter, vet
voice-weighting, channel fee recycling); measure quality-weighted total usage vs
epoch; predict monotone growth only with all three fixes.

---

## 7c. Parameter table (every economic constant)

| Constant | Location | Value | Claims sensitive |
|---|---|---|---|
| `VOICE_EPSILON` (ε) | `federated_reconcile.py:106` | 0.05 | **C3, C7, C10(ε-faucet)** — sybil floor; ε·K aggregate drain |
| `BASE_EMISSION_PER_EPOCH` | `federated_reconcile.py:122` | 100.0 | **C1, C6, C10(stall)** — fixed pie; per-clone share = pool/U_total |
| log1p damping | `federated_reconcile.py:406` | `ln(1+n)` | **C3** — saturation caps pump ROI (log1p(1000)=6.9) |
| `COMPOSITE_ROOT_SHARE` | `federated_reconcile.py:74` | 0.7 | **C5** — damp-then-split makes self-padding neutral |
| `COMPOSITE_MAX_DEPTH` | `federated_reconcile.py:75` | 4 | C5 (depth-farm bound) |
| `VET_QUORUM` | `federated_reconcile.py:88` | 2.0 | **C10(malware), attacks Attack 3** — 2 unbound vets clear it |
| `VET_ROYALTY_SHARE` | `federated_reconcile.py:89` | 0.1 | C1 (cloner net), C3; minor |
| `VET_ROYALTY_EPOCHS` | `federated_reconcile.py:90` | 8 | C1; minor |
| `VET_BUST_THRESHOLD` | `federated_reconcile.py:91` | 0.5 | inert (bust dormant, D5) |
| vet weight formula | `federated_reconcile.py:357` | `1/(1+busts)`, NOT voice-weighted | **C10, attacks Attack 3** — the load-bearing hole |
| `tanh` discovery lift | `infer.py:327` | `1+tanh(rating)` ∈ (0.24, 1.76) | **C2, C4, C8** — quality→discovery gain, 3.2× spread |
| `rating` axes | `infer.py:285-287` | mean(correctness, simplicity) | C2, C4 |
| author prior mass | `federated_reconcile.py:557` | 1.0 per axis | **C2, C4, C8** — drift inertia floor; low → nukeable |
| `COVERAGE_CLAIMED_WEIGHT` / `_DENSITY_WEIGHT` | `infer.py:58-59` | 0.5 / 0.5 | C10 (spam) — blend, floored so it only LIFTS |
| `COVERAGE_DENSITY_K` | `infer.py:61` | 5 | minor |
| candidate breadth | `infer.py:275` | `k*3`, pure cosine, unfiltered | **C10(spam), attacks Attack 4** — no greenlight/rating gate |
| `SERVICE_FEE_BPS` | `Substrate.sol:1021` | 250 (2.5%) | C1, C10 (recycling) |
| `FEE_TREASURY_BPS` | `Substrate.sol:1023` | 5000 (half burns/recycles) | C1, C10 (recycling) |
| channel settlement | `ServiceMarket.sol:328` | raw `safeTransfer`, **NO fee** | **C1, C10 (G1)** — recycling bypassed on default rail |
| `emission_pool=None` regime | `federated_reconcile.py:478` | uniform w_voice=1.0 | ALL — non-zero-sum; every attack maximal |
| `OUTPUT_DECIMALS` | `federated_reconcile.py:67` | 10 | determinism only |

---

## 8. Sim battery mapping (what to build next)

| Claim | Minimal sim |
|---|---|
| C1 | Service with demand `R = R_moat + φ·R`; cloner pays rediscovery cost `K`, publishes free tool; callers switch iff local cost `c < p`; sweep φ, `K`, the `c/p` distribution, and `M` (commons size); measure `clone_mint/(φ·R_net)` and the service's price path (compression toward `R_moat`). Pooled regime. |
| C2 | `n` tools, hidden true-quality vector; agents call ∝ rank; run T epochs; Spearman(quality, cum_mint). |
| C3 | Honest baseline + sybil ring of `k` (unbound/unfunded, co-hosted, co-owned variants); measure pumped mint share vs `k`. |
| C4 | Tool with reviews flipping negative at `t`; track rank + mint share decay. |
| C5 | Shared-dep tool with `d` dependents vs depth-4 self-chain; compare cum_mint. |
| C6 | Free entry of marginal tools; find `M*`; verify zero-usage tools don't dilute (raw_total invariance). |
| C7 | Long horizon honest play; Gini(reputation) trajectory; confirm ranking is voice-free. |
| C8 | Superior newcomer vs high-`mass` incumbent; epochs-to-overtake vs incumbent mass. |
| C9 | Assertion: mechanical-only receipts → usage_term 0 (unit + sim). |
| C10 | Long horizon, honest economy + Attacks 3/4/6 enabled; toggle {candidate greenlight-filter, vet voice-weighting, channel fee-recycle}; measure quality-weighted total usage vs epoch; monotone only with all three. |

All sims MUST: run the pooled regime (§6), weight by reputation not ATN (D1),
keep `apply_gate=False` (launch config), respect no-pruning + entry-head-zero +
attested-only-mint (§3).

---

## 9. The 3 most sim-critical parameters

1. **`BASE_EMISSION_PER_EPOCH = 100.0` × (fee-recycling on/off).** The fixed pie
   drives C1's cloning threshold and the C6/C10 dilution stall. Its interaction
   with the (as-built BROKEN, G1) channel fee-recycling decides whether service
   absorption is monotone or self-limiting. Highest-leverage knob because
   per-clone incentive `= pool/U_total` decays as the commons succeeds unless the
   pool grows with service volume — which it currently does not on the default
   channel rail. Sweep pool ∈ {100 fixed, 100+recycled} against commons size.

2. **`VOICE_EPSILON = 0.05`.** Sets the sybil floor for C3/C7 and the ε-faucet
   aggregate drain for C10 (Attack 6). Determines whether the honest-vs-sybil
   bootstrapping race is winnable (each ε-slice mints real reputation that lifts
   the sybil above ε next epoch). Sweep to find the ε where K·ε drain stays below
   a tolerable fraction of BASE across attacker scale K.

3. **`VET_QUORUM = 2.0` with un-voice-weighted vet weight (`:357`).** Gates
   C10's malware absorbing state (Attack 3): two unbound sybils clear it at zero
   cost and the bust deterrent is dormant (D5). Sim should price the delta from
   the two candidate fixes — (a) voice-weighting vets, (b) requiring owner-bound
   vetters — since this is the single covert-harm defense in the launch config.
