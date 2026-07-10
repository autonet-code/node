# fees-only + REP-from-earnings — adversarial sim verdict (2026-07-10)

**SIM-ONLY.** Validates the 2026-07-10 ratified model ("fees-only pool +
REP-from-earnings") BEFORE any spec/build. A NEW study alongside the
v3/v4/v4.1 harness, same style: it drives the REAL v4.1 tool close
(`v4_1_rules.compute_v41_epoch` — household log1p damping, composition
fan-out, mint-weight scaling, beta cap, drift + continuous-E' credibility)
and changes ONLY the two ratified things — the pool SOURCE [FO-1] and the
REP SOURCE [FO-2] — in a new rules module (`fees_only_rules.py`).
Production code untouched; nothing committed. Machine-rendered data tables:
`summary_fees_only_tables.md` (regenerated from `results/fees_only/*.json`
every run). Fixed seeds, conservation asserts (`total_tool_mint <= fee-burn
pool`) every epoch, 120-200 epoch runs.

Model under test:
1. **[FO-1] Pool = exactly the fees burned that epoch. No base pool.**
   Fee = 2.5% of service GMV, half burns -> pool = 1.25% of GMV. Zero
   service volume -> zero pool. Pool distributed over TOOL usage shares
   (v4.1 usage math).
2. **[FO-2] REP claimed 1:1 on ATN EARNINGS** — service providers on net
   service revenue, tool authors on their fee-pool distribution. Pure
   spenders/buyers claim nothing. (REPLACES v4.1's D' rep rule.)

---

## VERDICT

**The base model (uncapped) HOLDS on every safety axis the brief probed,
by a mechanism the brief did not anticipate — and that mechanism has one
real cost worth a decision.**

The feared loop — free usage flood -> capture the fee pool -> claim REP on
the capture -> REP buys weight -> compounds — **does NOT compound (S2:
`any_compounds=False` in all 18 cells, max ring REP-share 5.1e-05 ~ 0 over
160 epochs).** It is throttled by TWO independent defenses stacked:

- **Household damping kills the single-wallet ring outright** (S2
  `one_house`: pool capture = 0.0 at every K — author==caller household is
  excluded and K co-owned sybils collapse to one log1p term).
- **REP-from-earnings is self-diluting against the many-wallet ring.**
  REP is claimed on ALL earnings, and SERVICE revenue is ~98.75% of all
  ATN earned (the tool pool is only 1.25% of GMV), so REP supply is
  overwhelmingly minted to honest service providers. The ring's only
  earnings lever is tool-pool capture (<=1.25% of GMV), so its REP-share
  dilutes to ~0 no matter how much of the tiny pool it grabs. The loop the
  brief worried about lifts EVERYONE who earns, and honest earners
  dominate the earnings stream by ~80x.

**What the ring CAN still do: a one-shot ATN skim at the funding
transition.** In the FIRST funded epoch (before honest earners have
accrued rep), K distinct-wallet sybils skim a large share of that single
pool — **0.63 at K=100 / genesis (S2), up to 0.57 if it pre-farmed a dead
period (S5, honest-idle)** — then capture collapses to ~0 by the next
epoch as honest rep-weight takes over. A bounded, one-epoch, VOICELESS ATN
skim (the captured ATN mints negligible REP). Same "ATN is money you can't
turn into voice" trade-off v4.1 already accepted, localized to the
transition.

**What BREAKS: the supply-pegged beta cap is not usable under this model
(S6).** Because tool USERS (spenders) never earn REP, honest tool usage is
systematically zero-rep, so a beta cap that throttles zero-rep weight zeroes
honest AUTHOR mint too (capped honest quality<->mint corr collapses
0.90->0.00 on all fee-growth curves). Beta can suppress the transition spike
(0.50->0.00) but only by destroying honest author pricing permanently. The
v4.1 beta-cap machinery does NOT port; it must be dropped or redesigned.

Recommendation up top: **ship the base model UNCAPPED; do NOT carry the
supply-pegged beta cap over.** Address the transition skim (if at all) with a
transition-specific measure (Options A-D), not beta. Two SEEDED assumptions
(below) the build must confirm before relying on this verdict.

---

## Per-scenario numbers

### S1 honest baseline + dead-start — CLEAN
- quality<->ATN-earnings corr **0.72**, quality<->REP corr **0.72**
  (identical by construction — REP is 1:1 on ATN earnings, so cumulative
  REP is a monotone image of cumulative earnings).
- author income = **1.24% of service GMV** — the tool pool IS 1.25% of GMV
  (fee 2.5% x 0.5 burn), so tool authors collectively earn the burn.
  Conservation identity confirmed end to end.
- **dead-start transition clean = True**: pool exactly 0 for the whole
  dead window, positive the epoch after demand arrives, no negatives, no
  NaNs, mint 0 while pool 0. Conservation asserts held every epoch, all
  scenarios.

### S2 usage-flood ring (THE loop) — SELF-LIMITING (holds)
- `any_compounds = False` across all 18 cells (3 stages x K in {5,20,100} x
  2 topologies), 160 epochs each. Ring REP-share never exceeds **5.1e-05**.
- `one_house`: pool capture **0.0** at every K (household damping).
- `k_houses`: TRANSITION-epoch pool capture scales ~linearly in K —
  **0.079 / 0.257 / 0.628 at K=5/20/100 (genesis)** — then LATE capture is
  **0.0** in every cell. A single-epoch spike, not a standing skim.
- `service_rep_only=True` (candidate fix — REP only on service revenue,
  never on tool-pool capture) drives ring REP-share to a hard **0.0**
  without changing the (already-negligible) ATN capture. Clean cheap
  belt-and-suspenders (see Options).

### S3 wash trading — STRICT LOSS holds; but voice-per-dollar is CHEAP
- Strict-loss holds: ring pays **600 ATN** in fees, reclaims **0.86 ATN**
  from the pool (**0.14%** of fees paid) -> **net ATN cost 599**. A ~99.86%
  loss, as the fee arithmetic guarantees (pay 2.5%, reclaim <= pro-rata of
  the 1.25% burn).
- **The REP side is the real story.** The wash routes 24,000 ATN of
  self-dealing GMV through the ring's own service; the ring nets it and
  claims **23,400 REP** on it. Net ATN cost 599 -> **wash voice-per-dollar
  ~ 39** vs honest service-provision voice-per-dollar **1.0**.
  Wash-trading buys REP ~39x cheaper per net-dollar than honestly earning
  it, because REP is claimed on gross-of-fee net revenue while the only
  cost is the 2.5% fee leak. **A genuine REP-side finding even though the
  ATN side is strictly loss-making.**

### S4 whale spender — ZERO REP confirmed; positive externality
- Whale REP = **0.0** exactly, supply share **0.0** — a pure spender
  authors/provides nothing, so has no earnings to claim REP on.
- Whale spending is a **positive externality**: its service spend inflates
  the fee pool and pays providers, lifting author REP **148.75 -> 892.5
  (x6)** and provider REP **11,700 -> 70,200 (x6)** vs the no-whale
  control. A whale strengthens the honest rep base it cannot join.

### S5 retroactivity — CONDITIONAL; matters only in the realistic dead-start
Transition-epoch capture, two dead-period demand regimes:

| dead-period regime | same-epoch | carried | retro worse? |
|---|---|---|---|
| honest users BUSY | 0.336 | 0.262 | No (x0.78 — carry DILUTES) |
| honest users IDLE (only ring pre-farms) | 0.355 | **0.568** | **YES (x1.60)** |

- If honest users also use tools during the dead period, carrying banks
  their usage too and DILUTES the ring.
- **In the realistic dead-start (no service volume => no honest tool
  traffic, only the ring pre-farms at zero cost), carrying dead-period
  usage into the first funded epoch is STRICTLY worse (x1.6, 0.57 vs
  0.36).** Recommendation: **count usage same-epoch-only; do not carry
  dead-period usage weight into the first funded epoch.**

### S6 beta/S0 relevance — beta IS NOT USABLE under this model
- `beta_load_bearing = False` for the LATE ring (nothing to cap — it
  already self-limits to 0); `any_single_s0_robust = False`.
- The beta cap CAN suppress the transition spike (0.50->0.00) but only by
  zeroing ALL zero-rep weight — and under REP-from-earnings honest tool
  USERS are zero-rep, so it zeroes honest AUTHOR mint too: **capped honest
  quality<->mint corr collapses 0.90 (dead_slow) / 0.99 (dead_hot /
  hot_from_genesis) -> 0.00 on every curve.**
- The supply-peg question ("does exp(-S/S0) still behave when REP supply
  grows with fee volume?") is moot: REP supply now grows at ~net-GMV per
  epoch (fast, provider-dominated), so beta decays to beta_min almost
  immediately regardless of S0 — and even at beta~1 the cap starves authors
  the moment any provider rep exists. **No S0 works; the peg does not need
  re-denominating, it needs REMOVING under this model.**

---

## EMERGED vs SEEDED classification

**EMERGED from these sims (load-bearing findings):**
- The usage-flood loop DOES NOT compound — throttled by household damping
  (one-wallet) + earnings-dilution (many-wallet). EMERGED.
- The ring's residual exposure is a ONE-SHOT transition-epoch ATN skim
  (<=0.63 pool at K=100), voiceless, collapsing next epoch. EMERGED.
- The supply-pegged beta cap is INCOMPATIBLE with REP-from-earnings (zeroes
  honest author mint because tool users are zero-rep). EMERGED — the one
  thing that BREAKS.
- Wash-trading is strictly ATN-loss-making but buys REP ~39x cheaper per
  net-dollar than honest service provision. EMERGED.
- Retroactive dead-period usage is capturable ONLY when honest demand is
  absent during the dead period. EMERGED.

**SEEDED (modeling choices the build MUST confirm):**
- **[SEEDED-1] REP claimed on GROSS-of-fee NET service revenue at 1:1.**
  Provider REP mint ~ 98.75% of GMV per epoch. This is the assumption that
  makes REP supply provider-dominated and thus dilutes the ring to 0. If
  the real design claims REP on a SMALLER base (profit, or capped/decayed)
  the dilution weakens and the ring's REP-share rises. **The single most
  important assumption to pin before trusting the S2 "does not compound"
  verdict.**
- **[SEEDED-2] Tool users (spenders) earn no REP.** True by the ratified
  rule (only earners claim) — and it is WHY the beta cap breaks (all tool
  usage is zero-rep).
- **[SEEDED] Service market exogenous** (scenarios supply payments); no
  EIP-712 channels. Fee/burn arithmetic exact; demand elasticity not
  modeled.
- **[SEEDED] REP = cumulative claimed earnings, no decay** (carried from
  the harness 1:1 dual-token stub). A decay curve would change supply
  dynamics.
- **[SEEDED] eps=0.05, delta=0.7, pool=1.25% of GMV, seed reps, warmups** —
  conventional, carried from v4.1.

---

## What HOLDS / what BREAKS

**HOLDS:** honest quality tracks ATN + REP (S1); dead-start clean +
conservation every epoch; single-wallet ring fully defeated; many-wallet
ring does not compound, REP-share ~ 0, only a bounded one-shot transition
ATN skim; whale earns zero REP and is a positive externality; wash-trading
strictly ATN-loss-making.

**BREAKS / needs a decision:**
1. **Supply-pegged beta cap is incompatible with this model** (zeroes
   honest author mint). Drop or redesign. — HARD BREAK.
2. **One-shot transition ATN skim** (<=0.63 of one pool at K=100, worse
   with a pre-farmed dead period). Bounded, voiceless, but real. — BOUNDED
   EXPOSURE.
3. **Wash-trading buys REP ~39x cheaper than honest service** — voiceless
   ATN loss, but cheap voice. — REP-INTEGRITY EXPOSURE.
4. **[SEEDED-1] risk**: the "does not compound" verdict rests on providers
   minting REP on ~98.75% of GMV. If the real REP base is smaller, re-run
   S2 with that base first. — ASSUMPTION RISK.

---

## Recommendation (flags anything that should block the build)

**Ship the base fees-only + REP-from-earnings model UNCAPPED.** Sound on
the core attack surface: the usage-flood loop does not compound, and voice
is protected by earnings-dilution + household damping without any beta cap.

**BLOCK-worthy before build:**
- **Drop the supply-pegged beta cap for this model** (S6 — it destroys
  honest author pricing). This is a change from the v4.1 ruleset and needs
  user confirmation since beta was previously ratified.
- **Confirm [SEEDED-1]**: pin the exact REP-claim base (gross net revenue
  vs profit vs capped). The whole S2 verdict is conditional on it; if it
  changes, re-run S2.

**Non-blocking OPTIONS (parameter/mechanism choices — the user's call):**
- **Option A — same-epoch-only usage counting (recommended, S5).** Don't
  carry dead-period usage into the first funded epoch. Denies the ring its
  free pre-farm; costs honest users nothing.
- **Option B — `service_rep_only` REP (S2, S3).** Claim REP ONLY on
  service revenue, never on tool-pool capture. Drives ring REP-share to a
  hard 0, doesn't touch honest ATN pricing. Cost: tool authors who provide
  no service earn ATN but no voice — fine if "voice comes from serving
  customers" is intended, not if tool authorship should confer standing.
- **Option C — cap/decay the REP-claim base to blunt wash-trading (S3).**
  The 39x wash edge comes from claiming REP on gross-of-fee revenue 1:1.
  A larger haircut, REP decay, or a distinct-counterparty requirement (so
  self-dealt GMV doesn't count) compresses wash voice-per-dollar toward
  the honest 1.0.
- **Option D — a transition-epoch soft ATN cap (NOT beta).** If the
  one-shot transition skim (Break #2) is unacceptable, bound zero-rep ATN
  weight ONLY while rep supply is near zero (a genesis guard), not a
  standing supply-pegged cap. Unlike beta it would not starve the mature
  economy. Needs its own sim if pursued.

---

<!-- Machine-rendered data tables: summary_fees_only_tables.md, regenerated
from results/fees_only/*.json on every `python run_fees_only.py`. Numbers
above are pulled from that file; if they disagree, the JSON is ground
truth. -->
