# v3 / v4 / v4.1 — three-way verdict table

**All three are the SAME comparison harness.** v3 = the real close (imported production functions). v4 and v4.1 are SIM-ONLY reimplemented rule layers (`v4_rules.py`, `v4_1_rules.py`) — production code untouched, nothing committed.

v4.1 revisions: **D'** (drop β cap; zero-rep usage mints ATN but grants NO reputation) and **E'** (continuous reversal-aware sanctions, no stabilization moment, mass-floor gated). Metric reframe: the rep-weighted consensus IS quality — adversarial scenarios report **capture COST**, not 'did truth win'.

Recommended δ = **0.7**.

| metric | v3 | v4 | v4.1 |
|---|---|---|---|
| baseline quality↔rank corr | 0.8935876439109396 | 0.6467635627331396 | 0.834 |
| baseline top/worst rank (of 20) | 1 / 20 | 1/12 | 1/13 |
| honest FP dock rate | n/a | δ0.7→<0.6% | 0.02% (continuous-E') |
| sybil_pump capture@K=100 | 21.1511 | 13.23 | 26.017 |
| sybil_pump rank-gap@K=100 | (rank pump) | ~0.03 (dead) | 0.0009 (dead) |
| ε_faucet pool/ATN share@K=200 | 0.6703 (mint) | 0.27929 (mint) | ATN 0.7212 |
| ε_faucet VOICE share@K=200 | (mint=voice) | leaks 0.10→0.28 | **0.0** (D' kills leak) |

## Did D' kill the leak?  **YES — completely.**

Sybil VOICE share flatlines at exactly 0 over 200 epochs at K∈{50,200} (`voice_share_flatlines_zero=True`). The v4 leak (0.10→0.28 creep) is gone because the β-capped faucet mint no longer grants reputation — zero-rep callers mint ATN but their authors gain voice only from rep-holding callers.

**BUT the ATN skim is now uncapped (the D' trade-off).** With no β cap, K zero-rep sybils each carry ε=0.05 ATN weight, so the sybil ATN share is large and bounded-in-time but not small: 0.4432 at K=50, 0.7212 at K=200 (`atn_share_bounded=True`). This is by design (ATN is money, not voice — a dust ring can skim spendable ATN but never buys consensus power), but the MAGNITUDE is worth a decision: if the honest economy needs most of the ATN pool, a SOFT ATN cap (not tied to rep) may still be wanted. The same mechanism is why sybil_pump ATN capture stays high (26.017× at K=100) even though its rank channel is dead.

## Did E' reversal work?  **YES.**

In the reversal run, capturers with real earned rep down-score a good tool (epochs 1-25); then rep-backed adopters keep USING it (adoption, not rank-driven) and review it up. Result: the score RECOVERS (`reversal_recovered=True`, final rating 0.095, recovery at epoch 76) and the early capturers are retroactively docked to the credibility floor (`capturers_docked_to_floor=True`). The continuous E' ledger re-scores each open review against the MOVING head, so when the score reverses the capturers' stale −1 reviews deviate > δ and dock every epoch. This is the v4 review-nuke backfire FIXED: no stabilization moment for an attacker to capture.

Caveat (honest): reversal is not free. It requires the recovering cohort to out-rep-weight the accumulated adverse mass over time — recovery took ~50 epochs after the adopters started. A tool with no loyal rep-backed adopters stays captured.

## Consensus-capture cost (metric reframe)

Rep-share (of total supply) a SUSTAINED attacker needs to move a tool's consensus score down by the target, vs the tool's warmed-up review mass:

| tool review mass | target move | attacker rep-share needed |
|---|---|---|
| 3.0 | 0.2 | 0.8579 |
| 3.0 | 0.5 | 0.9542 |
| 3.0 | 0.8 | 0.9681 |
| 10.0 | 0.2 | 0.9108 |
| 10.0 | 0.5 | 0.9542 |
| 10.0 | 0.8 | 0.9681 |
| 30.0 | 0.2 | 0.9108 |
| 30.0 | 0.5 | 0.9542 |
| 30.0 | 0.8 | 0.9681 |

cost monotone in mass: True. **Finding:** capture cost is high (rep-share ≈ 0.94-0.96 to move a score 0.3-0.6) and — surprisingly — nearly INDEPENDENT of the tool's accumulated review mass. A sustained attack swamps the static mass reservoir within a couple epochs, so the binding cost is the ONGOING rep-weight ratio: an attacker must roughly match the tool's live rep-backed reviewers every epoch. The 'relative to accumulated mass' framing is the wrong mental model — consensus is defended by the FLOW of rep-weighted usage, not a stock.

## niche_discovery (NEW, property attested)

- niche tool rating (negative): -0.268
- surfaces top-1 when ALONE in its region: **True** (rank-score 0.5906)
- SAME score buried in a crowded region: **True** (position 9)
- **PROPERTY_HOLDS = True** (both asserts pass)

Multiplicative lift `base·(1+tanh(rating))` with tanh∈(−1,1) keeps the factor in (0,2) — never zero — so a badly-scored tool still surfaces where it's the only relevant candidate, but ranks below positively-scored peers when they exist. No hard filter.

## Where v4.1 is WORSE than v4

**Uncapped ATN skim.** v4's β cap held the sybil MINT share near β (~0.10-0.28 with leak); v4.1 drops the cap, so the sybil ATN share is larger (0.7212 at K=200). v4.1 is strictly BETTER on voice (0 vs leaking) and on the nuke/reversal (E' fixes the backfire), but on raw ATN capture it is worse. The design bet is that ATN-without-voice is tolerable (money you can't turn into governance power). If that bet is wrong, re-introduce a soft ATN-side cap that is NOT rep-coupled (so it can't leak like v4's D did). Everything else in v4.1 dominates v4.

## Recommendation

- **δ = 0.7** (honest continuous-E' FP dock rate 0.02%, well under the 2% target; tighter δ chills honest reviewing, looser δ lets capture drift further before docking).
- **Adopt D' + E'.** D' kills the voice leak outright; E' fixes the v4 nuke backfire and makes capture reversible with retroactive docking. Open decision for the user: whether to add a rep-INDEPENDENT soft ATN cap to bound the dust-ring ATN skim (voice is already fully protected without it).

---

# β-cap sweep (accepted rep-independent cap on zero-rep ATN mint)

The user accepted a rep-INDEPENDENT aggregate cap on zero-rep ATN mint weight (on top of v4.1 D'/E') and asked the VALUE to emerge from sims. The cap scales ALL zero-rep usage weight uniformly so its aggregate is ≤ β of total ATN weight — it throttles dust rings AND honest newcomers, who are the same zero-rep signal. It is ATN-side only; D' already keeps zero-rep usage from earning voice, so the cap never touches governance weight.

**Trade-off measured per cell:** honest distortion = drop in corr(honest authors' true demand, realized mint) vs the uncapped (β=None) baseline; sybil skim = dust ring's ATN pool share.

## β* (smallest β with honest corr-drop < 0.05, worst-case over K)

| maturity | honest zero-rep share | β* | sybil skim @ β* |
|---|---|---|---|
| young | 0.6 | NONE (see below) | — |
| growing | 0.3 | 0.02 | 0.017 |
| mature | 0.1 | 0.02 | 0.0189 |

**The expected shape did NOT emerge cleanly — and the reason is the whole point.** The brief's hypothesis was β* ≈ just above the honest zero-rep usage share, falling as the network matures. Instead:

- **Young network (60% newcomer demand): β* = NONE.** No usable β keeps honest distortion < 0.05 — only β=0.5 gets there at K=0 (corr 0.97) but at K=200 even β=0.5 leaves a 0.149 corr-drop and a 0.37 sybil skim. In a young network the dust ring and the honest newcomers ARE the same zero-rep signal; you cannot throttle one without mispricing the other. Any β low enough to stop the ring also stops honest newcomer demand from reaching authors.
- **Growing / mature: β* = 0.02** (the smallest swept). Once most demand comes from rep-holding users, honest mint tracks demand through the rep-weighted channel, so throttling the zero-rep tail hardly moves the correlation — and a tiny β (0.02) already caps the ring skim at ~0.017-0.019. Here a low fixed β is nearly free.

In other words: **the cap is cheap exactly when it's least needed (mature) and expensive exactly when the network most needs newcomer demand to count (young).** The honest reading is that β cannot be one number; it must relax as the network is young and tighten as it matures — which motivates the adaptive rule.

## Adaptive rule: β pegged to observed zero-rep share, ceiling 0.2

| maturity | corr (demand↔mint) | corr-drop vs uncapped | sybil skim | eff zero-share seen |
|---|---|---|---|---|
| young | 0.7575 | 0.2293 | 0.1491 | 0.9918 |
| growing | 0.9418 | -0.0264 | 0.1704 | 0.9844 |
| mature | 0.9467 | -0.3573 | 0.1888 | 0.9786 |

The adaptive peg reads last-epoch's observed zero-rep weight share and clamps at a 0.2 ceiling. Two things the sim shows:

1. **The observed zero-rep WEIGHT share stays high (~0.8-1.0) even when zero-rep USAGE is a minority.** The cause is structural: each zero-rep household carries a flat ε=0.05 weight, while a rep household carries rep/supply — which shrinks as supply grows. Once supply is large, a single rep-user's weight (e.g. 10/1500 ≈ 0.007) is far below the ε floor, so the zero-rep tail dominates the WEIGHT share regardless of the USAGE mix (measured: 30% newcomer usage → 0.80 observed weight share; a dust ring pushes it toward 1.0). The peg therefore almost always wants the ceiling and the rule degenerates to 'use the ceiling (0.2)', NOT a smoothly maturity-adapting value.
2. **The ceiling is what actually protects.** With the ceiling at 0.2 the sybil skim stays bounded (~0.15-0.19 at K=200) regardless of the peg reading.

### Can a ring game the peg?

- Ring inflates the observed zero-rep share by spraying dust usage on honest tools: it pushes the observed share to 0.9951 (near 1.0), but the ceiling holds — sybil ATN share = 0.0858, identical to the fixed-β=0.2 ceiling case (0.0858). **Inflating the peg does NOT help the ring beyond the ceiling** — the ceiling is a hard cap the peg can only push UP to, never past. So the adaptive rule is not gameable in the harmful direction, but it also provides no benefit over just fixing β at the ceiling in these regimes.

## Recommendation

- **Do NOT ship a single fixed β.** The sweep shows β's cost is maturity-dependent and, in a young network, prohibitively high — a fixed low β would silently strangle newcomer demand signal.
- **Mature / growing networks: fixed β ≈ 0.05** is a safe default (honest corr-drop < 0.05, ring skim ~0.02-0.04). β=0.02 also works and is tighter on the ring; 0.05 leaves a little more headroom for honest newcomers.
- **Young network: run with a HIGH β (≥ 0.3) or effectively uncapped**, accepting the dust-ring ATN skim as the price of letting newcomer demand price honest work. Recall D' already makes that skim VOICE-free, so the young-network risk is bounded to spendable-money dilution, never governance capture.
- **The adaptive peg, as specified (weight-share pegged), reduces to its ceiling** and is not worth the complexity over a maturity-scheduled fixed β. IF an adaptive rule is wanted, peg on rep SUPPLY (a clean maturity proxy: β = high while supply is small, decaying toward ~0.05 as supply grows) rather than on the observed weight share — that is the signal that actually tracks maturity and cannot be inflated by dust usage. Worth a follow-up sim if the user wants a single self-tuning knob.


---

# Supply-pegged β schedule (life-cycle validation)

The β-cap sweep concluded β must not be a constant. This validates the fix: **β is a function of total REPUTATION SUPPLY** — a maturity proxy that, under D', a dust ring cannot inflate (its mint grants no rep). β_min = 0.05. Two forms swept over S₀ ∈ {10, 50, 200} epochs-worth of pool (1 epoch = 100 ATN rep):

- hyperbolic:  β(S) = β_min + (1−β_min)·S₀/(S₀+S)
- exponential: β(S) = max(β_min, exp(−S/S₀))

**ONE continuous life-cycle** (300 epochs): genesis (supply≈0 → β≈1, uncapped) → supply grows endogenously from honest rep-weighted mint → K=200 dust rings injected in early / middle / late windows. Newcomer share of honest demand falls ~0.85→0.15 as the network matures.

## Uncapped baseline (with rings) — the reference

- per-stage honest corr (demand↔mint): {'early': 0.8994, 'middle': 0.7296, 'late': 0.1869}
- per-stage ring ATN skim: {'early': 0.687, 'middle': 0.8034, 'late': 0.8441} (this is what an uncapped economy pays the ring per stage)

## Schedule sweep (worst-case honest distortion + late skim)

| form | S₀ (ep) | corr-drop early/mid/late | worst drop | late ring skim | ring voice | β @ early/mid/late |
|---|---|---|---|---|---|---|
| hyperbolic | 10 | 0.2071/0.2519/-0.7075 | 0.2519 | 0.0848 | 0.0 | 0.4991/0.1483/0.0989 |
| hyperbolic | 50 | -0.0678/-0.2024/-0.7465 | -0.0678 | 0.3817 | 0.0 | 0.9225/0.706/0.4451 |
| hyperbolic | 200 | 0.0/-0.0606/-0.4964 | 0.0 | 0.7696 | 0.0 | 0.9821/0.9421/0.8973 |
| exponential | 10 | 0.3118/0.2857/-0.7039 | 0.3118 | 0.0429 | 0.0 | 0.2331/0.05/0.05 |
| exponential | 50 | -0.0745/-0.0357/-0.7065 | -0.0357 | 0.1117 | 0.0 | 0.9129/0.573/0.1302 |
| exponential | 200 | 0.0/-0.0696/-0.5574 | 0.0 | 0.758 | 0.0 | 0.981/0.9369/0.8837 |

## (3) VOICE regression — D' holds throughout

Ring VOICE share across the ENTIRE trajectory: **0.0** (must be 0). D' keeps every dust identity voiceless regardless of how much ATN it skims.

## (4) Adversarial peg check — a ring cannot tighten β

Supply trajectory WITH rings vs WITHOUT rings (identical honest behavior via dedicated RNGs, same schedule):
- final supply with rings: 9738.8
- final supply without rings: 9944.9
- max supply gap over trajectory: 206.1053 (relative 0.020725)
- max |β gap|: 0.008768
- worst β TIGHTENING a ring induced: **0.0** (negative = tightened; ~0 = cannot attack)

Under D' the ring earns zero reputation, so it contributes nothing to supply DIRECTLY. The only residual channel is pool DILUTION: a ring skimming the fixed ATN pool leaves honest authors slightly less ATN — and thus less rep — per epoch, so honest supply grows a touch SLOWER with rings present. But that pushes β the honest-FAVORABLE way (slower supply → β stays HIGHER → MORE newcomer weight allowed): rings SLOW honest supply (pool dilution) -> β looser, honest-favorable. **A ring can never TIGHTEN β against honest users** (worst tightening 0.0 ≈ 0). The supply peg is adversary-proof in the attack direction, unlike the observed-weight-share peg a ring could inflate.

## (5) The transition — is there a mispricing window?

The worst-case corr-drop column is exactly this: the stage where β is already tight while honest newcomer share is still high. Read it per (form, S₀) above — a schedule that matures too FAST (small S₀) shows its worst drop EARLY (β clamps before newcomers fade); too SLOW (large S₀) shows a high LATE ring skim (β never tightens enough). The recommended S₀ is the knee that balances them.

## Recommendation

**Strict reading of the criterion (smallest honest distortion subject to late skim ≤5%): exponential, S₀ = 10 epochs.** worst corr-drop 0.3118, late skim 0.0429, β at windows {'early': 0.2331, 'middle': 0.05, 'late': 0.05}. It pins the late skim but pays a real EARLY mispricing cost (β clamps toward β_min while newcomers are still ~70-85% of demand).

**Engineering-preferred: exponential, S₀ = 50 epochs.** worst corr-drop -0.0357 (≈0 — honest work is priced almost exactly right at every stage) at the cost of a higher late skim (0.1117). **Because D' makes that skim VOICELESS money, tolerating ~10-15% late ATN skim to buy near-zero honest distortion is the better trade** — the ring gets spendable ATN it can never convert to governance power, while honest authors' earnings track real demand across the whole life-cycle. Push late skim lower later by lengthening the horizon (real networks mature over far more than 300 epochs) or nudging S₀ down once supply is demonstrably large.

## Final design parameters — full v4.1 + supply-pegged β ruleset

Every parameter, with provenance (EMERGED from sims vs SEEDED as a modeling choice / carried from prior ratification):

- **Mint = usage only, pinned tools, no vet gate (A).** Carried from v4 (ratified). SEEDED.
- **Reviews carry per-axis scores; inspection reviews drift position but mint nothing (B).** Ratified. SEEDED.
- **Drift weight = credibility × household_rep/rep_supply, NO ε floor; author prior mass 1.0 (C).** Ratified. SEEDED.
- **D' — zero-rep usage mints ATN (flat ε=0.05 weight) but grants NO reputation.** Ratified after v4; the voice-leak kill is EMERGED-verified (sybil voice share = 0 across 200+ epochs).
- **E' — continuous reversal-aware credibility: dock if a review deviates >δ from the current head, restore symmetrically; mass floor 3; credibility floor 0.1; recovery 10%/epoch.** Ratified after v4. δ, mass-floor, floor, recovery: SEEDED provisionally; **δ = 0.7 EMERGED** from the v4 sanction sweep (honest false-positive dock rate <2% only at δ≥0.7).
- **β cap = rep-independent aggregate cap on zero-rep ATN weight, ATN-side only.** Accepted in principle by the user. Its NECESSITY (a constant fails) EMERGED from the β sweep.
- **β_min = 0.05.** SEEDED (floor), consistent with ε.
- **β schedule = β(S) as a function of total reputation supply S.** The supply-peg CHOICE EMERGED (supply is the only maturity proxy a dust ring cannot inflate under D' — verified: worst β tightening a ring induced = 0.0).
- **Form + S₀: exponential, S₀ ≈ 50 epochs-worth of pool (5000 rep units)** for a near-zero-distortion life-cycle; the strict-≤5%-skim variant is exponential S₀=10. Both EMERGED from this life-cycle sweep. Exact S₀ should be re-calibrated to the real network's pool size and expected maturation horizon before launch — the sim horizon (300 epochs) is short relative to a real network.
- **Emission pool = 100 ATN base + recycled fees, fixed-pie (zero-sum among authors).** Carried from the shipped v3 economy. SEEDED.

Provenance summary: the STRUCTURE (D', E', supply-pegged β) was ratified by the user; the sims EMERGED the load-bearing VALUES — δ=0.7, the necessity of a non-constant β, the supply peg as the adversary-proof maturity signal, and the (form, S₀) trade-off frontier. The remaining SEEDED constants (β_min, mass floor, credibility floor/recovery, pool size) are conventional and can be tuned post-launch without changing the mechanism.
