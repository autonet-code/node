# Service fee rate — wash-cost vs honest-volume sweep (2026-07-25)

**SIM-ONLY.** Answers the one parameter `docs/tool_substrate.md` §Decision
(2026-07-10) item 8 leaves explicitly open: *"Fee value: OPEN, pending a
sweep of wash-cost vs honest-volume elasticity before blessing a number."*

Drives the REAL fees-only harness (`scenarios_fees_only.s3_wash_trading`,
which itself drives the real v4.1 close) with only `FEE_RATE` varied
across the governed range. `Substrate.setServiceFeeBps` bounds are
50–1000 bps; the grid spans exactly that. 120 epochs, seed 3, fixed.

Raw: `results/fee_sweep_raw.json`.

---

## VERDICT

**The wash-advantage side of the tradeoff is not empirical — it is a
closed form, `(1−f)/f`, and the sim confirms it to within 0.2%.** The
honest-volume side is NOT measurable from anything we have, because no
real service volume exists yet. So this sweep brackets the decision; it
does not settle it.

What it does settle: the spec's stated ~39× at 2.5% is correct, the
"strict ATN loss" property holds at every fee in the governed range, and
the choice of fee reduces to a single unknown scalar (honest demand
elasticity) rather than to anything about the scoring rails.

---

## 1. Wash advantage is analytic

A washer pays fee `f` on self-dealt GMV and claims REP 1:1 on the net
`(1−f)` that lands back in its own pocket. Its voice-per-net-dollar is
therefore `(1−f)/f`. An honest provider claims REP 1:1 on net revenue,
so its voice-per-dollar is exactly 1.0 by construction.

| fee (bps) | wash voice/$ | honest voice/$ | advantage | closed form `(1−f)/f` | ratio |
|---|---|---|---|---|---|
| 50 | 199.28 | 1.0 | 199.28× | 199.00 | 1.0014 |
| 100 | 99.14 | 1.0 | 99.14× | 99.00 | 1.0014 |
| 150 | 65.76 | 1.0 | 65.76× | 65.67 | 1.0014 |
| 200 | 49.07 | 1.0 | 49.07× | 49.00 | 1.0014 |
| **250 (current)** | **39.06** | **1.0** | **39.06×** | **39.00** | **1.0015** |
| 300 | 32.38 | 1.0 | 32.38× | 32.33 | 1.0014 |
| 400 | 24.04 | 1.0 | 24.04× | 24.00 | 1.0017 |
| 500 | 19.03 | 1.0 | 19.03× | 19.00 | 1.0016 |
| 650 | 14.41 | 1.0 | 14.41× | 14.38 | 1.0018 |
| 800 | 11.52 | 1.0 | 11.52× | 11.50 | 1.0017 |
| 1000 | 9.02 | 1.0 | 9.02× | 9.00 | 1.0022 |

The 0.14–0.22% residual above the closed form is exactly the pool-reclaim
fraction (`reclaim_frac_of_fee` = 0.0014–0.0015), which `(1−f)/f` omits:
the washer also claws back a pro-rata sliver of the burned half via tool
mint. It is negligible and it does not vary with `f`.

**Consequence: the sim was not needed to find this number, only to confirm
it.** Any future fee decision can read `(1−f)/f` directly. This also
confirms the spec's claim that the fee is the only lever with leverage
here — the advantage depends on `f` alone, not on any scoring parameter.

## 2. Strict loss holds everywhere

`strict_loss_holds = True` at every grid point. The ring's net ATN cost
rises linearly with the fee (119.83 at 50bps → 2396.40 at 1000bps on the
same washed GMV) and it reclaims only ~0.14% of fees paid. Washing is
never ATN-profitable; it is a pure purchase of REP at `f` per net dollar.

This is worth stating plainly because it bounds what washing IS: not a
money attack, a **voice** attack. Conservation (Σ minted == Σ burned)
is never threatened.

## 3. The honest-volume side is assumed, not measured

The pool is `GMV × f × BURN_FRACTION`, so raising `f` raises the pool
only if honest GMV does not fall faster. Under constant-elasticity
demand `GMV ∝ f^(−e)`, pool `∝ f^(1−e)`:

| elasticity | effect of raising the fee |
|---|---|
| e < 1 (inelastic) | pool GROWS and wash advantage falls — strictly better, no tradeoff |
| e = 1 (unit) | pool FLAT at any fee — raise it, wash cost falls for free |
| e > 1 (elastic) | pool SHRINKS — a real tradeoff, and the only regime where 2.5% might be right |

**We cannot measure `e`.** There is no real service volume on the network
yet, so any number here is a prior, not a finding. Stating the priors
honestly:

- Agent buyers are highly price-elastic on **commodity** calls (bulk
  embeddings, cheap inference) — they comparison-shop across providers in
  milliseconds and have no switching cost.
- Agent buyers are highly price-**inelastic** on the call that unblocks a
  task — a blocked agent with budget remaining is the least
  price-sensitive buyer that has ever existed.
- The fee is charged on both, and the mix is unknown.

That argues for `e ≈ 1` overall, which is the regime where raising the fee
is free. But this is reasoning about a market that does not exist yet, and
it should be re-run against real volume before it is treated as evidence.

## 4. What this does NOT say

- It does not say 2.5% is wrong. It says 2.5% buys voice at 39× and that
  the number is a policy choice, not a discovered constant.
- It does not price the fee's effect on **provider participation** (supply
  side). A high fee may deter providers from listing at all; that is a
  second elasticity this sweep does not touch.
- It assumes REP is claimed on ~full net revenue. Per the pinned caveat in
  `docs/tool_substrate.md` (line 435-437), if the claim base is ever
  shrunk/capped/decayed, S2 must be re-run — and so must this.
- Genesis REP seeding (item 7) multiplies the attacker's bill while the
  network is young and is a separate, un-swept lever.

## 5. Recommendation

**Do not bless a number from this sweep alone.** What it supports:

1. Record `(1−f)/f` as the wash-advantage law in the spec, replacing the
   single "~39×" datapoint. It makes every future fee discussion cheap.
2. Treat the fee as a **governed** parameter to be revisited once real GMV
   exists, not a launch constant to get right now. `setServiceFeeBps` is
   already governor-only with 50–1000 bps bounds, so the lever is in place.
3. If a launch number is forced before real volume: the analysis gives no
   reason to keep 250 bps over something higher, and every reason to
   prefer higher IF demand is inelastic. But that "if" is unmeasured, and
   picking on an unmeasured prior is exactly what the spec's "pending a
   sweep" language was trying to avoid.

The honest summary: half the requested sweep is now done and turns out to
be analytic; the other half cannot be done until the network has
customers.

---

*Method: `FEE_RATE` monkeypatched in both `fees_only_rules` and
`scenarios_fees_only` (the scenario imports it by value), 120 epochs,
seed 3, 11-point grid over the full governed range. Production code
untouched. Elasticity table is arithmetic over the constant-elasticity
form, not simulated — no demand model exists in the harness.*
