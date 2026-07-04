# Tool-economy wash-trading damper — decision memo

**For:** project owner (economics blessing, `docs/tool_substrate.md` open
knob #1)
**From:** pre-registered game-theory simulator (`sims/tool_economy/`)
**Date:** 2026-07-04
**Reproduce:** `python sims/tool_economy/simulate.py --seed 1234`
(writes `out/sweep.csv`, `out/burn_sweep.csv`, `out/recall_sweep.csv`
and six PNGs; ~22 s).

---

## Headline recommendation

**Ship the `combo` damper: count only out-of-lineage callers, then weight
by caller diversity — `usage_term = Σ_unique_out-of-lineage_caller
log1p(calls_by_that_caller)`.** It makes wash-trading ROI-negative
across every intensity while costing honest authors essentially nothing,
and — unlike either half alone — it **degrades gracefully when the
network's sybil detection is imperfect**, which it always will be.

Do **not** ship a per-receipt burn as the primary defense. Burn *can*
make wash unprofitable, but only by taxing honest usage at the same
rate; it is a blunt instrument (details below). Keep burn in the back
pocket as an optional second layer if lineage detection ever proves
weaker than modeled.

---

## The problem, precisely

V2 mint is pinned-only, decay retired:

```
mint(tool) = max(0, standing(tool)) × log1p(ok_count(tool))
```

`standing` comes from the claim graph (author's own post = +1, each
organic supporter PRO child = +1, CONs subtract). `ok_count` is the raw
count of successful `tool_used` receipts. **Pinned calls are free**, so
receipts are a free knob a wash-trader can crank. The three adversary
profiles the sim models:

| profile | standing | usage pattern |
|---|---|---|
| **honest** | high (2–6 organic supporters) | 6–20 receipts from 6–20 *distinct* out-of-lineage callers |
| **wash** | 1 (author only, no supporters) | 20–60+ receipts pumped from a pool of 2–4 sybil callers it controls |
| **SEO-stuffer** | 1 | 1–3 receipts, ~1–2 sources (matters for retrieval, not mint) |

Standing already separates honest from wash — but the mint formula lets
a wash-trader **buy back the standing gap with volume**, because
`log1p(ok_count)` keeps rising. That is exactly the leak the dampers
must close.

## What standing already buys — and where it leaks

A first, important result: **standing does a lot of the work before any
damper**. Honest tools carry standing 4–7 (author + organic supporters);
a self-pumping wash-trader carries standing 1. Since mint scales
linearly in standing, that 4–7× gap means that even under the raw
baseline, honest authors hold **~81–87%** of tool mint across the
intensity sweep. Wash-trading is not free money under baseline — it is a
*minority* leak. But it **is** a monotonic leak: as wash intensity rises
0.5 → 4.0×, wash share climbs **11% → 18%** (`out/wash_share.png`,
baseline curve), because `log1p(ok_count)` keeps rewarding more sybil
receipts with no ceiling and no cost. A determined pumper still profits.

## Damper comparison (at max intensity, `out/wash_share.png`, `out/honest_share.png`; epochs=5)

| damper | wash share @4× | honest share @4× | verdict |
|---|---|---|---|
| baseline | 0.175 | 0.807 | leaky — wash share grows with volume, unbounded |
| diversity_sqrt | 0.112 | 0.868 | mild — 2–4 sybils still register as "breadth" |
| diversity_pow | 0.095 | 0.882 | mild, same reason |
| **out_of_lineage** | **0.000** | **0.978** | kills wash *if* lineage is known perfectly |
| **combo** | **0.000** | **0.981** | kills wash, and is robust to imperfect detection |

Caller-diversity weighting *alone* (forms c) is disappointing: a
wash-trader with even 2–4 distinct sybil callers gets counted as
"diverse," so it only shaves wash share from ~0.18 to ~0.10–0.11.
Diversity is helpful but not sufficient.

The out-of-lineage filter is the heavy hitter — receipts from the tool's
author or its own sybil callers simply don't count, so a pure
self-pumping wash-trader mints **exactly zero**. Honest authors are
barely touched: their organic callers are, by construction,
out-of-lineage, so honest share *rises* to ~0.98 (the small remainder is
legitimate SEO tools with real, if thin, usage — correct behavior).

## The catch, and why `combo` wins: imperfect lineage detection

**The sim knows ground-truth lineage. The real network does not.** On
Autonet, a wash-trader's sybil callers are just other addresses; the
system can only *infer* lineage (wallet-funding clustering, correlated
call timing, shared peers, co-registration bursts). It will miss some
sybils. `recall_sweep.csv` + `out/recall_degradation.png` sweep
detection recall from 1.0 (omniscient) down to 0.0 (blind), at 2×
intensity:

| lineage recall | out_of_lineage wash share | **combo** wash share |
|---|---|---|
| 1.0 (perfect) | 0.000 | 0.000 |
| 0.9 | 0.039 | **0.013** |
| 0.75 | 0.081 | **0.051** |
| 0.5 | 0.098 | **0.068** |
| 0.25 | 0.141 | **0.099** |
| 0.0 (blind) | 0.154 (≈ baseline) | **0.103** |

When detection is perfect the two are identical (wash = 0). As it
degrades, **`out_of_lineage` drifts back toward the baseline leak, but
`combo` stays roughly a third lower at every recall level** because the
diversity weighting is a second, detection-free line of defense: a sybil
that evades the lineage filter still only counts *once* as breadth, no
matter how many receipts it pumps. Combo is dominant under uncertainty,
at a negligible honest-author cost (honest share ~0.98, indistinguishable
from out_of_lineage). This is why combo, not raw out-of-lineage, is the
pick.

**What the real system can and cannot know (stated plainly):** it cannot
read ground-truth lineage the way the sim does. It *can* build a
probabilistic lineage estimate from on-chain funding graphs, receipt
timing correlation, and peer topology, and it *can* apply diversity
weighting with zero lineage knowledge at all. Combo is the design that
uses the strong signal (lineage) when available and never fully relies
on it.

## Why not per-receipt burn (`out/burn_roi.png`, `out/burn_threshold.png`)

Burn makes wash-traders *pay* to pump: wash ROI = `mint_gained /
ATN_burned`. It works — but the sim exposes two problems (`burn_sweep.csv`):

1. **Log1p saturation sets the required burn, and it is high.** Because
   `mint ≈ standing × log1p(N)`, the mint earned *per receipt* shrinks as
   N grows. A low-volume wash-trader (0.5×) earns the *most* mint per
   receipt, so it is the hardest to make unprofitable. To push ROI < 1 at
   **every** intensity you need burn ≈ **0.2 ATN/receipt** (at burn 0.2
   the 0.5× wash-trader hits ROI 0.76). At burn 0.05, low-intensity wash
   still enjoys ROI ≈ 3.0 and only goes negative at ≥4× volume.
2. **That burn taxes honest usage at the same rate.** Burn is paid by the
   *caller*, not the author — so honest authors keep their mint, but
   honest *users* collectively pay `honest_burn` ≈ **28 ATN/epoch** at
   burn 0.2 in this population (vs ~7 at burn 0.05). Burn cannot tell a
   wash receipt from a real one; it prices *all* usage. That is friction
   on exactly the behavior the network wants to encourage.

Burn's one virtue is that it needs **no lineage knowledge** — which is
also combo's diversity half, but combo gets it without taxing honest
users. So burn earns a place only as an **optional second layer** if
lineage detection turns out weaker than the recall sweep assumes; if
used, keep the rate low (≤0.05) and treat it as a nuisance tax, not the
main defense.

## Inequality (Gini, `out/gini.png`)

Author-earnings Gini is a sanity check that the winning damper doesn't
just concentrate mint in a few hands. At intensity 1×: baseline
Gini_all ≈ 0.38, out_of_lineage ≈ 0.35, combo ≈ 0.33 — combo *lowers*
overall inequality by stripping the outsized wash earnings. Gini among
*honest* authors stays moderate (~0.14–0.17, driven by the natural
spread in supporter counts 2–6) and is essentially unchanged across
dampers — the dampers remove wash earnings without distorting the honest
distribution. No red flag.

## Surprises worth flagging to the economics owner

1. **log1p saturation is the whole story on the burn side.** Because
   mint-per-receipt *falls* with volume, the marginal receipt is worth
   least to a high-volume pumper and most to a low-volume one. Any
   *per-receipt* cost therefore bites the smallest cheaters hardest and
   the biggest cheaters least — the opposite of what you'd want. This is
   the core reason burn is a poor primary defense and why the fix belongs
   in the *usage-term definition* (combo), not in a flat per-call price.
2. **Standing does most of the work — but not all of it.** The organic
   standing gap (honest 4–7 vs wash 1) already caps wash at a *minority*
   share (~11–18%) even under the raw baseline; wash-trading is not the
   free-for-all one might fear. The residual leak is real, though: because
   `log1p(ok_count)` has no ceiling and no cost, a wash-trader can keep
   buying share with volume without limit, and the leak *grows* with
   intensity. So the damper only needs to plug a minority leak — but that
   leak is unbounded if left open, which is why plugging it matters.
3. **Diversity weighting is weaker than intuition suggests.** A handful
   of sybil identities is cheap; 2–4 of them already defeat naive
   diversity forms. Diversity only becomes strong when *stacked on* the
   out-of-lineage filter (combo), where it acts as the graceful-
   degradation floor rather than the front line.

## Recommended parameters (to be blessed)

- **Damper:** `combo` — `usage_term = Σ over unique out-of-lineage
  callers of log1p(calls_by_caller)`; `mint = max(0, standing) ×
  log1p(usage_term)`.
- **Lineage estimate:** best-effort probabilistic (funding graph +
  timing + topology). Combo does not require it to be perfect; it only
  needs to be *directionally* right, and it fails safe toward diversity-
  only when the estimate is weak.
- **Per-receipt burn:** **off** at launch. Revisit as a low-rate
  (≤0.05 ATN) second layer only if measured lineage recall in production
  falls below ~0.5.

All numbers above are reproducible from the committed CSVs; the plots in
`out/` are referenced inline by filename.
