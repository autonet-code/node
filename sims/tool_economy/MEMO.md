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

---

# Vetting-knob sweep — quorum, royalty, slash curve (2026-07-05)

**For:** project owner (economics blessing — the PROVISIONAL constants in
`nodes/common/federated_reconcile.py`: `VET_QUORUM`, `VET_ROYALTY_SHARE`,
`VET_ROYALTY_EPOCHS`, and whether `1/(1+busts)` is a sufficient slash)
**From:** `sims/tool_economy/vet_sweep.py` — drives the REAL
`compute_tool_mint` epoch-by-epoch with fabricated canonical events and
the carried `vetting`/`registrations` state; no formula reimplemented.
**Reproduce:** `python sims/tool_economy/vet_sweep.py --seed 1234`
(writes `out/vet_sweep.json` + `out/vet_quorum.png`,
`out/vet_royalty_perverse.png`, `out/vet_bust_deterrence.png`,
`out/vet_collusion.png`; ~4 s).

## Q1 — Quorum N: honest friction vs sybil cost (`out/vet_quorum.png`)

Population: 12 honest tools, 6 honest validators each vetting a random
not-yet-greenlit tool w.p. 0.35/epoch; collusion families = author + 3
sock vetters, each sock missed by the owner map at rate `evasion`
(same modeling move as the lineage_recall sweep above). A caught sock
is excluded wholesale by the real merge pass; an evading sock is its
own "distinct fleet" at weight 1.

| N | epochs to 80% honest greenlit | slip @ evasion 0.3 | slip @ evasion 1.0 |
|---|---|---|---|
| 1 | 1.0–1.2 | 0.70 | 1.00 |
| 2 | ~2.0 | 0.10 | 1.00 |
| 3 | ~2.8 | 0.05 | 1.00 |
| 5 | 5.0–5.6 | 0.00 | 0.00 |

The closed form the sim confirms: **a family slips iff it controls ≥ N
fully-unlinked identities** — quorum is exactly a price in unlinked
sybils. N=1 is a turnstile (one evader suffices, 70% slip at modest
evasion). N=2→3 buys a 2× cut in moderate-evasion slip (all three socks
must evade) for ~0.8 epoch of extra honest latency. N=5 stops 3-sock
families outright but doubles honest time-to-greenlight and, with a
small validator pool, risks starving the candidate pipeline.

## Q2 — Royalty share × K (`out/vet_royalty_perverse.png`)

Median-usage tool (5 attesters/epoch), quorum 2. Closed forms verified
through the real close: author retains `1 − share/2` of the tool's
first-2K mint; each of the 2 validators earns `share × K × mint/2` per
vet. Dilution is mild everywhere (author keeps 85–97.5%).

The binding constraint is NOT dilution — it's the **vet factory**: a
validator rubber-stamping a stream of mediocre tools (standing 1,
median usage) earns steady-state `K × share × mint_med / n_validators`
per epoch. R* = mediocre vets/epoch needed to out-earn authoring one
good tool (standing 4, same usage) = `n_val × 4 / (K × share)`:

| | K=4 | K=8 | K=16 |
|---|---|---|---|
| share 0.05 | 40 | 20 | 10 |
| share 0.10 | 20 | **10** | 5 |
| share 0.20 | 10 | 5 | 2.5 |
| share 0.30 | 6.7 | 3.3 | **1.67** |

At share 0.3 × K=16, vetting **two** mediocre tools per epoch beats
authoring a good one — a real perverse incentive. Keeping
`K × share ≤ ~0.8` holds R* ≥ 10, i.e. a factory must find and vet ten
plausible-but-mediocre tools every epoch, each one a bust risk.

## Q3 — Is `1/(1+busts)` a deterrent? Only when the vet is pivotal (`out/vet_bust_deterrence.png`)

Rubber-stamp validator (vets everything, 2 tools/epoch, dirty tools
bust 2 epochs post-greenlight) vs careful validator (perfect filter,
half the volume), 50 epochs, current share/K:

| p_bust | pivotal regime (1 partner) | redundant regime (2 partners) |
|---|---|---|
| 0.05 | rubber 25.6 < careful 33.5 | **rubber 42.0 > careful 22.3** |
| 0.20 | rubber 4.1 < careful 32.4 | **rubber 36.4 > careful 22.0** |
| 0.50 | rubber 2.0 < careful 24.9 | **rubber 26.7 > careful 16.0** |

**LOUD FLAG: rubber-stamping wins the redundant regime at every bust
rate — even at p_bust=0.5 with 46 busts and vet weight 0.02.** The
mechanism: `1/(1+busts)` discounts a validator's *greenlight power*
only. Royalty is split **equally** among frozen validators, so once a
tool greenlights without needing your weight, your bust history is
economically irrelevant; volume dominates. Deterrence only appears in
the pivotal regime, where one bust (weight 0.5) makes quorum 2
unreachable with a single partner and the pipeline dies. A young
network IS mostly pivotal (few validators), so the current curve is
survivable at launch — but it decays into a rubber-stamp subsidy
exactly as the validator pool deepens. Cheapest fix, sim-verified in
direction: **weight each validator's royalty slice by their current
vet weight** (busts then cut income everywhere, not just where
pivotal); the freed remainder stays with the author (conservation
holds).

## Q4 — Collusion economics: the gate is an on/off switch, not a new leak (`out/vet_collusion.png`)

Author + 3 socks (evading the owner map at rate `evasion`) vet AND
wash-pump their own tool (5 attestations/sock/epoch, 16 epochs):

| N | evasion | greenlight rate | family mint | damper leak (no gate) | sock royalty |
|---|---|---|---|---|---|
| 2 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 |
| 2 | 0.3 | 0.375 | 25.1 | 28.7 | 1.25 |
| 2 | 1.0 | 1.00 | 86.0 | 86.0 | 4.3 |
| 3 | 0.3 | 0.00 | 0.0 | 17.9 | 0.0 |
| 3 | 1.0 | 1.00 | 86.0 | 86.0 | 4.3 |

Three findings, all clean:
1. **The royalty leaks nothing new.** Family total is bit-identical
   with share 0.1 vs 0.0 (test-asserted): the sock "royalty" is an
   intra-family transfer taken from the author's own mint. The damper's
   exclusions already decided what the family can extract.
2. **Vetting-collusion's marginal profit = the damper leak × the
   greenlight bit.** Un-greenlit candidates mint zero, so collusion
   buys access to exactly what the usage damper already leaks at that
   evasion level — never more. Gate and damper multiply; they don't
   overlap.
3. **Quorum is the whole sybil price.** At N=3, a 3-sock family needs
   all three unlinked (2.7% at evasion 0.3 — observed 0). The wash
   family's real cost is N unlinked identities that each risk the
   bust ladder.

## RECOMMENDATION (PROVISIONAL — for user blessing, not shipped)

These are sim-derived proposals for the constants currently marked
PROVISIONAL in `nodes/common/federated_reconcile.py`. Economic
parameters are the owner's call per CLAUDE.md; nothing below is
changed in code.

- **VET_QUORUM = 3** (currently 2.0): cuts moderate-evasion sybil slip
  ~2× vs N=2 (slip only when every sock evades the owner map) for
  under one epoch of extra honest latency; N=5's doubling of
  time-to-greenlight isn't worth it while the validator pool is small.
- **VET_ROYALTY_EPOCHS = 8** (unchanged): with share 0.1 it keeps the
  vet-factory threshold at R* = 10 mediocre vets/epoch — safely
  unprofitable — while paying a validator ~0.4 good-tool-epochs of
  mint per successful vet. Guardrail to keep on redeploys:
  `K × share ≤ 0.8`.
- **VET_ROYALTY_SHARE = 0.1** (unchanged): author retains 95% of
  first-2K mint; doubling the share halves the factory threshold
  (share 0.2×K=16 → R*=2.5, entering perverse territory).
- **Slash curve: `1/(1+busts)` is NOT sufficient on its own.** It
  deters only pivotal vets; in a deep validator pool a rubber-stamper
  out-earns a careful vetter at every bust rate (up to 2.6× at
  p_bust=0.05). Recommend **multiplying each validator's royalty
  slice by their current vet weight** (remainder stays with the
  author) so busts price income everywhere — or, blunter, hard vet
  ineligibility at ≥2 busts. Fine at launch as-is (young network ≈
  pivotal regime), but fix before the validator pool deepens.
