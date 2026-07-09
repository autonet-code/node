# Attestation: does the v3 tool economy produce a commons of increasingly capable tools?

Date: 2026-07-09. Method: game-theoretic model (`game_model.md`, code-grounded,
every payoff term cited) + attack catalog (`attacks.md`) + agent-based sims
(`sim/`) driving the REAL `federated_epoch_close` / `compute_tool_mint` /
position-drift / emission-pool code over synthetic populations (120–200
epochs, fixed seeds, conservation asserted). Nothing here modifies production
code. All numbers in `sim/results/`.

---

## Verdict

**The core hypothesis holds in the honest regime and the service→tool
commoditization loop works exactly as designed. The architecture does NOT yet
hold at adversarial scale: three rails (vetting, the ε voice floor, position
drift) admit cheap attacks that the sims confirm quantitatively, and one piece
of ratified doctrine (fee recycling) is unwired on the default service rail.**

The attestation is therefore conditional: with the four fixes below, the
"commons of increasingly capable tools" claim is supported by both the formal
model and the simulations. Without them, C10 (monotone capability
accumulation) fails at scale via three absorbing bad states.

---

## 1. What holds (attested)

**A1 — Quality sorts earnings and ranking (C2/C4).** Honest baseline:
true-quality ↔ cumulative-mint correlation **0.92**; quality ↔ discovery-rank
**0.89**; the worst tool sinks to last place. The
review → drift → rank → usage → mint loop converges as designed. Burial works
as the only GC.

**A2 — Service→tool commoditization (C1, the user's headline hypothesis).**
Modeled as: service value = moat rent + expressible fraction φ; cloner pays a
mechanism-rediscovery cost (services are black boxes); users switch when their
local execution cost beats the price. Sim results:
- Cloning paid in **every** swept cell (φ ∈ {0.3, 0.7, 1.0} × rediscovery
  cost ∈ {0, 5, 20, 50 ATN}).
- Surviving service revenue converges **exactly** to the moat rent (1−φ):
  0.70 / 0.30 / 0.00.
- Equilibrium: prices compress toward moat rent; everything expressible gets
  tooled into the commons. Second-order accelerant: as the commons grows,
  caller-side cost falls (capability ratchet), expanding the set of users for
  whom self-hosting beats paying — absorption speeds up even at fixed φ.

**A3 — The structural defenses that work.** Self-composition depth farming is
exactly neutral (damp-then-split, verified no post-split log1p path);
greenlight-then-mutate is blocked by content addressing; mechanical
(non-attested) receipts mint zero; zero-usage spam tools do NOT dilute honest
earners (they contribute 0 to the pool denominator); voice cannot be bought
(soulbound reputation, not ATN — a whale gains no review power).

**A4 — Zero-sum pool clarity.** Production mint is share-of-pool
(`apply_emission_pool` scales Σmint to pool = 100 + recycled fees each
epoch). Consequences now explicit: publishing competes for shares; a
service's fees part-fund its own replacement, but fractionally (the pool is
split among ALL authors); per-clone incentive = pool/U_total decays as the
commons grows — absorption is self-limiting UNLESS the pool grows with
service volume (see G1).

---

## 2. What breaks (gaps, ranked by severity × cost-to-attacker)

**G-VET — Vetting is free to game and toothless. [HIGHEST]**
Vet weight is `1/(1+busts)` and NOT voice-weighted; unbound sybils each count
as a distinct fleet. Two free identities clear `VET_QUORUM=2.0` and greenlight
arbitrary pinned code. The royalty-as-stake deterrent has no teeth: the
CON-bust clawback is dormant in v3, so a colluding vetter risks nothing.
Vetting is the ONLY defense against covert harm invisible to satisfied users —
it is currently a single point of failure that costs the attacker zero.
*Fix candidates:* (a) voice-weight vet contributions (rep-weighted quorum),
(b) require owner-bound vetters, (c) hold vet royalties in escrow until the
CON rail reactivates. (a)+(b) together close the free-sybil path.

**G-EPS — The ε floor is a pool faucet, unbounded in identity count.**
Each unfunded household gets voice ε=0.05; per-identity bounded, but K
identities cross-attesting a (sybil-greenlit) ring skim linearly: sim shows
sybil pool share **0.05 → 0.67 for K = 5 → 200**. The dust also mints real
reputation, compounding. Related surprise: honest users who consume but never
author hold ε forever too — the defensible honest voice mass is only the
authoring subset, so the race is worse than "sybils vs whole economy".
*Fix candidates:* an **aggregate ε budget** — cap the total ε-floored weight
mass per epoch at a fixed fraction of voice supply, so K dust identities split
a bounded slice (kills the unbounded-K scaling while preserving newcomer
bootstrap); and/or restrict ε to owner-bound households. Parameter-level (ε
value, budget fraction) = user decision.

**G-DRIFT — Position drift is far cheaper to pump than mint.**
Sybil pump sim: mint capture ratio grows with ring size (1.0 → **21× at
K=100**) — the per-identity ε bound holds but the aggregate does not, AND the
pumped tool's drifted head hit **+0.97** vs +0.29 for an equal-quality honest
control, immediately (rank-cross at epoch 0). Why: the drift denominator's
only inertia is the author prior mass 1.0, and coordinated +1.0 reviews beat
honest noisy reviews even at ε weight. The rank channel then steers HONEST
usage to the pumped tool — the economics protect mint far better than they
protect discovery. Mirror image: review-nuking a young tool works at J≥10
nukers (rank ratio 0.47 at J=30); tools with accumulated mass are safe.
*Fix candidates:* raise the author prior mass (more inertia at birth), drop
the ε floor for DRIFT specifically (drift weight = earned rep only — reviews
from zero-rep households mint usage credit but move no position), and the
aggregate ε budget from G-EPS also caps this.

**G-SPAM — Discovery candidacy is unfiltered.**
The `probe_tools` candidate stage is pure cosine over k×3 with no greenlight
or rating gate; unreviewed SEO manifests (rating 0 → lift 1.0, coverage blend
only lifts) can crowd honest tools out of top-k before the review loop can
start. Burial cannot grip an un-reviewed manifest. *Fix:* filter candidacy to
greenlit tools (one-line predicate; composes with G-VET fix).

**G1 — Fee recycling is bypassed on the default service rail.**
The 2.5% burn/recycle fires only on `Substrate.payForService`; the ratified
default settlement (`ServiceMarket.PaymentChannel`) pays providers via raw
`safeTransfer` — no fee, no burn, recycled = 0, pool pinned at base 100. The
"services finance the commons that replaces them" doctrine has no wiring on
the canonical path, which turns the self-limiting absorption of A4 from
"counterweighted" into "unmitigated". Sim confirms recycling is directionally
correct but second-order at current parameters (clone cum mint 3621 vs 3608)
— wiring it matters more as service volume grows. *Fix:* take the fee at
channel settlement (or route channel payouts through payForService).
Note: channels are any-ERC20; the fee/burn mechanic is ATN-denominated —
design decision needed for non-ATN channels.

---

## 3. Doc/code divergences to clean up (no behavior change)

- **D1**: `docs/tool_substrate.md` voice addendum + `federated_reconcile.py`
  docstring still say voice = ATN balance; code (correctly, per the ratified
  ATN=money/rep=voice decision) uses reputation. Update both.
- **D2**: `federated_reconcile_epoch` still defaults `apply_gate=True` while
  `federated_epoch_close` defaults False — direct callers silently reactivate
  the dormant gate. Align the inner default.
- **D3**: `tool_usage.py` docstring still describes v2 standing×usage mint.
- **D5**: `tool_substrate.md` describes vet royalty as slashable; it is not,
  in the launch config (dormant bust). State the launch-config truth.

---

## 4. Conditional attestation statement

Given the mechanism as built (usage-only pooled mint, reputation voice,
review-drifted discovery, vet-gated entry, no pruning), and given honest-play
sims + formal break-even analysis:

1. Market-validated expressible service capability WILL migrate into the free
   tool commons, and service prices WILL compress toward moat rent —
   **attested** (A2), with the caveat that the incentive decays with commons
   size unless fee recycling is wired on the channel rail (G1).
2. Within the commons, genuinely better tools WILL out-earn and out-rank
   worse ones — **attested** (A1) in the honest regime.
3. The commons is NOT yet safe against cheap coordinated identities: vetting
   collusion (G-VET), ε-faucet drain (G-EPS), and drift pumping/nuking
   (G-DRIFT) are all confirmed profitable at launch parameters, and G-SPAM
   lets junk pre-empt the quality signal entirely. Monotone capability
   accumulation (C10) currently FAILS at adversarial scale; the sims predict
   it holds with the G-VET + G-SPAM + G-EPS fixes applied (G-DRIFT largely
   falls out of the ε-budget + rep-weighted-drift pieces).

All fix parameters (ε value/budget, vet weighting scheme, prior mass, channel
fee) are economic-model changes → user decisions per scope rules. This
document proposes; it does not rule.

## 5. Artifacts

- `attacks.md` — 7-attack catalog with line cites and payoff sketches.
- `game_model.md` — formal model: players, exact payoff transcription,
  claims C1–C10, equilibrium sketches, parameter table, divergences D1–D5, G1.
- `sim/` — harness (real close code), 5 scenarios, `results/*.json` +
  `results/summary.md`. Real vs stubbed documented in `sim/README.md`.
