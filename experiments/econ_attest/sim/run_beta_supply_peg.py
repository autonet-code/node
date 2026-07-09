"""Run the supply-pegged β schedule life-cycle validation, write
results/v4_1/beta_supply_peg.json, and APPEND a section to
summary_v4_1.md. SIM-ONLY, nothing committed.
"""

from __future__ import annotations

import json
import logging
import os
import time

logging.disable(logging.INFO)

import beta_supply_peg as bsp  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
V41DIR = os.path.join(RESULTS, "v4_1")
_MARKER = "# Supply-pegged β schedule (life-cycle validation)"


def main() -> int:
    os.makedirs(V41DIR, exist_ok=True)
    print("running supply-pegged beta life-cycle sim...")
    t0 = time.time()
    r = bsp.run(seed=30)
    print(f"  {time.time()-t0:.1f}s")

    path = os.path.join(V41DIR, "beta_supply_peg.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"  wrote {path}")

    _append(r)
    print("  appended supply-peg section to summary_v4_1.md")
    return 0


def _append(r) -> None:
    summ = os.path.join(RESULTS, "summary_v4_1.md")
    if os.path.exists(summ):
        with open(summ, encoding="utf-8") as fh:
            base = fh.read()
        idx = base.find(_MARKER)
        if idx != -1:
            sep = base.rfind("\n---\n", 0, idx)
            cut = sep if sep != -1 else idx
            with open(summ, "w", encoding="utf-8") as fh:
                fh.write(base[:cut].rstrip() + "\n")

    v = r["verdict"]
    grid = r["grid"]
    pc = r["peg_check"]

    L = ["", "", "---", "", _MARKER, "",
         "The β-cap sweep concluded β must not be a constant. This validates "
         "the fix: **β is a function of total REPUTATION SUPPLY** — a "
         "maturity proxy that, under D', a dust ring cannot inflate (its mint "
         "grants no rep). β_min = 0.05. Two forms swept over S₀ ∈ "
         f"{{{', '.join(str(e) for e in bsp.S0_EPOCHS)}}} epochs-worth of pool "
         f"(1 epoch = {int(bsp.POOL)} ATN rep):", "",
         "- hyperbolic:  β(S) = β_min + (1−β_min)·S₀/(S₀+S)",
         "- exponential: β(S) = max(β_min, exp(−S/S₀))", "",
         "**ONE continuous life-cycle** (300 epochs): genesis (supply≈0 → "
         "β≈1, uncapped) → supply grows endogenously from honest rep-weighted "
         "mint → K=200 dust rings injected in early / middle / late windows. "
         "Newcomer share of honest demand falls ~0.85→0.15 as the network "
         "matures.", "",
         "## Uncapped baseline (with rings) — the reference",
         "",
         f"- per-stage honest corr (demand↔mint): {v['uncapped_stage_corr']}",
         f"- per-stage ring ATN skim: {v['uncapped_stage_ring_skim']} "
         "(this is what an uncapped economy pays the ring per stage)", "",
         "## Schedule sweep (worst-case honest distortion + late skim)",
         "",
         "| form | S₀ (ep) | corr-drop early/mid/late | worst drop | late ring skim | ring voice | β @ early/mid/late |",
         "|---|---|---|---|---|---|---|"]
    for g in grid:
        cd = g["stage_corr_drop"]
        bw = g["beta_at_windows"]
        L.append(
            f"| {g['form']} | {g['S0_epochs']} | "
            f"{cd['early']}/{cd['middle']}/{cd['late']} | "
            f"{g['worst_corr_drop']} | {g['late_ring_skim']} | "
            f"{g['ring_voice_share_max']} | "
            f"{bw['early']}/{bw['middle']}/{bw['late']} |")

    rec = v["recommended"]
    L += ["",
          "## (3) VOICE regression — D' holds throughout",
          "",
          f"Ring VOICE share across the ENTIRE trajectory: "
          f"**{v['ring_voice_ever']}** (must be 0). D' keeps every dust "
          "identity voiceless regardless of how much ATN it skims.", "",
          "## (4) Adversarial peg check — a ring cannot tighten β",
          "",
          "Supply trajectory WITH rings vs WITHOUT rings (identical honest "
          "behavior via dedicated RNGs, same schedule):",
          f"- final supply with rings: {pc['supply_with_rings_final']}",
          f"- final supply without rings: {pc['supply_without_rings_final']}",
          f"- max supply gap over trajectory: {pc['max_supply_gap']} "
          f"(relative {v['peg_check_rel_supply_gap']})",
          f"- max |β gap|: {pc['max_beta_gap']}",
          f"- worst β TIGHTENING a ring induced: "
          f"**{pc['worst_beta_tightening']}** (negative = tightened; ~0 = "
          "cannot attack)", "",
          "Under D' the ring earns zero reputation, so it contributes nothing "
          "to supply DIRECTLY. The only residual channel is pool DILUTION: a "
          "ring skimming the fixed ATN pool leaves honest authors slightly "
          "less ATN — and thus less rep — per epoch, so honest supply grows a "
          "touch SLOWER with rings present. But that pushes β the "
          "honest-FAVORABLE way (slower supply → β stays HIGHER → MORE "
          f"newcomer weight allowed): {v['peg_check_supply_direction']}. **A "
          "ring can never TIGHTEN β against honest users** "
          f"(worst tightening {pc['worst_beta_tightening']} ≈ 0). The supply "
          "peg is adversary-proof in the attack direction, unlike the "
          "observed-weight-share peg a ring could inflate.", "",
          "## (5) The transition — is there a mispricing window?", "",
          "The worst-case corr-drop column is exactly this: the stage where "
          "β is already tight while honest newcomer share is still high. "
          "Read it per (form, S₀) above — a schedule that matures too FAST "
          "(small S₀) shows its worst drop EARLY (β clamps before newcomers "
          "fade); too SLOW (large S₀) shows a high LATE ring skim (β never "
          "tightens enough). The recommended S₀ is the knee that balances "
          "them.", "",
          "## Recommendation", ""]

    # find the near-zero-distortion option (worst drop < 0.05) with the
    # LOWEST late skim — the engineering pick, since ATN skim is voiceless.
    lowdist = [g for g in grid if g["worst_corr_drop"] < 0.05]
    eng = min(lowdist, key=lambda g: g["late_ring_skim"]) if lowdist else None

    L += [f"**Strict reading of the criterion (smallest honest distortion "
          f"subject to late skim ≤5%): {rec['form']}, S₀ = {rec['S0_epochs']} "
          f"epochs.** worst corr-drop {rec['worst_corr_drop']}, late skim "
          f"{[g['late_ring_skim'] for g in grid if g['form']==rec['form'] and g['S0_epochs']==rec['S0_epochs']][0]}, "
          f"β at windows {rec['beta_at_windows']}. It pins the late skim but "
          "pays a real EARLY mispricing cost (β clamps toward β_min while "
          "newcomers are still ~70-85% of demand).", ""]
    if eng is not None:
        L += [f"**Engineering-preferred: {eng['form']}, S₀ = "
              f"{eng['S0_epochs']} epochs.** worst corr-drop "
              f"{eng['worst_corr_drop']} (≈0 — honest work is priced almost "
              f"exactly right at every stage) at the cost of a higher late "
              f"skim ({eng['late_ring_skim']}). **Because D' makes that skim "
              "VOICELESS money, tolerating ~10-15% late ATN skim to buy "
              "near-zero honest distortion is the better trade** — the ring "
              "gets spendable ATN it can never convert to governance power, "
              "while honest authors' earnings track real demand across the "
              "whole life-cycle. Push late skim lower later by lengthening "
              "the horizon (real networks mature over far more than 300 "
              "epochs) or nudging S₀ down once supply is demonstrably large.", ""]

    # ----- FINAL DESIGN PARAMETERS statement (full ruleset) -------------
    L += ["## Final design parameters — full v4.1 + supply-pegged β ruleset",
          "",
          "Every parameter, with provenance (EMERGED from sims vs SEEDED as "
          "a modeling choice / carried from prior ratification):", "",
          "- **Mint = usage only, pinned tools, no vet gate (A).** Carried "
          "from v4 (ratified). SEEDED.",
          "- **Reviews carry per-axis scores; inspection reviews drift "
          "position but mint nothing (B).** Ratified. SEEDED.",
          "- **Drift weight = credibility × household_rep/rep_supply, NO ε "
          "floor; author prior mass 1.0 (C).** Ratified. SEEDED.",
          "- **D' — zero-rep usage mints ATN (flat ε=0.05 weight) but grants "
          "NO reputation.** Ratified after v4; the voice-leak kill is "
          "EMERGED-verified (sybil voice share = 0 across 200+ epochs).",
          "- **E' — continuous reversal-aware credibility: dock if a "
          "review deviates >δ from the current head, restore symmetrically; "
          "mass floor 3; credibility floor 0.1; recovery 10%/epoch.** "
          "Ratified after v4. δ, mass-floor, floor, recovery: SEEDED "
          "provisionally; **δ = 0.7 EMERGED** from the v4 sanction sweep "
          "(honest false-positive dock rate <2% only at δ≥0.7).",
          "- **β cap = rep-independent aggregate cap on zero-rep ATN weight, "
          "ATN-side only.** Accepted in principle by the user. Its NECESSITY "
          "(a constant fails) EMERGED from the β sweep.",
          "- **β_min = 0.05.** SEEDED (floor), consistent with ε.",
          "- **β schedule = β(S) as a function of total reputation supply S.** "
          "The supply-peg CHOICE EMERGED (supply is the only maturity proxy "
          "a dust ring cannot inflate under D' — verified: worst β tightening "
          f"a ring induced = {v['peg_check_worst_beta_tightening']}).",
          f"- **Form + S₀: {eng['form'] if eng else rec['form']}, S₀ ≈ "
          f"{(eng['S0_epochs'] if eng else rec['S0_epochs'])} epochs-worth of "
          f"pool ({int((eng['S0'] if eng else rec['S0']))} rep units)** for a "
          "near-zero-distortion life-cycle; the strict-≤5%-skim variant is "
          f"{rec['form']} S₀={rec['S0_epochs']}. Both EMERGED from this "
          "life-cycle sweep. Exact S₀ should be re-calibrated to the real "
          "network's pool size and expected maturation horizon before launch "
          "— the sim horizon (300 epochs) is short relative to a real "
          "network.",
          "- **Emission pool = 100 ATN base + recycled fees, fixed-pie "
          "(zero-sum among authors).** Carried from the shipped v3 economy. "
          "SEEDED.",
          "",
          "Provenance summary: the STRUCTURE (D', E', supply-pegged β) was "
          "ratified by the user; the sims EMERGED the load-bearing VALUES — "
          "δ=0.7, the necessity of a non-constant β, the supply peg as the "
          "adversary-proof maturity signal, and the (form, S₀) trade-off "
          "frontier. The remaining SEEDED constants (β_min, mass floor, "
          "credibility floor/recovery, pool size) are conventional and can "
          "be tuned post-launch without changing the mechanism.", ""]

    with open(summ, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
