"""Render results/summary_fees_only.md from the fees-only verdict dict.
Kept in a separate module so run_fees_only.py stays a thin runner and the
prose can be iterated without re-running the sims. SIM-ONLY.

The narrative prose (verdict, findings, options) is authored in
summary_fees_only.md AFTER the numbers are known; this module fills the
data tables and the machine-checkable verdict flags so they can never
drift from the JSON. See the file header comment there.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _f(x, n=4):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def write_summary(v: Dict[str, Any], results_dir: str) -> None:
    s1 = v.get("s1", {})
    s2 = v.get("s2", {})
    s2f = v.get("s2_fix", {})
    s3 = v.get("s3", {})
    s4 = v.get("s4", {})
    s5 = v.get("s5", {})
    s6 = v.get("s6", {})

    L = []
    L.append("<!-- DATA TABLES auto-rendered by summary_fees_only.py from "
             "results/fees_only/*.json. The narrative sections above the "
             "'--- machine tables ---' marker are hand-authored; do not let "
             "them contradict these tables. -->")
    L.append("")
    L.append("# fees-only + REP-from-earnings — machine tables")
    L.append("")

    # S1
    L.append("## S1 honest baseline")
    L.append(f"- quality↔ATN-earnings corr: **{_f(s1.get('quality_vs_atn_earnings_corr'))}**")
    L.append(f"- quality↔REP corr: **{_f(s1.get('quality_vs_rep_corr'))}**")
    L.append(f"- author income as frac of service GMV: {_f(s1.get('author_income_frac_of_gmv'),5)}")
    L.append(f"- burn as frac of GMV (should ≈ {0.0125}): {_f(s1.get('burn_frac_of_gmv'),5)}")
    L.append(f"- dead-start transition clean: **{s1.get('dead_start_transition_clean')}** "
             f"(pool 0 during dead: {s1.get('pool_zero_during_dead')}, "
             f">0 after: {s1.get('pool_positive_after_dead')})")
    L.append("")

    # S2
    L.append("## S2 usage-flood ring — THE loop")
    L.append(f"- ANY cell compounds (earnings→REP→weight loop grows): **{s2.get('any_compounds')}**")
    L.append(f"- max TRANSITION pool capture (first funded epoch, one-shot): "
             f"**{_f(s2.get('max_transition_capture'))}** "
             f"(worst cell: {s2.get('worst_transition_cell')})")
    L.append(f"- max LATE pool capture (steady state): **{_f(s2.get('max_late_capture'))}**")
    L.append(f"- max ring REP-share (of supply): **{_f(s2.get('max_ring_rep_share'),6)}**")
    L.append("")
    L.append("| cell (stage/K/topology) | transition cap | peak cap | late cap | final ring REP-share | compounds |")
    L.append("|---|---|---|---|---|---|")
    pct = s2.get("pool_capture_transition_by_cell", {})
    pcp = s2.get("pool_capture_peak_by_cell", {})
    pcl = s2.get("pool_capture_late_by_cell", {})
    rr = s2.get("ring_rep_share_final_by_cell", {})
    cp = s2.get("compounds_by_cell", {})
    for cell in pct:
        L.append(f"| {cell} | {_f(pct[cell])} | {_f(pcp.get(cell))} | {_f(pcl.get(cell))} | "
                 f"{_f(rr.get(cell),6)} | {cp.get(cell)} |")
    L.append("")
    L.append("### S2 with `service_rep_only=True` (fix candidate: REP only on service revenue)")
    L.append(f"- ANY cell compounds: **{s2f.get('any_compounds')}**")
    L.append(f"- max transition capture: **{_f(s2f.get('max_transition_capture'))}**")
    L.append(f"- max ring REP-share: **{_f(s2f.get('max_ring_rep_share'),6)}**")
    L.append("")

    # S3
    L.append("## S3 wash trading")
    L.append(f"- ring fee paid: {_f(s3.get('ring_fee_paid'),3)} | pool reclaimed: "
             f"{_f(s3.get('ring_pool_reclaimed'),3)} | net ATN cost: "
             f"**{_f(s3.get('net_atn_cost_of_wash'),3)}**")
    L.append(f"- strict-loss holds (ring loses ATN net): **{s3.get('strict_loss_holds')}** "
             f"(reclaim = {_f(s3.get('reclaim_frac_of_fee'))} of fee paid)")
    L.append(f"- ring REP gained from wash: {_f(s3.get('ring_rep_gained'),3)}")
    L.append(f"- wash voice-per-dollar: **{s3.get('wash_voice_per_dollar')}** vs "
             f"honest voice-per-dollar: **{_f(s3.get('honest_voice_per_dollar'))}**")
    L.append(f"- washing buys voice cheaper than honest service: "
             f"**{s3.get('wash_cheaper_than_honest_voice')}**")
    L.append("")

    # S4
    L.append("## S4 whale spender")
    L.append(f"- whale REP: {_f(s4.get('whale_rep'),6)} → earns zero REP: "
             f"**{s4.get('whale_earns_zero_rep')}** "
             f"(supply share {_f(s4.get('whale_supply_share'),6)})")
    L.append(f"- author REP with/without whale: {_f(s4.get('author_rep_with_whale'),2)} / "
             f"{_f(s4.get('author_rep_without_whale'),2)} "
             f"(uplift {_f(s4.get('author_rep_uplift_from_whale'),2)})")
    L.append(f"- provider REP with/without whale: {_f(s4.get('provider_rep_with_whale'),2)} / "
             f"{_f(s4.get('provider_rep_without_whale'),2)}")
    L.append("")

    # S5
    L.append("## S5 retroactivity (same-epoch vs carried dead-period usage)")
    L.append("Transition-epoch (first funded epoch) ring pool capture, by "
             "dead-period demand regime:")
    L.append("")
    L.append("| dead-period regime | same-epoch | carried | amplification | retro worse? |")
    L.append("|---|---|---|---|---|")
    L.append(f"| honest users BUSY | {_f(s5.get('busy_same_epoch_transition_capture'))} | "
             f"{_f(s5.get('busy_carried_transition_capture'))} | "
             f"×{_f(s5.get('busy_retro_amplification'),3)} | "
             f"{s5.get('retro_more_capturable_when_busy')} |")
    L.append(f"| honest users IDLE (only ring pre-farms) | "
             f"{_f(s5.get('idle_same_epoch_transition_capture'))} | "
             f"{_f(s5.get('idle_carried_transition_capture'))} | "
             f"×{_f(s5.get('idle_retro_amplification'),3)} | "
             f"**{s5.get('retro_more_capturable_when_idle')}** |")
    L.append(f"- steady-state capture (post-transition, both): "
             f"~{_f(s5.get('steady_capture_same_epoch'))} (collapses)")
    L.append("")

    # S6
    L.append("## S6 β/S0 relevance under demand-backed REP")
    L.append(f"- β still load-bearing (uncapped ring capture materially > capped): "
             f"**{s6.get('beta_load_bearing')}** "
             f"(max uncapped ring capture {_f(s6.get('max_uncapped_ring_capture'))})")
    L.append(f"- any single S0 robust across all fee-growth curves: "
             f"**{s6.get('any_single_s0_robust')}**")
    L.append("")
    L.append("| curve | S0 | uncapped cap | capped cap | uncapped corr | capped corr | corr drop |")
    L.append("|---|---|---|---|---|---|---|")
    for g in s6.get("grid", []):
        L.append(f"| {g['curve']} | {g['S0']} | {g['uncapped_ring_capture']} | "
                 f"{g['capped_ring_capture']} | {g['uncapped_honest_corr']} | "
                 f"{g['capped_honest_corr']} | {g['corr_drop']} |")
    L.append("")
    L.append("### S0 robustness across curves")
    L.append("| S0 | robust | worst ring capture | worst corr drop |")
    L.append("|---|---|---|---|")
    for s0, d in s6.get("s0_robustness", {}).items():
        L.append(f"| {s0} | {d['robust_across_curves']} | "
                 f"{d['worst_ring_capture']} | {d['worst_corr_drop']} |")
    L.append("")

    path = os.path.join(results_dir, "summary_fees_only_tables.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote {path}")
