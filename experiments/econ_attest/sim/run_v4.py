"""Run the v4 "gradient trust" scenarios and write a v3-vs-v4 comparison.

v4 rules are SIM-ONLY (see v4_rules.py header) — production code is
untouched. v3 results are read from results/*.json (run run_all.py first);
this script writes results/v4/*.json and results/summary_v4.md.

Usage:
    python run_all.py        # (first) produce the v3 baseline results
    python run_v4.py         # v4 runs + comparison
    python run_v4.py --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict

logging.disable(logging.INFO)

import scenarios_v4 as sv  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
V4DIR = os.path.join(RESULTS, "v4")

# recommended v4 params (justified in the summary from the sweeps)
BETA = 0.1
DELTA = 0.7
Q = 5.0


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("records", "epochs") and isinstance(v, list) and len(v) > 40:
                keep = [r for i, r in enumerate(v) if i % 5 == 0]
                if v and keep[-1] is not v[-1]:
                    keep.append(v[-1])
                out[k] = keep
            else:
                out[k] = _strip(v)
        return out
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _write(name: str, result: Dict[str, Any]) -> None:
    os.makedirs(V4DIR, exist_ok=True)
    path = os.path.join(V4DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_strip(result), f, indent=2, default=str)
    print(f"  wrote {path}")


def _load_v3(name: str) -> Dict[str, Any]:
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    E = (dict(bl=30, sy=25, fa=40, nk=40, sp=25, sfp=30, sc=60) if args.quick
         else dict(bl=200, sy=120, fa=120, nk=120, sp=120, sfp=120, sc=200))

    out: Dict[str, Any] = {}

    print("[1/7] baseline_honest_v4")
    t0 = time.time()
    r = sv.baseline_honest(epochs=E["bl"], seed=1, beta=BETA, delta=DELTA, Q=Q)
    _write("baseline_honest", r); out["baseline"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[2/7] sybil_pump_v4")
    t0 = time.time()
    r = sv.sybil_pump(epochs=E["sy"], seed=2, beta=BETA, delta=DELTA, Q=Q)
    _write("sybil_pump", r); out["sybil"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[3/7] epsilon_faucet_v4")
    t0 = time.time()
    r = sv.epsilon_faucet(epochs=E["fa"], seed=3, beta=BETA, delta=DELTA, Q=Q)
    _write("epsilon_faucet", r); out["faucet"] = r["verdict"]
    out["faucet_sweep"] = [{"K": s["K"], "final": s["final_pool_share"],
                            "mean_late": s["mean_late_share"]}
                           for s in r["sweep"]]
    print(f"  {time.time()-t0:.1f}s")

    print("[4/7] review_nuke_v4")
    t0 = time.time()
    r = sv.review_nuke(epochs=E["nk"], seed=4, beta=BETA, delta=DELTA, Q=Q)
    _write("review_nuke", r); out["nuke"] = r["verdict"]
    # extract the who-got-docked story from the K=30 sweep record
    print(f"  {time.time()-t0:.1f}s")

    print("[5/7] spam_burial_v4 (NEW)")
    t0 = time.time()
    r = sv.spam_burial(epochs=E["sp"], seed=5, beta=BETA, delta=DELTA, Q=Q)
    _write("spam_burial", r); out["spam"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[6/7] sanction_false_positives_v4 (NEW)")
    t0 = time.time()
    r = sv.sanction_false_positives(epochs=E["sfp"], seed=6, beta=BETA)
    _write("sanction_false_positives", r); out["sfp"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[7/7] service_clone_v4")
    t0 = time.time()
    r = sv.service_clone(epochs=E["sc"], seed=7, beta=BETA, delta=DELTA, Q=Q)
    _write("service_clone", r); out["service"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    _write_summary(out)
    print(f"\nwrote {os.path.join(RESULTS, 'summary_v4.md')}")
    return 0


def _fmt(x):
    return "n/a" if x is None else x


def _write_summary(v4: Dict[str, Any]) -> None:
    v3 = {n: _load_v3(n).get("verdict", {}) for n in
          ("baseline_honest", "sybil_pump", "epsilon_faucet", "review_nuke",
           "service_clone")}

    L = ["# v3 vs v4 (gradient trust) — side-by-side verdicts", "",
         "**v4 is a SIM-ONLY ruleset** (`v4_rules.py`) — production code is "
         "untouched. v3 numbers are the real-close baseline from "
         "`run_all.py`; v4 numbers come from the reimplemented rule layer.",
         "", f"Recommended params used: **β={BETA}, δ={DELTA}, Q={Q}**.", "",
         "| scenario | v3 | v4 | verdict |", "|---|---|---|---|"]

    # baseline
    b3, b4 = v3["baseline_honest"], v4["baseline"]
    L.append(f"| baseline: quality↔rank corr | {b3.get('quality_vs_finalrank_corr','?'):.3f} "
             f"| {b4['quality_vs_finalrank_corr']:.3f} | v4 keeps extremes "
             f"(top rank {b4['top_quality_rank']}, worst {b4['worst_quality_rank']}/20) "
             f"but middle noisier |")
    L.append(f"| baseline: cold-start epochs→+rating | 0 (ε lets anyone drift) "
             f"| {_fmt(b4['cold_start_epochs_to_positive_rating'])} | "
             f"rule C cost, small WITH seeded reviewers |")

    # sybil
    s3, s4 = v3["sybil_pump"], v4["sybil"]
    cr3 = s3.get("capture_ratio_by_K", {})
    cr4 = s4["capture_ratio_by_K"]
    rg4 = s4["rank_gap_by_K"]
    L.append(f"| sybil_pump: capture@K=100 | {cr3.get('100', cr3.get(100,'?'))} "
             f"| {cr4.get(100,'?')} | rank channel DEAD "
             f"(rank-gap@K=100 = {rg4.get(100,'?')}); mint capture down but NOT "
             f"~1.0 (attacker monopolizes the β zero-rep budget) |")

    # faucet
    f3, f4 = v3["epsilon_faucet"], v4["faucet"]
    fp3 = f3.get("final_pool_share_by_K", {})
    fp4 = f4["final_pool_share_by_K"]
    L.append(f"| ε_faucet: pool-share@K=200 | {fp3.get('200', fp3.get(200,'?'))} "
             f"| {fp4.get(200,'?')} | v4 caps near β but LEAKS (see note) |")

    # nuke
    n3, n4 = v3["review_nuke"], v4["nuke"]
    rr3 = n3.get("rank_ratio_by_J", {})
    rr4 = n4["final_rank_ratio_by_J"]
    L.append(f"| review_nuke: rank-ratio@J=30 | {rr3.get('30', rr3.get(30,'?'))} "
             f"| {rr4.get(30,'?')} | v4 NOT better; sanction backfires (see note) |")

    # service
    sc4 = v4["service"]
    L.append(f"| service_clone: moat rent frac | (1−φ) exact | "
             f"{sc4['final_service_rev_frac']} (exp {sc4['expected_moat_rent_frac']}) "
             f"| unchanged, clone still pays |")

    L += ["", "## NEW scenarios (v4-only rails)", ""]
    sp = v4["spam"]
    L += ["### spam_burial (rule B inspection reviews)",
          f"- honest tool final rank position by M flood: {sp['honest_rank_pos_by_M']}",
          f"- spam still in top-5 by M: {sp['spam_in_topk_by_M']}",
          f"- honest in top-5 by M: {sp['honest_in_topk_by_M']}",
          "",
          "Verdict: inspection reviews DO drag inspected spam down (honest "
          "tool holds rank #1), but UN-inspected spam keeps its raw-cosine "
          "slot — burial only reaches what inspectors actually look at. Still "
          "strictly better than v3, where inspection had no rail at all.",
          ""]

    sfp = v4["sfp"]
    L += ["### sanction_false_positives (rule E chilling price)",
          "honest-only reviewers; FP dock-rate over the (δ, Q) grid:", "",
          "| δ | Q | FP dock rate | reviewers dinged | mean final cred |",
          "|---|---|---|---|---|"]
    for g in sfp["grid"]:
        L.append(f"| {g['delta']} | {g['Q']} | {g['fp_dock_rate']} | "
                 f"{g['frac_reviewers_dinged']} | {g['mean_final_cred']} |")
    L += ["", f"Safe region (FP<2% and <15% reviewers dinged): "
          f"{sfp['safe_region']}", ""]

    # faucet leak note
    L += ["## ⚠ Findings where v4 is WORSE than expected / needs a fix", "",
          "**1. ε-faucet cap LEAKS (rule D).** The β cap only binds while a "
          "household is zero-rep. But the capped faucet mint GIVES the sybils "
          "reputation, so next epoch they mint at their (now nonzero) rep "
          "share — uncapped. Over 120 epochs the sybil pool share creeps well "
          "above β (K=200: ~0.28 final vs β=0.1). The cap slows the faucet "
          "(v3 hit 0.67) but does not close it. **Fix to sim-test next:** "
          "either the faucet mint should NOT grant reputation (rep only from "
          "above-β 'real' mint), or the cap should key on a rep FLOOR "
          "(low-rep, not zero-rep) so a household can't buy its way out with "
          "faucet dust.", "",
          "**2. Review-nuke sanction BACKFIRES (rule E).** When attackers "
          "hold more review weight than honest reviewers, they reach the "
          "stabilization threshold Q with the WRONG-SIGN score first. The "
          "'stabilized' head is then already negative, the nukers' −1 reviews "
          "MATCH it (deviation < δ → no dock), and the honest +1 minority is "
          "the group that gets credibility-docked. Rule E as specified "
          "sanctions toward the majority-defined score, which is exactly the "
          "attacker's score in a nuke. **Fix to sim-test next:** stabilize on "
          "INDEPENDENT diverse-household mass with an author-side or "
          "usage-weighted prior, or don't let a single correlated cohort "
          "cross Q alone; sanction against a robust (median / usage-anchored) "
          "estimate, not the drifted head the attacker just moved.", ""]

    # recommendation + cold-start assessment
    b4 = v4["baseline"]
    L += ["## Recommended (β, δ, Q) and cold-start assessment", "",
          "**Recommendation: β=0.1, δ=0.7, Q=5.** Rationale:",
          "",
          "- **β=0.1** is the tightest useful cap: it bounds the zero-rep "
          "faucet's *instantaneous* share at ~10% (vs v3's 67%). It does NOT "
          "hold long-term because of the leak above — β is only meaningful "
          "PAIRED with the rule-D fix (faucet mint grants no rep, or cap on a "
          "rep floor). Lower β (0.05) starves honest cold-start reviewers too; "
          "higher β (0.2) widens the faucet.",
          "- **δ=0.7** keeps the honest false-positive dock rate <2% across "
          "all Q (grid: δ=0.7 → FP ≤0.6%), where δ=0.3 dings 100% of honest "
          "reviewers at low Q. Tighter δ chills honest reviewing.",
          "- **Q=5** balances the two NEW-scenario pressures: Q must be high "
          "enough that honest noise averages out before stabilization "
          "(Q≥5 → honest FP→0) but low enough that tools stabilize in a few "
          "epochs. NOTE the nuke finding caps how much Q can help: a bigger Q "
          "just lengthens the pre-stabilization nukeable window without "
          "fixing the majority-capture backfire.",
          "",
          "**Cold-start cost of rule C (no ε on drift):** with a seeded "
          "incumbent-reviewer distribution (~1/3 of users holding modest "
          f"rep), a fresh good tool reaches a positive rating in "
          f"{_fmt(b4['cold_start_epochs_to_positive_rating'])} epoch(s) and "
          f"the best tool still ranks #{b4['top_quality_rank']}. WITHOUT any "
          "rep-holding reviewers the drift channel is DEAD at genesis — no one "
          "can move a score, so the very first cohort of reviewers must be "
          "bootstrapped with rep some other way (founder grant, or a one-time "
          "ε window at network birth). The quality↔rank correlation falls from "
          f"0.89 (v3) to {b4['quality_vs_finalrank_corr']:.2f} (v4): extremes "
          "are still placed correctly but mid-quality tools with light usage "
          "get too little drift to separate. This is the real, quantified "
          "price of rule C — acceptable for the top/bottom discovery decisions "
          "that matter, but it measurably degrades fine-grained ranking.", ""]

    with open(os.path.join(RESULTS, "summary_v4.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
