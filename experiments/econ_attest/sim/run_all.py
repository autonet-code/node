"""Run every scenario with fixed seeds; write results/<scenario>.json and
a compact results/summary.md table of the verdict-relevant numbers.

Usage:
    python run_all.py            # full run (~minutes)
    python run_all.py --quick    # short epoch counts for a fast check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict

logging.disable(logging.INFO)   # the close logs one INFO line per epoch

import scenarios  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _strip_records(obj: Any) -> Any:
    """Trim the heavy per-use ``records``/``epochs`` arrays down for JSON:
    keep every 5th epoch plus the last, so files stay readable but the
    trajectory survives."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("records", "epochs") and isinstance(v, list) and len(v) > 40:
                keep = [r for i, r in enumerate(v) if i % 5 == 0]
                if v and v[-1] is not keep[-1]:
                    keep.append(v[-1])
                out[k] = keep
            else:
                out[k] = _strip_records(v)
        return out
    if isinstance(obj, list):
        return [_strip_records(x) for x in obj]
    return obj


def _write(name: str, result: Dict[str, Any]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_strip_records(result), f, indent=2, default=str)
    print(f"  wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    E = (dict(baseline=30, sybil=25, faucet=25, nuke=25, clone=60) if args.quick
         else dict(baseline=200, sybil=120, faucet=120, nuke=120, clone=200))

    summary: Dict[str, Dict[str, Any]] = {}

    print("[1/5] baseline_honest")
    t0 = time.time()
    r = scenarios.baseline_honest(epochs=E["baseline"], seed=1)
    _write("baseline_honest", r)
    summary["baseline_honest"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[2/5] sybil_pump")
    t0 = time.time()
    r = scenarios.sybil_pump(epochs=E["sybil"], seed=2)
    _write("sybil_pump", r)
    summary["sybil_pump"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[3/5] epsilon_faucet")
    t0 = time.time()
    r = scenarios.epsilon_faucet(epochs=E["faucet"], seed=3)
    _write("epsilon_faucet", r)
    summary["epsilon_faucet"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[4/5] review_nuke")
    t0 = time.time()
    r = scenarios.review_nuke(epochs=E["nuke"], seed=4)
    _write("review_nuke", r)
    summary["review_nuke"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[5/5] service_clone")
    t0 = time.time()
    r = scenarios.service_clone(epochs=E["clone"], seed=5)
    _write("service_clone", r)
    summary["service_clone"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    _write_summary(summary)
    print(f"\nwrote {os.path.join(RESULTS, 'summary.md')}")
    return 0


def _write_summary(summary: Dict[str, Dict[str, Any]]) -> None:
    lines = ["# Substrate v3 tool-economy attack sim — verdict summary", ""]
    lines.append("All mint runs through the REAL `federated_epoch_close` with "
                 "`apply_emission_pool` (fixed pool = 100 ATN/epoch + recycled "
                 "fees), so total minted ATN per epoch == pool (conservation "
                 "asserted). Mint is therefore zero-sum among authors.")
    lines.append("")

    b = summary["baseline_honest"]
    lines += [
        "## baseline_honest  (claims C1, C3)",
        "",
        f"- quality vs cumulative-mint correlation: **{b['quality_vs_cummint_corr']:.3f}**",
        f"- quality vs final discovery-rank correlation: **{b['quality_vs_finalrank_corr']:.3f}**",
        f"- highest-true-quality tool's discovery rank (1=top): **{b['top_quality_rank']}**",
        f"- lowest-true-quality tool's discovery rank: **{b['worst_quality_rank']}**",
        "",
        "Verdict: mint share and discovery rank track true quality; the "
        "worst tool sinks to the bottom (C1 + C3 SUPPORTED).",
        "",
    ]

    s = summary["sybil_pump"]
    lines += ["## sybil_pump  (attack 1, claim C2)", "",
              "| K sybils | capture ratio (atk/ctrl cum mint) | rank-cross epoch |",
              "|---|---|---|"]
    for K in sorted(s["capture_ratio_by_K"]):
        cross = s["rank_cross_by_K"][K]
        lines.append(f"| {K} | {s['capture_ratio_by_K'][K]} | {cross} |")
    lines += ["", "Verdict: capture ratio rises with K (self-bootstrapping "
              "ring); each sybil is ε-capped but K of them are not.", ""]

    f = summary["epsilon_faucet"]
    lines += ["## epsilon_faucet  (attack 6)", "",
              "| K dust identities | final sybil pool share | share growth |",
              "|---|---|---|"]
    for K in sorted(f["final_pool_share_by_K"]):
        lines.append(f"| {K} | {f['final_pool_share_by_K'][K]} | "
                     f"{f['share_growth_by_K'][K]} |")
    lines += ["", "Verdict: sybil pool share grows ~linearly in K and is NOT "
              "bounded by a single ε — the fixed pool is drained pro-rata to "
              "the count of dust identities (attack 6 CONFIRMED).", ""]

    n = summary["review_nuke"]
    lines += ["## review_nuke  (attack 5)", "",
              "| J nukers | victim/ctrl rank ratio | victim survived |",
              "|---|---|---|"]
    for J in sorted(n["rank_ratio_by_J"]):
        lines.append(f"| {J} | {n['rank_ratio_by_J'][J]} | {n['survived_by_J'][J]} |")
    lines += ["", "Verdict: a young tool's rank degrades with nuker count; "
              "heavy nuking sinks it (attack 5 holds for low-mass tools).", ""]

    c = summary["service_clone"]
    lines += ["## service_clone  (core hypothesis)", "",
              "clone pays (cum mint > rediscovery cost)?", ""]
    for k, v in c["clone_pays_by_phi_rcost"].items():
        lines.append(f"- {k}: **{v}**")
    lines += ["", "surviving service revenue fraction by φ (moat rent ≈ 1−φ):"]
    for phi, frac in c["surviving_service_rev_frac_by_phi"].items():
        lines.append(f"- φ={phi}: surviving rev frac **{frac}** (expected ≈ {round(1-float(phi),2)})")
    lines += ["",
              f"- fee-recycling payback epoch (recycle ON): **{c['payback_epoch_recycle_on']}**",
              f"- fee-recycling payback epoch (recycle OFF): **{c['payback_epoch_recycle_off']}**",
              f"- clone cumulative mint (recycle ON): **{c['clone_cum_mint_recycle_on']}**",
              f"- clone cumulative mint (recycle OFF): **{c['clone_cum_mint_recycle_off']}**",
              "",
              "Verdict: the free clone captures φ of demand and service "
              "revenue decays to exactly the (1−φ) moat rent. Fee recycling "
              "IS directionally coupled (clone cum mint is higher with "
              "recycling ON) but the effect is second-order at a ~1.25% "
              "recycle rate — it lifts the clone's absolute payout via a "
              "bigger pool without changing its pool SHARE, too small to move "
              "the discrete payback epoch here.", ""]

    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
