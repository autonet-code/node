"""Run the fees-only + REP-from-earnings scenarios; write
results/fees_only/*.json + results/summary_fees_only.md.

SIM-ONLY (see fees_only_rules.py header). Nothing committed. This is a
NEW study alongside the v3/v4/v4.1 harness — it adversarially validates
the 2026-07-10 ratified model BEFORE any spec/build.

    python run_fees_only.py            # full run
    python run_fees_only.py --quick    # short epoch counts, fast smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict

logging.disable(logging.INFO)

import scenarios_fees_only as s  # noqa: E402
from summary_fees_only import write_summary  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FODIR = os.path.join(RESULTS, "fees_only")


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("records", "same_epoch_records", "carried_records",
                     "idle_carried_records") and \
                    isinstance(v, list) and len(v) > 40:
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
    os.makedirs(FODIR, exist_ok=True)
    path = os.path.join(FODIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_strip(result), f, indent=2, default=str)
    print(f"  wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    Q = args.quick
    E = (dict(s1=40, s2=50, s3=40, s4=40, s5=50, s6=60) if Q
         else dict(s1=200, s2=160, s3=120, s4=120, s5=120, s6=200))
    K2 = [5, 20] if Q else [5, 20, 100]
    stages = ["genesis", "mature"] if Q else ["genesis", "young", "mature"]

    out: Dict[str, Any] = {}

    print("[1/6] s1_honest_baseline")
    t0 = time.time()
    r = s.s1_honest_baseline(epochs=E["s1"], seed=1)
    _write("s1_honest_baseline", r); out["s1"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[2/6] s2_usage_flood (THE loop)")
    t0 = time.time()
    r = s.s2_usage_flood(epochs=E["s2"], seed=2, K_values=K2, stages=stages)
    _write("s2_usage_flood", r); out["s2"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[2b/6] s2_usage_flood (service_rep_only fix candidate)")
    t0 = time.time()
    r2 = s.s2_usage_flood(epochs=E["s2"], seed=2, K_values=K2, stages=stages,
                          service_rep_only=True)
    _write("s2_usage_flood_servicereponly", r2); out["s2_fix"] = r2["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[3/6] s3_wash_trading")
    t0 = time.time()
    r = s.s3_wash_trading(epochs=E["s3"], seed=3)
    _write("s3_wash_trading", r); out["s3"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[4/6] s4_whale_spender")
    t0 = time.time()
    r = s.s4_whale_spender(epochs=E["s4"], seed=4)
    _write("s4_whale_spender", r); out["s4"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[5/6] s5_retroactivity")
    t0 = time.time()
    r = s.s5_retroactivity(epochs=E["s5"], seed=5)
    _write("s5_retroactivity", r); out["s5"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    print("[6/6] s6_beta_relevance")
    t0 = time.time()
    r = s.s6_beta_relevance(epochs=E["s6"], seed=6)
    _write("s6_beta_relevance", r); out["s6"] = r["verdict"]
    print(f"  {time.time()-t0:.1f}s")

    write_summary(out, RESULTS)
    print(f"\nwrote {os.path.join(RESULTS, 'summary_fees_only_tables.md')} "
          "(machine tables).")
    print("The hand-authored narrative verdict is results/summary_fees_only.md "
          "(regenerate its numbers from the tables file if scenarios change).")
    return 0


def main_safe(argv=None):
    try:
        return main(argv)
    except AssertionError as e:
        print(f"SCENARIO ASSERT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main_safe())
