#!/usr/bin/env python3
"""Phase 10 analysis — pure over persisted artifacts (guard #3).

Implements docs/phase10_prereg.md decision rules EXACTLY. Reads only:
  events_T.jsonl, events_E.jsonl   (H1 standing rows)
  retrieval_rows.jsonl             (H2 rows)
  mint_rows.jsonl                  (H3 rows)
and writes aggregate10.json + a readable summary. No LLM, no re-running
of the substrate — every number here is derived from rows the run
already persisted.

H1 (FINAL both directions): AUC of -standing as a classifier of
ground-truth defectiveness, per sweep cell; decision confirmed iff
mean(AUC(E)-AUC(T)) >= 0.15 AND AUC(E) >= 0.90 in every cell with H>=2.

H2: hit@1/hit@5/SEO-share per arm; density retained iff
hit@5(D)-hit@5(B) >= +0.10 (salted) AND hit@5(D) >= hit@5(B)-0.02 (clean).

H3 (exploratory sanity gate): Spearman rho of mint vs pass_rate with a
paired bootstrap 95% CI; expectation rho>=0.5, no pre-committed action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent

# Pre-registered thresholds (docs/phase10_prereg.md).
H1_AUC_GAP = 0.15         # mean AUC(E)-AUC(T) bar
H1_AUC_FLOOR = 0.90       # AUC(E) floor in every H>=2 cell
H2_HIT5_GAIN = 0.10       # hit@5(D)-hit@5(B) bar (salted)
H2_CLEAN_TOL = 0.02       # no-regression tolerance on the clean corpus
H3_RHO_EXPECT = 0.50      # exploratory expectation, not a gate

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 10


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# AUC (rank-based, ties handled) — pure numpy, no sklearn.
# ---------------------------------------------------------------------------

def auc(scores: List[float], labels: List[int]) -> float:
    """AUC of ``scores`` as a classifier of the positive ``labels`` (1).
    Mann-Whitney U / (n_pos * n_neg), tie-corrected via average ranks.
    Returns NaN when a class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0    # 1-based average rank for ties
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------

def h1_cell_aucs(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    """Per (H, S) cell: AUC of -standing predicting defectiveness."""
    by_cell: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault((r["H"], r["S"]), []).append(r)
    out: Dict[Tuple[int, int], float] = {}
    for cell, cell_rows in by_cell.items():
        scores = [-r["standing"] for r in cell_rows]        # -standing
        labels = [1 if r["defective"] else 0 for r in cell_rows]
        out[cell] = auc(scores, labels)
    return out


def h1_flip_boundary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exploratory: per (H,S) cell, the fraction of genuinely defective
    tools whose standing is still POSITIVE (the defect went unpunished) —
    the flip frontier the prereg asks for."""
    by_cell: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault((r["H"], r["S"]), []).append(r)
    out: List[Dict[str, Any]] = []
    for (H, S), cell_rows in sorted(by_cell.items()):
        defective = [r for r in cell_rows if r["defective"]]
        pos = sum(1 for r in defective if r["standing"] > 0)
        out.append({
            "H": H, "S": S, "n_defective": len(defective),
            "defective_still_positive": pos,
            "flip_rate": pos / len(defective) if defective else float("nan"),
        })
    return out


def h1_analyze(rows_t: List[Dict[str, Any]],
               rows_e: List[Dict[str, Any]]) -> Dict[str, Any]:
    auc_t = h1_cell_aucs(rows_t)
    auc_e = h1_cell_aucs(rows_e)
    cells = sorted(set(auc_t) & set(auc_e))
    gaps = [auc_e[c] - auc_t[c] for c in cells
            if not (np.isnan(auc_e[c]) or np.isnan(auc_t[c]))]
    mean_gap = float(np.mean(gaps)) if gaps else float("nan")

    # AUC(E) >= 0.90 in EVERY cell with H >= 2.
    floor_cells = [c for c in cells if c[0] >= 2]
    floor_ok = all(auc_e[c] >= H1_AUC_FLOOR for c in floor_cells
                   if not np.isnan(auc_e[c]))
    worst_floor = (min((auc_e[c] for c in floor_cells
                        if not np.isnan(auc_e[c])), default=float("nan")))

    confirmed = (not np.isnan(mean_gap) and mean_gap >= H1_AUC_GAP
                 and floor_ok)
    if confirmed:
        verdict = (
            f"H1 CONFIRMED: mean AUC(E)-AUC(T)={mean_gap:.4f} >= {H1_AUC_GAP} "
            f"AND AUC(E)>={H1_AUC_FLOOR} in every H>=2 cell "
            f"(worst={worst_floor:.4f}). CONSEQUENCE: the evidence rail "
            "(replay-verified CON support) is promoted to a launch "
            "requirement — build the in-protocol invocation-evidence "
            "attachment as its own task.")
    else:
        verdict = (
            f"H1 REFUTED: mean AUC(E)-AUC(T)={mean_gap:.4f} "
            f"(bar {H1_AUC_GAP}); AUC(E) floor in H>=2 cells "
            f"{'held' if floor_ok else 'FAILED'} (worst={worst_floor:.4f}, "
            f"bar {H1_AUC_FLOOR}). CONSEQUENCE: the 'executable ground "
            "truth' framing comes OUT of the spec's motivation section; "
            "tool mint launches gated on vetting + damper alone.")
    return {
        "auc_T": {f"H{h}_S{s}": auc_t[(h, s)] for (h, s) in cells},
        "auc_E": {f"H{h}_S{s}": auc_e[(h, s)] for (h, s) in cells},
        "auc_gap_per_cell": {f"H{h}_S{s}": auc_e[(h, s)] - auc_t[(h, s)]
                             for (h, s) in cells},
        "mean_auc_gap": mean_gap,
        "auc_gap_bar": H1_AUC_GAP,
        "aucE_floor_bar": H1_AUC_FLOOR,
        "aucE_floor_held": bool(floor_ok),
        "aucE_worst_H2plus": worst_floor,
        "confirmed": bool(confirmed),
        "verdict": verdict,
        "flip_boundary_T_EXPLORATORY": h1_flip_boundary(rows_t),
        "flip_boundary_E_EXPLORATORY": h1_flip_boundary(rows_e),
    }


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------

def h2_rates(rows: List[Dict[str, Any]], corpus: str, arm: str) -> Dict[str, float]:
    sub = [r for r in rows if r["corpus"] == corpus and r["arm"] == arm]
    n = len(sub)
    if n == 0:
        return {"n": 0, "hit1": float("nan"), "hit5": float("nan"),
                "seo_share_top5": float("nan")}
    return {
        "n": n,
        "hit1": sum(1 for r in sub if r["hit1"]) / n,
        "hit5": sum(1 for r in sub if r["hit5"]) / n,
        "seo_share_top5": float(np.mean([r["seo_share_top5"] for r in sub])),
    }


def h2_analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    salted = {arm: h2_rates(rows, "salted", arm) for arm in ("B", "C", "D")}
    clean = {arm: h2_rates(rows, "clean", arm) for arm in ("B", "D")}

    gain = salted["D"]["hit5"] - salted["B"]["hit5"]
    clean_reg = clean["D"]["hit5"] - clean["B"]["hit5"]
    retained = (gain >= H2_HIT5_GAIN and clean_reg >= -H2_CLEAN_TOL)
    if retained:
        verdict = (
            f"H2 RETAIN density blend: salted hit@5(D)-hit@5(B)={gain:+.4f} "
            f">= {H2_HIT5_GAIN} AND clean-corpus regression={clean_reg:+.4f} "
            f">= -{H2_CLEAN_TOL}. COVERAGE_DENSITY_WEIGHT stays at 0.5.")
    else:
        verdict = (
            f"H2 DEMOTE density: salted hit@5(D)-hit@5(B)={gain:+.4f} "
            f"(bar {H2_HIT5_GAIN}) / clean regression={clean_reg:+.4f} "
            f"(tol -{H2_CLEAN_TOL}). COVERAGE_DENSITY_WEIGHT -> 0 by "
            "default; density demotes to an experimental flag.")
    return {
        "salted": salted, "clean": clean,
        "hit5_gain_salted": gain,
        "hit5_gain_bar": H2_HIT5_GAIN,
        "clean_regression": clean_reg,
        "clean_tolerance": H2_CLEAN_TOL,
        "retained": bool(retained),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------

def spearman(x: List[float], y: List[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def h3_analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = [r["pass_rate"] for r in rows]
    y = [r["mint"] for r in rows]
    rho = spearman(x, y)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(rows)
    boots = []
    xa = np.asarray(x); ya = np.asarray(y)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        boots.append(spearman(list(xa[idx]), list(ya[idx])))
    boots = [b for b in boots if not np.isnan(b)]
    lo = float(np.percentile(boots, 2.5)) if boots else float("nan")
    hi = float(np.percentile(boots, 97.5)) if boots else float("nan")

    minted = [r for r in rows if r["mint"] > 0]
    wash_minted = sum(1 for r in minted if r["wash"])
    return {
        "spearman_rho": rho,
        "ci95": [lo, hi],
        "expectation": H3_RHO_EXPECT,
        "meets_expectation": (not np.isnan(rho)) and rho >= H3_RHO_EXPECT,
        "n_tools": n,
        "n_minted": len(minted),
        "wash_tools_minted": wash_minted,
        "note": ("H3 is a sanity gate, not a pre-committed action — the "
                 "mint path already carries its own ratified mechanisms "
                 "(sims, damper, vetting). rho below expectation is a "
                 "finding to investigate, not a gate."),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE))
    ap.add_argument("--out", default=str(HERE / "aggregate10.json"))
    args = ap.parse_args()
    d = Path(args.dir)

    rows_t = load_jsonl(d / "events_T.jsonl")
    rows_e = load_jsonl(d / "events_E.jsonl")
    retr = load_jsonl(d / "retrieval_rows.jsonl")
    mint = load_jsonl(d / "mint_rows.jsonl")

    h1 = h1_analyze(rows_t, rows_e)
    h2 = h2_analyze(retr)
    h3 = h3_analyze(mint)

    aggregate = {
        "H1": h1, "H2": h2, "H3": h3,
        "bootstrap": {"n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
    }
    Path(args.out).write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    print(f"wrote {args.out}\n")

    _print_summary(aggregate)
    return 0


def _print_summary(agg: Dict[str, Any]) -> None:
    h1, h2, h3 = agg["H1"], agg["H2"], agg["H3"]
    print("  === H1: CON termination (AUC of -standing) ===")
    print("    cell  AUC(T)  AUC(E)   gap")
    for cell in sorted(h1["auc_T"]):
        print(f"    {cell:>7}  {h1['auc_T'][cell]:.3f}  {h1['auc_E'][cell]:.3f}  "
              f"{h1['auc_gap_per_cell'][cell]:+.3f}")
    print(f"    mean AUC gap = {h1['mean_auc_gap']:+.4f} (bar {h1['auc_gap_bar']})")
    print(f"    AUC(E) floor in H>=2 cells: worst {h1['aucE_worst_H2plus']:.3f} "
          f"(bar {h1['aucE_floor_bar']}, {'held' if h1['aucE_floor_held'] else 'FAILED'})")
    print(f"    {h1['verdict']}\n")

    print("  === H2: density-blend retrieval ===")
    for arm in ("B", "C", "D"):
        s = h2["salted"][arm]
        print(f"    salted {arm}: hit@1={s['hit1']:.3f} hit@5={s['hit5']:.3f} "
              f"seo_share={s['seo_share_top5']:.3f}  n={s['n']}")
    for arm in ("B", "D"):
        c = h2["clean"][arm]
        print(f"    clean  {arm}: hit@5={c['hit5']:.3f}  n={c['n']}")
    print(f"    {h2['verdict']}\n")

    print("  === H3: mint prices quality (Spearman) ===")
    lo, hi = h3["ci95"]
    print(f"    rho(mint, pass_rate) = {h3['spearman_rho']:.4f}  "
          f"CI95=[{lo:.4f}, {hi:.4f}]  (expectation {h3['expectation']})")
    print(f"    minted {h3['n_minted']}/{h3['n_tools']} tools; "
          f"wash tools minted: {h3['wash_tools_minted']}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
