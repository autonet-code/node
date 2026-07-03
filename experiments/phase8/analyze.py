#!/usr/bin/env python3
"""Phase 8 analysis — pure over grades.jsonl (contest) + contest_rows.jsonl.

Implements docs/phase8_prereg.md "Analysis (pre-registered)" EXACTLY:
  - Per (qid, arm) metric = mean of the two graders' `overall`.
  - Paired contrasts (paired per question): primary D256-C; secondary
    B-A, C-B, D64-D256.
  - scipy.stats.ttest_rel + Holm correction across the 4 contrasts.
  - Bootstrap 95% CI, 10k resamples, seed 8.
  - Inter-grader Pearson r + mean absolute disagreement on `overall`
    (across all graded answers); enforce r < 0.4 -> ungradeable gate.
  - Decision rule (verbatim): equilibration retained iff D256 > C at
    Holm-corrected p < 0.05 AND mean difference >= 0.25. Else demoted.
  - Expansion rule: if primary CI spans 0 AND |mean diff| < 0.3 ->
    recommend expansion to 50 questions.
  - Per-arm means +/- CI, per-category means (exploratory, labeled),
    all contrasts (raw p / Holm p / CI), agreement stats, decision
    verdict string, expansion recommendation flag -> aggregate8.json.
  - Readable summary table printed to stdout.

This script does not call any LLM and does not touch llm_cache/ — it is
pure over already-persisted grades.jsonl + contest_rows.jsonl (prereg
guard #7: "analysis script is pure over persisted artifacts and runs only
after all rows exist").

Usage:
    python analyze.py --grades grades.jsonl --contest contest_rows.jsonl \
        --out aggregate8.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
from scipy import stats  # type: ignore

ARMS = ("A", "B", "C", "D256", "D64")

# Contrast order matters: Holm correction is applied across exactly these 4,
# and the "primary" label is used by the decision rule / expansion rule.
CONTRASTS: List[Tuple[str, str, str]] = [
    ("primary", "D256", "C"),
    ("secondary_B_minus_A", "B", "A"),
    ("secondary_C_minus_B", "C", "B"),
    ("secondary_D64_minus_D256", "D64", "D256"),
]

DECISION_P_THRESHOLD = 0.05
DECISION_DIFF_THRESHOLD = 0.25
EXPANSION_DIFF_THRESHOLD = 0.3
AGREEMENT_R_GATE = 0.4

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 8


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Per-(qid, arm) metric = mean of the two graders' overall.
# ---------------------------------------------------------------------------


def build_metric_table(grade_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    """(qid, arm) -> mean overall across graders (non-null only)."""
    by_key: Dict[Tuple[str, str], List[float]] = {}
    for r in grade_rows:
        if r.get("null"):
            continue
        overall = r.get("overall")
        if overall is None:
            continue
        by_key.setdefault((r["qid"], r["arm"]), []).append(float(overall))
    return {k: sum(v) / len(v) for k, v in by_key.items()}


def paired_vectors(
    metric: Dict[Tuple[str, str], float], qids: List[str], arm_a: str, arm_b: str,
) -> Tuple[List[float], List[float], List[str]]:
    """Return (xa, xb, used_qids) for qids that have BOTH arms present."""
    xa, xb, used = [], [], []
    for qid in qids:
        a = metric.get((qid, arm_a))
        b = metric.get((qid, arm_b))
        if a is None or b is None:
            continue
        xa.append(a)
        xb.append(b)
        used.append(qid)
    return xa, xb, used


# ---------------------------------------------------------------------------
# Bootstrap CI for paired mean difference
# ---------------------------------------------------------------------------


def bootstrap_ci_mean_diff(
    xa: List[float], xb: List[float], n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED, alpha: float = 0.05,
) -> Tuple[float, float]:
    """Paired bootstrap 95% CI on mean(xa - xb). Resamples question indices
    with replacement (paired resample — preserves pairing)."""
    diffs = np.asarray(xa, dtype=np.float64) - np.asarray(xb, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    hi = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def bootstrap_ci_mean(
    x: List[float], n_resamples: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    hi = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def holm_correction(labels: List[str], raw_p: List[float]) -> Dict[str, float]:
    """Standard Holm step-down correction. Returns label -> holm-adjusted p.

    Adjusted p_(i) = max_{j<=i} ( (m - j + 1) * p_(j) ), sorted ascending,
    each capped at 1.0, and enforced monotone non-decreasing (standard Holm
    definition).
    """
    m = len(raw_p)
    order = sorted(range(m), key=lambda i: raw_p[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):  # rank is 0-indexed position in sorted order
        factor = m - rank
        val = min(1.0, factor * raw_p[i])
        running_max = max(running_max, val)
        adjusted[i] = running_max
    return {labels[i]: adjusted[i] for i in range(m)}


# ---------------------------------------------------------------------------
# Inter-grader agreement
# ---------------------------------------------------------------------------


def inter_grader_agreement(
    grade_rows: List[Dict[str, Any]], graders: Tuple[str, str] = ("opus", "sonnet"),
) -> Dict[str, Any]:
    """Pearson r + mean absolute disagreement on `overall`, paired by (qid, arm),
    across all graded answers (non-null in both graders)."""
    g1, g2 = graders
    by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in grade_rows:
        if r.get("null"):
            continue
        if r.get("grader") not in graders:
            continue
        overall = r.get("overall")
        if overall is None:
            continue
        key = (r["qid"], r["arm"])
        by_key.setdefault(key, {})[r["grader"]] = float(overall)

    xs, ys = [], []
    for key, d in by_key.items():
        if g1 in d and g2 in d:
            xs.append(d[g1])
            ys.append(d[g2])

    n = len(xs)
    if n < 2:
        return {
            "n_paired": n, "pearson_r": None, "mean_abs_disagreement": None,
            "ungradeable": True, "reason": "fewer than 2 paired (qid,arm) grades",
        }

    r, p = stats.pearsonr(xs, ys)
    mean_abs_disagreement = float(np.mean(np.abs(np.asarray(xs) - np.asarray(ys))))
    ungradeable = bool(r < AGREEMENT_R_GATE)
    return {
        "n_paired": n,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mean_abs_disagreement": mean_abs_disagreement,
        "ungradeable": ungradeable,
        "gate_threshold": AGREEMENT_R_GATE,
    }


# ---------------------------------------------------------------------------
# Per-arm means +/- CI, per-category (exploratory)
# ---------------------------------------------------------------------------


def per_arm_stats(metric: Dict[Tuple[str, str], float], qids: List[str]) -> Dict[str, Any]:
    out = {}
    for arm in ARMS:
        vals = [metric[(qid, arm)] for qid in qids if (qid, arm) in metric]
        if not vals:
            out[arm] = {"n": 0, "mean": None, "ci95": [None, None]}
            continue
        mean = float(np.mean(vals))
        lo, hi = bootstrap_ci_mean(vals)
        out[arm] = {"n": len(vals), "mean": mean, "ci95": [lo, hi]}
    return out


def per_category_stats(
    metric: Dict[Tuple[str, str], float], qid_category: Dict[str, str],
) -> Dict[str, Any]:
    """Exploratory only — labeled as such in output."""
    categories = sorted(set(qid_category.values()))
    out: Dict[str, Any] = {}
    for cat in categories:
        cat_qids = [qid for qid, c in qid_category.items() if c == cat]
        out[cat] = per_arm_stats(metric, cat_qids)
    return out


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def compute_contrasts(
    metric: Dict[Tuple[str, str], float], qids: List[str],
) -> Dict[str, Any]:
    labels = [c[0] for c in CONTRASTS]
    raw_results: Dict[str, Dict[str, Any]] = {}
    raw_p: List[float] = []

    for label, arm_hi, arm_lo in CONTRASTS:
        xa, xb, used = paired_vectors(metric, qids, arm_hi, arm_lo)
        n = len(used)
        if n < 2:
            raw_results[label] = {
                "arm_a": arm_hi, "arm_b": arm_lo, "n": n,
                "mean_diff": None, "t": None, "p_raw": None,
                "ci95": [None, None], "insufficient_data": True,
            }
            raw_p.append(1.0)
            continue
        t, p = stats.ttest_rel(xa, xb)
        mean_diff = float(np.mean(np.asarray(xa) - np.asarray(xb)))
        ci_lo, ci_hi = bootstrap_ci_mean_diff(xa, xb)
        raw_results[label] = {
            "arm_a": arm_hi, "arm_b": arm_lo, "n": n,
            "mean_diff": mean_diff, "t": float(t), "p_raw": float(p),
            "ci95": [ci_lo, ci_hi], "insufficient_data": False,
        }
        raw_p.append(float(p))

    holm_p = holm_correction(labels, raw_p)
    for label in labels:
        raw_results[label]["p_holm"] = holm_p[label]

    return raw_results


# ---------------------------------------------------------------------------
# Decision rule + expansion rule (verbatim from prereg)
# ---------------------------------------------------------------------------


def decision_rule(primary: Dict[str, Any]) -> Dict[str, Any]:
    """
    'equilibration is retained iff D256 > C at Holm-corrected p < 0.05 AND
    mean difference >= 0.25 (5-pt scale). Otherwise the gate in the
    course-of-action stands: mint pricing moves toward explicit debate
    standing (economics change ratified by the user), equilibration
    demoted to experimental kernel.'
    """
    if primary.get("insufficient_data"):
        return {
            "verdict": "INDETERMINATE",
            "reason": "insufficient paired data for primary contrast (D256-C)",
            "retain_equilibration": None,
        }
    mean_diff = primary["mean_diff"]
    p_holm = primary["p_holm"]
    retained = (mean_diff is not None and mean_diff > 0
                and p_holm is not None and p_holm < DECISION_P_THRESHOLD
                and mean_diff >= DECISION_DIFF_THRESHOLD)
    if retained:
        verdict = ("RETAIN equilibration: D256 > C at Holm p={:.4g} < {} "
                   "AND mean diff={:.4f} >= {}").format(
            p_holm, DECISION_P_THRESHOLD, mean_diff, DECISION_DIFF_THRESHOLD)
    else:
        verdict = (
            "DEMOTE equilibration (gate stands): D256-C mean diff={} "
            "Holm p={} — does not clear D256>C AND p<{} AND diff>={}. "
            "Mint pricing moves toward explicit debate standing (economics "
            "change ratified by the user); equilibration demoted to "
            "experimental kernel."
        ).format(
            "n/a" if mean_diff is None else f"{mean_diff:.4f}",
            "n/a" if p_holm is None else f"{p_holm:.4g}",
            DECISION_P_THRESHOLD, DECISION_DIFF_THRESHOLD,
        )
    return {
        "verdict": verdict,
        "retain_equilibration": retained,
        "mean_diff": mean_diff,
        "p_holm": p_holm,
        "p_threshold": DECISION_P_THRESHOLD,
        "diff_threshold": DECISION_DIFF_THRESHOLD,
    }


def expansion_rule(primary: Dict[str, Any]) -> Dict[str, Any]:
    """
    'if the primary CI spans 0 with |mean diff| < 0.3, expand once to 50
    questions (new questions, same protocol), then final.'
    """
    if primary.get("insufficient_data"):
        return {
            "recommend_expansion": None,
            "reason": "insufficient paired data for primary contrast (D256-C)",
        }
    ci_lo, ci_hi = primary["ci95"]
    mean_diff = primary["mean_diff"]
    ci_spans_zero = (ci_lo is not None and ci_hi is not None and ci_lo <= 0 <= ci_hi)
    small_effect = abs(mean_diff) < EXPANSION_DIFF_THRESHOLD if mean_diff is not None else False
    recommend = bool(ci_spans_zero and small_effect)
    return {
        "recommend_expansion": recommend,
        "ci_spans_zero": ci_spans_zero,
        "abs_mean_diff": None if mean_diff is None else abs(mean_diff),
        "diff_threshold": EXPANSION_DIFF_THRESHOLD,
        "reason": (
            f"CI [{ci_lo}, {ci_hi}] spans 0 ({ci_spans_zero}) AND "
            f"|mean diff|={None if mean_diff is None else abs(mean_diff):.4f} < "
            f"{EXPANSION_DIFF_THRESHOLD} ({small_effect})"
        ) if not primary.get("insufficient_data") else "n/a",
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(aggregate: Dict[str, Any]) -> None:
    print()
    print("  === Per-arm means (95% bootstrap CI) ===")
    for arm in ARMS:
        s = aggregate["per_arm"].get(arm, {})
        if s.get("mean") is None:
            print(f"    {arm:>6}: n=0 (no data)")
            continue
        lo, hi = s["ci95"]
        print(f"    {arm:>6}: mean={s['mean']:.3f}  CI=[{lo:.3f}, {hi:.3f}]  n={s['n']}")

    print()
    print("  === Contrasts (Holm-corrected across 4) ===")
    for label, _, _ in CONTRASTS:
        c = aggregate["contrasts"][label]
        if c.get("insufficient_data"):
            print(f"    {label:>28}: insufficient data")
            continue
        lo, hi = c["ci95"]
        print(f"    {label:>28}: {c['arm_a']}-{c['arm_b']} = {c['mean_diff']:+.4f}  "
              f"t={c['t']:+.3f}  p_raw={c['p_raw']:.4g}  p_holm={c['p_holm']:.4g}  "
              f"CI=[{lo:+.4f}, {hi:+.4f}]  n={c['n']}")

    print()
    print("  === Inter-grader agreement ===")
    ag = aggregate["agreement"]
    if ag.get("pearson_r") is not None:
        print(f"    Pearson r = {ag['pearson_r']:.4f}  "
              f"mean|disagreement| = {ag['mean_abs_disagreement']:.4f}  "
              f"n_paired = {ag['n_paired']}")
        print(f"    ungradeable gate (r < {AGREEMENT_R_GATE}): "
              f"{'TRIPPED — STOP, REVISE RUBRIC' if ag['ungradeable'] else 'clear'}")
    else:
        print(f"    insufficient data: {ag.get('reason')}")

    print()
    print("  === Decision rule ===")
    print(f"    {aggregate['decision']['verdict']}")

    print()
    print("  === Expansion rule ===")
    exp = aggregate["expansion"]
    print(f"    recommend_expansion = {exp['recommend_expansion']}  ({exp.get('reason')})")

    if aggregate.get("per_category"):
        print()
        print("  === Per-category means (EXPLORATORY, not confirmatory) ===")
        for cat, arms in aggregate["per_category"].items():
            parts = []
            for arm in ARMS:
                s = arms.get(arm, {})
                if s.get("mean") is not None:
                    parts.append(f"{arm}={s['mean']:.2f}")
            print(f"    {cat:>20}: " + "  ".join(parts))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades", required=True, help="grades.jsonl (contest grading)")
    parser.add_argument("--contest", required=True, help="contest_rows.jsonl")
    parser.add_argument("--out", required=True, help="aggregate8.json output path")
    args = parser.parse_args()

    grade_rows = load_jsonl(Path(args.grades))
    contest_rows = load_jsonl(Path(args.contest))

    qids = [r["qid"] for r in contest_rows]
    qid_category: Dict[str, str] = {
        r["qid"]: r.get("category", "unknown") for r in contest_rows
        if "category" in r
    }

    metric = build_metric_table(grade_rows)

    per_arm = per_arm_stats(metric, qids)
    contrasts = compute_contrasts(metric, qids)
    agreement = inter_grader_agreement(grade_rows)
    decision = decision_rule(contrasts["primary"])
    expansion = expansion_rule(contrasts["primary"])
    per_category = per_category_stats(metric, qid_category) if qid_category else {}

    aggregate: Dict[str, Any] = {
        "n_questions": len(qids),
        "per_arm": per_arm,
        "contrasts": contrasts,
        "agreement": agreement,
        "decision": decision,
        "expansion": expansion,
        "per_category_EXPLORATORY": per_category,
        "bootstrap": {"n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
    }

    if agreement.get("ungradeable"):
        print("  " + "*" * 68)
        print(f"  UNGRADEABLE: inter-grader Pearson r = {agreement.get('pearson_r')} "
              f"< {AGREEMENT_R_GATE}.")
        print("  Per prereg: STOP, revise rubric, amend.")
        print("  " + "*" * 68)

    Path(args.out).write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {args.out}")

    print_summary_table(aggregate)

    if agreement.get("ungradeable"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
