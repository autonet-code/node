#!/usr/bin/env python3
"""Phase 8 question selection — guard #5 + calibration gate #3.

Reads calibration grades (bare Haiku, both graders), computes the mean bare
score per question (mean of both graders' `overall`), selects the 25 lowest
(ties broken by sha256(qid) ascending), writes selected_questions.jsonl
(question rows + "bare_calibration_score").

Enforces gate #3 (docs/phase8_prereg.md guard #3 / Analysis "Decision rule"
precondition): if the selected set's mean bare score >= 2.5, print the STOP
message and exit nonzero — the contest must not proceed on a saturated
domain (this is exactly the phase 7 flaw the gate exists to prevent).

Usage:
    python select_questions.py --grades grades.jsonl --questions questions.jsonl \
        --out selected_questions.jsonl --n 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent

CALIBRATION_GATE_THRESHOLD = 2.5  # prereg guard #3


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def qid_sort_key(qid: str):
    return hashlib.sha256(qid.encode("utf-8")).hexdigest()


def mean_bare_scores(grade_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """qid -> {"mean": float, "n_graders": int, "graders": [overall,...]}

    Only considers arm == "bare" rows (calibration.jsonl has one arm per row);
    non-null rows only. A question with no valid grades is excluded (and
    reported) rather than silently scored as 0.
    """
    by_qid: Dict[str, List[float]] = {}
    for r in grade_rows:
        if r.get("arm") != "bare":
            continue
        if r.get("null"):
            continue
        overall = r.get("overall")
        if overall is None:
            continue
        by_qid.setdefault(r["qid"], []).append(float(overall))

    out: Dict[str, Dict[str, Any]] = {}
    for qid, scores in by_qid.items():
        out[qid] = {
            "mean": sum(scores) / len(scores),
            "n_graders": len(scores),
            "graders_overall": scores,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades", required=True, help="calibration grades.jsonl")
    parser.add_argument("--questions", required=True, help="questions.jsonl (40 questions)")
    parser.add_argument("--out", required=True, help="selected_questions.jsonl output")
    parser.add_argument("--n", type=int, default=25, help="number of questions to select")
    args = parser.parse_args()

    grades_path = Path(args.grades)
    questions_path = Path(args.questions)
    out_path = Path(args.out)

    grade_rows = load_jsonl(grades_path)
    questions = load_jsonl(questions_path)
    questions_by_qid = {q["qid"]: q for q in questions}

    scores = mean_bare_scores(grade_rows)

    missing = [q["qid"] for q in questions if q["qid"] not in scores]
    if missing:
        print(f"  WARNING: {len(missing)} question(s) have no valid bare-calibration "
              f"grade and are excluded from selection: {missing}")

    scored_qids = [qid for qid in questions_by_qid if qid in scores]
    if len(scored_qids) < args.n:
        print(f"  ABORT: only {len(scored_qids)} scored questions available, "
              f"need {args.n}")
        return 1

    # Sort by (mean score ascending, sha256(qid) ascending) for deterministic
    # tie-break per prereg.
    scored_qids.sort(key=lambda qid: (scores[qid]["mean"], qid_sort_key(qid)))
    selected_qids = scored_qids[: args.n]

    selected_rows = []
    for qid in selected_qids:
        q = dict(questions_by_qid[qid])
        q["bare_calibration_score"] = scores[qid]["mean"]
        selected_rows.append(q)

    with out_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_mean = sum(r["bare_calibration_score"] for r in selected_rows) / len(selected_rows)

    print(f"  scored questions: {len(scored_qids)}/{len(questions)}")
    print(f"  selected: {len(selected_rows)} lowest-scoring (of {len(scored_qids)})")
    print(f"  selected set mean bare score: {selected_mean:.4f}")
    print(f"  wrote {out_path}")

    if selected_mean >= CALIBRATION_GATE_THRESHOLD:
        print()
        print("  " + "*" * 68)
        print("  STOP: calibration gate #3 failed.")
        print(f"  Selected question set mean bare score = {selected_mean:.4f} "
              f">= {CALIBRATION_GATE_THRESHOLD} threshold.")
        print("  The bare model does not demonstrably fail on this question set")
        print("  (saturated domain — the exact phase 7 flaw this gate exists to")
        print("  prevent). Contest must NOT proceed. Revise question generation")
        print("  or corpus and re-run calibration.")
        print("  " + "*" * 68)
        return 1

    print(f"  gate #3 PASSED: {selected_mean:.4f} < {CALIBRATION_GATE_THRESHOLD}. "
          f"Contest may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
