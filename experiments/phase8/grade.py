#!/usr/bin/env python3
"""Phase 8 grading — two independent graders (Opus, Sonnet) score contestant
(Haiku) answers against a reference key the grader writes itself first.

Per docs/phase8_prereg.md ("Grading" + guard #4/#7):
  - Two graders, both != contestant (haiku). model="opus" (G1), model="sonnet" (G2).
  - Per (question, grader):
      call 1 -> grader gets the question + expected_modules file text (truncated
                ~8k chars each) and writes a reference key.
      call 2 -> grader gets question + its own key + rubric + ALL answers for
                that question, blinded (random single-letter labels, order
                shuffled with a per-question seed derived from sha256(qid)),
                returns strict JSON: per-answer scores 1-5 on
                correctness/completeness/reference_accuracy.
  - Parse defensively; on 2nd parse failure retry once with a "JSON only"
    nudge; then record a null row and count it; abort if >10% null.
  - Works on calibration.jsonl (one answer/question, arm="bare") or
    contest_rows.jsonl (five answers/question, arms A/B/C/D256/D64).
  - Output: grades.jsonl rows {"qid","arm","grader","correctness",
    "completeness","reference_accuracy","overall"}.
  - Blind label mapping persisted separately: label_map.jsonl (audit trail).
  - Cache all calls in llm_cache/ by prompt hash; resume skips graded
    (qid, grader) pairs already present in grades.jsonl.
  - --mock: deterministic stub grades (hash-derived, varied), zero bridge calls.

Usage:
    python grade.py --input calibration.jsonl --out grades.jsonl --mock
    python grade.py --input contest_rows.jsonl --out grades.jsonl --mock

    # real (after --mock verification passes):
    python grade.py --input calibration.jsonl --out grades.jsonl
    python grade.py --input contest_rows.jsonl --out grades.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- sys.path bootstrap (copied from phase7/run_contest.py) ---------------
_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

log = logging.getLogger("phase8.grade")

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "llm_cache"
LABEL_MAP_PATH = HERE / "label_map.jsonl"

GRADERS = ("opus", "sonnet")  # model names via BridgeProvider; contestant is haiku.
CONTESTANT_MODEL = "haiku"

DIMENSIONS = ("correctness", "completeness", "reference_accuracy")

MODULE_CHAR_CAP = 8000  # truncate each expected_modules file to ~8k chars
NULL_RATE_ABORT_THRESHOLD = 0.10  # abort if >10% null rows

KEY_SYSTEM = """You are an expert reviewer building a grading reference key.

You will be given a question about a codebase and the text of the source
files that answer it. Write a concise, code-grounded reference key: the
correct answer, citing specific mechanisms/functions/files from the
provided material. This key will be used later to grade candidate answers.
Do not grade anything now — only produce the reference key.

Output the key as plain text (no JSON, no markdown fences needed)."""

RUBRIC = """Score each candidate answer 1-5 (integers) on three dimensions:

  correctness        — factually correct relative to the reference key?
                        1 = wrong/contradicts key, 5 = fully correct.
  completeness        — covers the material aspects of the key?
                        1 = misses everything, 5 = covers all key points.
  reference_accuracy  — cites/uses correct mechanisms, functions, files
                        (vs. vague or fabricated references)?
                        1 = fabricated/irrelevant references, 5 = precise
                        and accurate references.

Return STRICT JSON ONLY, no markdown fences, no commentary, in this exact
shape (one entry per labeled answer you were given):

{"scores": {"<LABEL>": {"correctness": <int>, "completeness": <int>, "reference_accuracy": <int>}, ...}}
"""

JSON_NUDGE = (
    "Your previous response could not be parsed as JSON. Reply with JSON "
    "ONLY — no markdown fences, no prose before or after — matching exactly "
    "this shape: "
    '{"scores": {"<LABEL>": {"correctness": <int>, "completeness": <int>, '
    '"reference_accuracy": <int>}, ...}}'
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _prompt_hash(system: str, user: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()


def cache_get(key: str) -> Optional[str]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["text"]
    except Exception:
        return None


def cache_put(key: str, text: str, meta: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps({"text": text, **meta}, ensure_ascii=False, indent=2),
                 encoding="utf-8")


# ---------------------------------------------------------------------------
# Blinding (guard #7: deterministic per-question seed from sha256(qid))
# ---------------------------------------------------------------------------


def blind_seed(qid: str) -> int:
    """Deterministic seed derived from sha256(qid); stable across runs/machines."""
    h = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def build_blind_labels(qid: str, arm_names: List[str]) -> Dict[str, str]:
    """arm_name -> single-letter label, order shuffled deterministically.

    Uses a local Random instance seeded from sha256(qid) so this never
    perturbs global random state (important since --mock also uses random
    elsewhere) and is fully reproducible.
    """
    rng = random.Random(blind_seed(qid))
    letters = list(string.ascii_uppercase[: len(arm_names)])
    shuffled_arms = list(arm_names)
    rng.shuffle(shuffled_arms)
    rng.shuffle(letters)
    return {arm: label for arm, label in zip(shuffled_arms, letters)}


# ---------------------------------------------------------------------------
# Expected-modules file loading
# ---------------------------------------------------------------------------


def load_expected_modules_text(expected_modules: List[str]) -> str:
    """Read each expected_modules path (relative to C:\\code\\autonet), truncate
    to ~8k chars each, concatenate with headers."""
    parts: List[str] = []
    for rel in expected_modules:
        full = _AUTONET / rel
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            parts.append(f"=== {rel} ===\n[could not read: {type(e).__name__}: {e}]\n")
            continue
        truncated = text[:MODULE_CHAR_CAP]
        if len(text) > MODULE_CHAR_CAP:
            truncated += f"\n... [truncated, {len(text)} chars total]"
        parts.append(f"=== {rel} ===\n{truncated}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Defensive JSON parsing
# ---------------------------------------------------------------------------


def parse_scores_json(text: str) -> Optional[Dict[str, Dict[str, int]]]:
    """Defensively extract {"scores": {...}} from grader output."""
    if not text:
        return None
    candidates = [text.strip()]
    # Strip markdown fences if present.
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidates.insert(0, m.group(1).strip())
    # Try to find the outermost {...} block.
    m2 = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m2:
        candidates.append(m2.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        scores = obj.get("scores") if isinstance(obj, dict) else None
        if not isinstance(scores, dict):
            continue
        out: Dict[str, Dict[str, int]] = {}
        ok = True
        for label, dims in scores.items():
            if not isinstance(dims, dict):
                ok = False
                break
            row: Dict[str, int] = {}
            for d in DIMENSIONS:
                v = dims.get(d)
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not (1 <= iv <= 5):
                    ok = False
                    break
                row[d] = iv
            if not ok:
                break
            out[label] = row
        if ok and out:
            return out
    return None


# ---------------------------------------------------------------------------
# Row loading — works on either calibration.jsonl or contest_rows.jsonl
# ---------------------------------------------------------------------------


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def row_answers(row: Dict[str, Any]) -> Dict[str, str]:
    """Return {arm_name: answer_text} regardless of schema.

    calibration.jsonl: {"qid","arm":"bare","prompt","answer"}  -> one arm.
    contest_rows.jsonl: {"qid","arms":{"A":{"prompt","answer",...}, ...}} -> five arms.
    """
    if "arms" in row:
        return {arm: data.get("answer", "") for arm, data in row["arms"].items()}
    arm = row.get("arm", "bare")
    return {arm: row.get("answer", "")}


def row_qid(row: Dict[str, Any]) -> str:
    return row["qid"]


# ---------------------------------------------------------------------------
# Mock grading (deterministic, hash-derived, varied — no bridge calls)
# ---------------------------------------------------------------------------


def mock_score(qid: str, grader: str, arm: str, dim: str) -> int:
    h = hashlib.sha256(f"{qid}|{grader}|{arm}|{dim}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16)
    return 1 + (v % 5)  # 1..5


# ---------------------------------------------------------------------------
# Grading core
# ---------------------------------------------------------------------------


async def get_reference_key(
    provider, mock: bool, qid: str, grader: str, question: str,
    expected_modules: List[str], mock_cache_note: Dict[str, Any],
) -> str:
    module_text = load_expected_modules_text(expected_modules)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCE MATERIAL:\n{module_text}\n\n"
        "Write the reference key now."
    )
    key = _prompt_hash(KEY_SYSTEM, user, grader)
    cached = cache_get(key)
    if cached is not None:
        return cached

    if mock:
        text = f"[MOCK KEY qid={qid} grader={grader}] " + hashlib.sha256(
            f"key|{qid}|{grader}".encode()).hexdigest()[:32]
    else:
        assert provider is not None
        result = await provider.send(
            messages=[{"role": "user", "content": user}],
            system=KEY_SYSTEM,
            model=grader,
            max_tokens=2000,
        )
        text = result.text or ""

    cache_put(key, text, {"kind": "reference_key", "qid": qid, "grader": grader})
    return text


def build_grading_prompt(
    question: str, key: str, labeled_answers: Dict[str, str],
) -> str:
    lines = [
        f"QUESTION:\n{question}\n",
        f"YOUR REFERENCE KEY:\n{key}\n",
        RUBRIC,
        "CANDIDATE ANSWERS (blinded, order randomized):\n",
    ]
    for label in sorted(labeled_answers.keys()):
        lines.append(f"--- Answer [{label}] ---")
        lines.append(labeled_answers[label])
        lines.append("")
    return "\n".join(lines)


async def call_grader_json(
    provider, mock: bool, grader: str, qid: str, prompt: str,
    labels: List[str], nudge: bool = False,
) -> Tuple[Optional[Dict[str, Dict[str, int]]], str]:
    """Returns (parsed_scores_or_None, raw_text)."""
    effective_prompt = prompt if not nudge else prompt + "\n\n" + JSON_NUDGE
    key = _prompt_hash(RUBRIC, effective_prompt, grader) + ("-nudge" if nudge else "")
    cached = cache_get(key)
    if cached is not None:
        return parse_scores_json(cached), cached

    if mock:
        scores = {
            label: {d: mock_score(qid, grader, label, d) for d in DIMENSIONS}
            for label in labels
        }
        text = json.dumps({"scores": scores})
    else:
        assert provider is not None
        result = await provider.send(
            messages=[{"role": "user", "content": effective_prompt}],
            system=RUBRIC,
            model=grader,
            max_tokens=2000,
        )
        text = result.text or ""

    cache_put(key, text, {"kind": "grading", "qid": qid, "grader": grader, "nudge": nudge})
    return parse_scores_json(text), text


async def grade_question(
    provider, mock: bool, grader: str, row: Dict[str, Any],
    questions_by_qid: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Grade one (question, grader) pair across all its arms.

    Returns (grade_rows, label_map_rows, had_null).
    """
    qid = row_qid(row)
    qmeta = questions_by_qid.get(qid, {})
    question_text = qmeta.get("question", row.get("question", ""))
    expected_modules = qmeta.get("expected_modules", row.get("expected_modules", []))

    answers = row_answers(row)
    arm_names = list(answers.keys())

    key = await get_reference_key(
        provider, mock, qid, grader, question_text, expected_modules, {})

    label_of_arm = build_blind_labels(qid, arm_names)
    arm_of_label = {v: k for k, v in label_of_arm.items()}
    labeled_answers = {label_of_arm[arm]: answers[arm] for arm in arm_names}
    labels = list(labeled_answers.keys())

    prompt = build_grading_prompt(question_text, key, labeled_answers)

    parsed, raw1 = await call_grader_json(provider, mock, grader, qid, prompt, labels)
    attempt = 1
    if parsed is None:
        parsed, raw2 = await call_grader_json(
            provider, mock, grader, qid, prompt, labels, nudge=True)
        attempt = 2

    label_map_rows = [{
        "qid": qid, "grader": grader, "arm": arm, "label": label_of_arm[arm],
        "seed": blind_seed(qid),
    } for arm in arm_names]

    if parsed is None:
        # Null row(s): record one null grade per arm for this (qid, grader).
        null_rows = [{
            "qid": qid, "arm": arm, "grader": grader,
            "correctness": None, "completeness": None,
            "reference_accuracy": None, "overall": None,
            "null": True, "parse_attempts": attempt,
        } for arm in arm_names]
        return null_rows, label_map_rows, True

    grade_rows = []
    for label, arm in arm_of_label.items():
        dims = parsed.get(label)
        if dims is None:
            grade_rows.append({
                "qid": qid, "arm": arm, "grader": grader,
                "correctness": None, "completeness": None,
                "reference_accuracy": None, "overall": None,
                "null": True, "parse_attempts": attempt,
                "note": "label missing from grader JSON",
            })
            continue
        overall = sum(dims[d] for d in DIMENSIONS) / len(DIMENSIONS)
        grade_rows.append({
            "qid": qid, "arm": arm, "grader": grader,
            "correctness": dims["correctness"],
            "completeness": dims["completeness"],
            "reference_accuracy": dims["reference_accuracy"],
            "overall": overall,
            "null": False, "parse_attempts": attempt,
        })
    had_null = any(r["null"] for r in grade_rows)
    return grade_rows, label_map_rows, had_null


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_existing_grades(out_path: Path) -> Tuple[List[Dict[str, Any]], set]:
    if not out_path.exists():
        return [], set()
    rows = load_rows(out_path)
    done_pairs = set()
    # A (qid, grader) pair counts as "done" only if every arm-row for it is present
    # and non-null-due-to-crash. We conservatively treat presence of ANY row for
    # (qid, grader) as done — resume skips already-graded pairs by design; a
    # partial write is prevented by writing per-question-per-grader atomically
    # (all arm rows appended together at once, see main loop).
    from collections import defaultdict
    by_pair: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in rows:
        by_pair[(r["qid"], r["grader"])] += 1
    for pair in by_pair:
        done_pairs.add(pair)
    return rows, done_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="calibration.jsonl or contest_rows.jsonl")
    parser.add_argument("--questions", default=None,
                        help="questions.jsonl (for expected_modules); "
                             "defaults to <phase8>/questions.jsonl")
    parser.add_argument("--out", required=True, help="grades.jsonl output path")
    parser.add_argument("--mock", action="store_true",
                        help="deterministic offline stub; zero bridge calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    input_path = Path(args.input)
    out_path = Path(args.out)
    questions_path = Path(args.questions) if args.questions else HERE / "questions.jsonl"

    rows = load_rows(input_path)
    questions_by_qid: Dict[str, Dict[str, Any]] = {}
    if questions_path.exists():
        for q in load_rows(questions_path):
            questions_by_qid[q["qid"]] = q
    else:
        log.warning("questions.jsonl not found at %s; expected_modules will be "
                    "read from input rows if present, else empty", questions_path)

    print(f"  input={input_path.name}  rows={len(rows)}  mock={args.mock}")

    existing_rows, done_pairs = load_existing_grades(out_path)
    if done_pairs:
        print(f"  resume: {len(done_pairs)} (qid,grader) pairs already graded, skipping")

    provider = None
    providers_by_model: Dict[str, Any] = {}
    if not args.mock:
        from atn.providers.bridge import BridgeProvider  # type: ignore
        for g in GRADERS:
            assert g != CONTESTANT_MODEL, "grader must not equal contestant model"
            providers_by_model[g] = BridgeProvider(model=g)

    all_grade_rows: List[Dict[str, Any]] = list(existing_rows)
    all_label_rows: List[Dict[str, Any]] = []
    if LABEL_MAP_PATH.exists():
        all_label_rows = load_rows(LABEL_MAP_PATH)

    total_arm_rows = 0
    null_arm_rows = 0
    started = time.time()

    try:
        out_f = out_path.open("a", encoding="utf-8")
        label_f = LABEL_MAP_PATH.open("a", encoding="utf-8")
        try:
            for i, row in enumerate(rows, start=1):
                qid = row_qid(row)
                for grader in GRADERS:
                    if (qid, grader) in done_pairs:
                        continue
                    prov = None if args.mock else providers_by_model[grader]
                    grade_rows, label_rows, had_null = await grade_question(
                        prov, args.mock, grader, row, questions_by_qid)

                    for gr in grade_rows:
                        out_f.write(json.dumps(gr, ensure_ascii=False) + "\n")
                    out_f.flush()
                    for lr in label_rows:
                        label_f.write(json.dumps(lr, ensure_ascii=False) + "\n")
                    label_f.flush()

                    total_arm_rows += len(grade_rows)
                    null_arm_rows += sum(1 for r in grade_rows if r["null"])
                    done_pairs.add((qid, grader))

                    if total_arm_rows > 0:
                        rate = null_arm_rows / total_arm_rows
                        if rate > NULL_RATE_ABORT_THRESHOLD and total_arm_rows >= 10:
                            print(f"\n  ABORT: null rate {rate:.1%} exceeds "
                                  f"{NULL_RATE_ABORT_THRESHOLD:.0%} threshold "
                                  f"({null_arm_rows}/{total_arm_rows})")
                            return 1

                elapsed = time.time() - started
                if i % 5 == 0 or i == len(rows):
                    print(f"  [{i:>3}/{len(rows)}] qid={qid}  "
                          f"total_rows={total_arm_rows} nulls={null_arm_rows}  "
                          f"({elapsed:.0f}s)")
        finally:
            out_f.close()
            label_f.close()
    finally:
        for prov in providers_by_model.values():
            try:
                await prov.close()
            except Exception:
                pass

    null_rate = (null_arm_rows / total_arm_rows) if total_arm_rows else 0.0
    print(f"\n  done: {total_arm_rows} grade rows written, {null_arm_rows} null "
          f"({null_rate:.1%})")
    if null_rate > NULL_RATE_ABORT_THRESHOLD:
        print(f"  ABORT: final null rate {null_rate:.1%} exceeds "
              f"{NULL_RATE_ABORT_THRESHOLD:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
