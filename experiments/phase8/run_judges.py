#!/usr/bin/env python3
"""Phase 8 judge runner.

For each sampled work unit (problem + resolution), ONE Sonnet call
produces 2-3 structured sub-claims. Each claim:

    {"axis": "correctness"|"simplicity"|"robustness",
     "position": "pro"|"con",
     "text": str}

Per prereg (docs/phase8_prereg.md, Corpus section): honest CONs are
REQUIRED where the resolution has real flaws; the judge must NOT emit
all-PRO. The SAME cache feeds arms C and D so verdict content is
identical across them.

Caching / crash-safety:
  - Every raw call cached to llm_cache/<sha256(prompt)>.json so reruns
    are free (both the prompt-keyed raw cache AND the per-unit
    judge_cache.jsonl are checked; a uid already in judge_cache.jsonl is
    skipped for crash-safe resume).
  - judges_progress.jsonl gets one row per unit as it completes.
  - Parse defensively; if a call fails to parse twice, record
    {"uid", "claims": []} and count it. Abort loudly if >10% of units
    end up with empty claims.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "corpus_sample.jsonl"
CACHE_PATH = HERE / "judge_cache.jsonl"
PROGRESS_PATH = HERE / "judges_progress.jsonl"
LLM_CACHE_DIR = HERE / "llm_cache"

AXES = ("correctness", "simplicity", "robustness")

JUDGE_SYSTEM = """You are a rigorous senior engineer extracting structured verdict sub-claims about a completed unit of software work, for use in a debate/standing substrate.

You are given a WORK UNIT: the PROBLEM the developer faced and the RESOLUTION that was produced (a distilled transcript of what was investigated, decided, and done).

Produce 2 or 3 sub-claims total (not per axis — 2 or 3 in all). Each sub-claim is a single factual, specific sentence a careful reader of this unit would defend. Spread them across these axes where each is relevant:

  correctness: whether the resolution actually solves the stated problem; whether its technical reasoning holds; whether it addresses the real root cause or a symptom.
  simplicity: whether the approach is as simple as the problem allows; over-engineering, unnecessary steps, or conversely commendable minimalism.
  robustness: whether the resolution holds up under edge cases, failure modes, concurrency, platform differences, or future change; what it leaves unhandled.

Each sub-claim takes a POSITION:
  "pro"  = a claim asserting a genuine STRENGTH of the resolution on that axis.
  "con"  = a claim asserting a genuine WEAKNESS, gap, risk, or flaw of the resolution on that axis.

CRITICAL HONESTY REQUIREMENT: Real resolutions have real flaws. You MUST emit a "con" claim whenever the resolution has any genuine weakness, unhandled case, unproven assumption, or shortcut — which is true of most real work. Do NOT default to all-"pro". An all-"pro" verdict on a non-trivial unit is a failure of your job. If the resolution is genuinely strong on one axis, say so with "pro" — but look hard for the weaknesses.

Respond with ONLY this JSON (no markdown fence, no preamble, no trailing prose):
{"claims": [{"axis": "correctness"|"simplicity"|"robustness", "position": "pro"|"con", "text": "one specific factual sentence"}, ...]}"""


def judge_user_prompt(unit: Dict[str, Any]) -> str:
    outcome = unit.get("outcome") or {}
    outcome_line = (
        f"Recorded outcome signals: {json.dumps(outcome)}\n\n"
        if outcome
        else ""
    )
    return (
        "WORK UNIT\n\n"
        f"PROBLEM:\n{unit['problem']}\n\n"
        f"RESOLUTION:\n{unit['resolution']}\n\n"
        f"{outcome_line}"
        "Extract 2-3 verdict sub-claims now, following the honesty "
        "requirement (emit con claims for real weaknesses)."
    )


def prompt_hash(system: str, user: str, model: str) -> str:
    payload = json.dumps(
        {"system": system, "user": user, "model": model}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def llm_cache_path(h: str) -> Path:
    return LLM_CACHE_DIR / f"{h}.json"


def load_done_uids() -> set:
    if not CACHE_PATH.exists():
        return set()
    done = set()
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["uid"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


_CLAIM_OBJ_RE = re.compile(
    r'\{\s*"axis"\s*:\s*"(correctness|simplicity|robustness)"\s*,\s*'
    r'"position"\s*:\s*"(pro|con)"\s*,\s*'
    r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
    re.DOTALL,
)


def salvage_claims(text: str) -> Optional[List[Dict[str, str]]]:
    """Recover COMPLETE claim objects from truncated/malformed JSON.

    The observed failure mode is a response cut off mid-string: the
    leading claim objects are complete and valid, the last one is
    truncated. Extract every complete object; drop the broken tail.
    """
    out: List[Dict[str, str]] = []
    for m in _CLAIM_OBJ_RE.finditer(text):
        axis, position, raw_text = m.group(1), m.group(2), m.group(3)
        try:
            ctext = json.loads(f'"{raw_text}"').strip()
        except json.JSONDecodeError:
            continue
        if ctext:
            out.append({"axis": axis, "position": position, "text": ctext})
    return out[:3] if out else None


def parse_claims(text: str) -> Optional[List[Dict[str, str]]]:
    """Defensive parse -> list of validated claim dicts, or None on failure."""
    if not text:
        return None
    candidate = text.strip()
    # strip a markdown fence if present
    if candidate.startswith("```"):
        nl = candidate.find("\n")
        if nl != -1:
            candidate = candidate[nl + 1 :]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
        candidate = candidate.strip()
    parsed = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first >= 0 and last > first:
            try:
                parsed = json.loads(candidate[first : last + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if isinstance(parsed, list):
        raw_claims = parsed
    elif isinstance(parsed, dict):
        raw_claims = parsed.get("claims")
    else:
        return None
    if not isinstance(raw_claims, list):
        return None

    out: List[Dict[str, str]] = []
    for c in raw_claims:
        if not isinstance(c, dict):
            continue
        axis = str(c.get("axis", "")).strip().lower()
        position = str(c.get("position", "")).strip().lower()
        ctext = str(c.get("text", "")).strip()
        if axis not in AXES:
            continue
        if position not in ("pro", "con"):
            continue
        if not ctext:
            continue
        out.append({"axis": axis, "position": position, "text": ctext})
    # A valid parse must yield at least one well-formed claim.
    if not out:
        return None
    # cap at 3 to honor the 2-3 spec
    return out[:3]


async def call_judge(
    provider: BridgeProvider, system: str, user: str, model: str
) -> tuple[str, bool]:
    """Return (raw_text, from_cache)."""
    h = prompt_hash(system, user, model)
    cp = llm_cache_path(h)
    if cp.exists():
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
            return cached.get("text", ""), True
        except (json.JSONDecodeError, OSError):
            pass  # fall through to live call
    result = await provider.send(
        messages=[{"role": "user", "content": user}],
        system=system,
        model=model,
    )
    text = result.text or ""
    cp.write_text(
        json.dumps(
            {
                "prompt_hash": h,
                "system": system,
                "user": user,
                "model": model,
                "text": text,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                },
                "ts": time.time(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return text, False


async def main() -> int:
    LLM_CACHE_DIR.mkdir(exist_ok=True)
    if not CORPUS_PATH.exists():
        raise SystemExit(f"corpus not found: {CORPUS_PATH}; run build_corpus.py")

    units = [
        json.loads(l)
        for l in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"corpus units: {len(units)}")

    done = load_done_uids()
    if done:
        print(f"resuming: {len(done)} units already in judge_cache.jsonl")

    provider = BridgeProvider(model="sonnet")
    model = "sonnet"

    live_calls = 0
    cache_hits = 0
    empty_units = 0
    total_claims = 0
    pro_count = 0
    con_count = 0
    processed = 0

    try:
        for i, unit in enumerate(units, start=1):
            uid = unit["uid"]
            if uid in done:
                continue
            user = judge_user_prompt(unit)

            claims: Optional[List[Dict[str, str]]] = None
            raw_last = ""
            attempt_texts: List[str] = []
            # Two attempts: first uses the prompt-keyed cache; the retry
            # forces a fresh live call by nudging the prompt so its hash
            # differs (a bad cached parse won't wedge the retry).
            for attempt in range(2):
                if attempt == 0:
                    text, from_cache = await call_judge(
                        provider, JUDGE_SYSTEM, user, model
                    )
                else:
                    retry_user = (
                        user
                        + "\n\n[RETRY] Your previous output did not parse. "
                        "Respond with ONLY the JSON object described, nothing else."
                    )
                    text, from_cache = await call_judge(
                        provider, JUDGE_SYSTEM, retry_user, model
                    )
                if from_cache:
                    cache_hits += 1
                else:
                    live_calls += 1
                raw_last = text
                attempt_texts.append(text)
                claims = parse_claims(text)
                if claims is not None:
                    break

            if claims is None:
                # Salvage pass: recover complete claim objects from a
                # truncated response (observed failure mode: JSON cut
                # off mid-string). Tries each attempt's cached raw text.
                for t in attempt_texts:
                    claims = salvage_claims(t)
                    if claims is not None:
                        print(
                            f"  [{i:>3}/{len(units)}] {uid[:12]}  "
                            f"salvaged {len(claims)} claims from truncated JSON"
                        )
                        break

            if claims is None:
                claims = []
                empty_units += 1
                print(
                    f"  [{i:>3}/{len(units)}] {uid[:12]}  PARSE FAIL x2 "
                    f"(raw[:100]={raw_last[:100]!r})"
                )

            n_pro = sum(1 for c in claims if c["position"] == "pro")
            n_con = sum(1 for c in claims if c["position"] == "con")
            pro_count += n_pro
            con_count += n_con
            total_claims += len(claims)
            processed += 1

            cache_row = {"uid": uid, "claims": claims}
            append_jsonl(CACHE_PATH, cache_row)
            append_jsonl(
                PROGRESS_PATH,
                {
                    "uid": uid,
                    "n_claims": len(claims),
                    "pro": n_pro,
                    "con": n_con,
                    "ts": time.time(),
                },
            )
            done.add(uid)

            print(
                f"  [{i:>3}/{len(units)}] {uid[:12]}  "
                f"claims={len(claims)} pro={n_pro} con={n_con}"
            )
    finally:
        try:
            await provider.close()
        except Exception:
            pass

    total_units = len(units)
    empty_frac = empty_units / max(total_units, 1)
    print()
    print("=" * 60)
    print("JUDGES DONE")
    print(f"  units processed this run: {processed}")
    print(f"  live calls: {live_calls}, cache hits: {cache_hits}")
    print(f"  total claims (this run): {total_claims}")
    if processed:
        print(f"  mean claims/unit (this run): {total_claims/processed:.2f}")
    print(f"  PRO: {pro_count}  CON: {con_count}")
    if (pro_count + con_count):
        print(f"  PRO:CON ratio: {pro_count}:{con_count} "
              f"(CON = {con_count/(pro_count+con_count):.1%} of claims)")
    print(f"  empty units (parse failed x2): {empty_units} ({empty_frac:.1%})")

    if empty_frac > 0.10:
        print()
        print(f"  ABORT: {empty_frac:.1%} of units empty (> 10% threshold).")
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
