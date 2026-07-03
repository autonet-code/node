#!/usr/bin/env python3
"""Phase 8 question generator.

Generates 40 questions (8 each: mechanism, architectural, code-reference,
tradeoff, gotcha) via Opus (model="opus" through the bridge), with access
to:
  - a digest of the sampled corpus (problems + resolutions), and
  - the autonet substrate code under
      nodes/common/world_model_substrate/  and
      world_model/generalized/
    (relevant excerpts included inline per category).

Prereg constraint (docs/phase8_prereg.md, Questions section): each
question must be answerable from corpus/code content. The generation
prompt instructs the model accordingly, and each question carries
`expected_modules` (paths) grounding it.

One Opus call PER category (5 calls), each asked for exactly 8 questions.
Every raw call is cached to llm_cache/<sha256(prompt)>.json keyed by the
prompt, so reruns are free. Output rows:
  {"qid": "q001", "category", "question", "expected_modules": [paths]}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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
OUT_PATH = HERE / "questions.jsonl"
LLM_CACHE_DIR = HERE / "llm_cache"

SUBSTRATE_DIR = _AUTONET / "nodes" / "common" / "world_model_substrate"
GENERALIZED_DIR = _AUTONET / "world_model" / "generalized"

CATEGORIES = ["mechanism", "architectural", "code-reference", "tradeoff", "gotcha"]
PER_CATEGORY = 8

# Files whose head/full text we make available to the generator, grouped
# by which categories they most inform. rel path -> line budget.
CODE_FILES = {
    "nodes/common/world_model_substrate/reconcile.py": 411,
    "nodes/common/world_model_substrate/mint_gate.py": 232,
    "nodes/common/world_model_substrate/adapter.py": 180,
    "nodes/common/world_model_substrate/infer.py": 120,
    "nodes/common/world_model_substrate/events.py": 90,
    "nodes/common/world_model_substrate/aggregate.py": 90,
    "nodes/common/world_model_substrate/artifact_index.py": 90,
    "nodes/common/world_model_substrate/usefulness_coords.py": 90,
    "world_model/generalized/equilibrate.py": 240,
    "world_model/generalized/coordinate_frame.py": 90,
    "world_model/generalized/locate.py": 100,
    "world_model/generalized/decay.py": 40,
    "world_model/generalized/prune.py": 40,
    "world_model/generalized/world.py": 40,
    "world_model/generalized/render.py": 40,
    "world_model/generalized/tendency.py": 60,
}

# Which files to surface per category (all categories get the core set;
# code-reference gets the broadest spread).
ALL_MODULES = list(CODE_FILES.keys())


def read_excerpt(rel: str, n_lines: int) -> str:
    p = _AUTONET / rel
    if not p.exists():
        return f"[missing: {rel}]"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(lines[:n_lines])
    return f"===== FILE: {rel} (first {n_lines} lines) =====\n{body}\n"


def build_code_context() -> str:
    parts = [read_excerpt(rel, n) for rel, n in CODE_FILES.items()]
    return "\n\n".join(parts)


def load_units() -> List[Dict[str, Any]]:
    return [
        json.loads(l)
        for l in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def build_corpus_digest(units: List[Dict[str, Any]], chunk_idx: int,
                        n_chunks: int) -> str:
    """Corpus digest, chunked across calls (prereg: 'chunked across
    calls as needed').

    Every call receives:
      - a compact INDEX of ALL units (uid prefix + one-line problem),
        so the generator sees the whole corpus's shape; and
      - a DETAILED chunk (fuller problem + resolution) of
        len(units)/n_chunks units, rotating by chunk_idx, so across
        the n_chunks category calls every unit appears in detail once.
    """
    n = len(units)
    lines = [
        f"CORPUS: {n} real Claude Code work units (autonet + world-model "
        f"repos).\n\n=== INDEX OF ALL {n} UNITS (one line each) ==="
    ]
    for i, u in enumerate(units):
        prob = " ".join(str(u["problem"]).split())[:110]
        lines.append(f"[{i:03d}|{u['uid'][:8]}] {prob}")

    per = max(1, (n + n_chunks - 1) // n_chunks)
    lo, hi = chunk_idx * per, min(n, (chunk_idx + 1) * per)
    lines.append(
        f"\n=== DETAILED UNITS {lo}..{hi - 1} (this call's chunk; other "
        f"chunks go to other category calls) ==="
    )
    for i in range(lo, hi):
        u = units[i]
        prob = " ".join(str(u["problem"]).split())[:350]
        res = " ".join(str(u["resolution"]).split())[:600]
        out = u.get("outcome") or {}
        lines.append(
            f"--- unit {i:03d} (uid {u['uid'][:8]}) ---\n"
            f"PROBLEM: {prob}\n"
            f"RESOLUTION: {res}\n"
            f"OUTCOME: {json.dumps(out)}"
        )
    return "\n".join(lines)


SYSTEM = """You are setting exam questions for a graduate-level assessment of understanding of the autonet substrate engine (a decentralized-training world-model system) and the real engineering work done in its repo.

You are given (1) the actual substrate SOURCE CODE excerpts and (2) a DIGEST of real work units (problem + resolution) from the repo's session traces.

HARD CONSTRAINT: every question you write MUST be answerable purely from the code excerpts and/or the corpus digest provided — no outside knowledge required, no speculation. A competent reader with ONLY these materials must be able to produce a correct, specific answer. Do NOT ask about anything not present in the supplied material. Do NOT ask vague opinion questions.

Write questions that discriminate real understanding: they should be specific, reference concrete mechanisms/functions/decisions, and have a definite correct answer grounded in the material. Avoid trivia; target the load-bearing ideas.

For the category you are given, produce EXACTLY the requested number of questions. Each question must name the specific code file(s) or corpus content it is answerable from.

Respond with ONLY a JSON array (no markdown fence, no preamble):
[{"question": "the full question text", "expected_modules": ["relative/path/to/file.py", ...]}, ...]

expected_modules must be relative paths drawn from the file list I give you (e.g. "nodes/common/world_model_substrate/reconcile.py"); include the corpus by using the literal token "corpus_sample.jsonl" when a question is grounded in the work-unit digest rather than code."""


CATEGORY_GUIDANCE = {
    "mechanism": (
        "MECHANISM questions: 'how does X actually work?' — the step-by-step "
        "operation of a specific algorithm or formula in the code. E.g. how mint "
        "is computed from score movement, how the survival factor is derived, how "
        "equilibration decides convergence, how the charter violation score is "
        "computed, how the Lindblad continuous kernel maps score<->population. "
        "The answer should trace concrete code."
    ),
    "architectural": (
        "ARCHITECTURAL questions: 'why is the system structured this way?' — the "
        "shape and separation of concerns. E.g. why the mint gate is a post-mint "
        "gate rather than a coordinate axis or a hard precondition; why the "
        "artifact index is daemon-local derived state and not consensus state; why "
        "the two-plane split (verdict graph vs payload retrieval); why locate is a "
        "single swappable primitive with three consumers; why emission-pool "
        "normalization flips the economics. Answer grounded in code/docstrings."
    ),
    "code-reference": (
        "CODE-REFERENCE questions: pin down a specific fact in the code — a "
        "function's exact return shape, a specific default/threshold/constant, "
        "which function calls which, the exact keys in a returned dict, the value "
        "of a default charter id list, the condition under which a branch is taken. "
        "The answer is a precise, checkable citation from the source."
    ),
    "tradeoff": (
        "TRADEOFF questions: a design or engineering decision with competing "
        "options, where the material states or implies why one was chosen. Draw on "
        "BOTH the substrate code's documented tradeoffs (e.g. uncapped mint vs "
        "fixed emission pool; veto pruning on/off; write-back suppressed in "
        "exploration mode) AND the real work units (e.g. OS file-locking vs "
        "port-locking for the daemon singleton, modularizing for a releasable "
        "binary). The answer must state the tradeoff and the resolution taken."
    ),
    "gotcha": (
        "GOTCHA questions: a non-obvious pitfall, invariant, or subtle bug the "
        "code explicitly guards against. E.g. the visited-set-gates-requeuing "
        "guard against cycles in co-parented graphs (MemoryError); why "
        "emission_pool must be derived from anchored timestamps not a local clock; "
        "why *_cid fields are sha256 not IPFS CIDs; why the mint-gate agent "
        "reweighting uses a global ratio approximation; ROOTs being sacred in "
        "pruning. The answer names the trap and why it exists."
    ),
}


def prompt_hash(system: str, user: str, model: str) -> str:
    payload = json.dumps(
        {"system": system, "user": user, "model": model}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def call_opus(
    provider: BridgeProvider, system: str, user: str, model: str
) -> tuple[str, bool]:
    h = prompt_hash(system, user, model)
    cp = LLM_CACHE_DIR / f"{h}.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8")).get("text", ""), True
        except (json.JSONDecodeError, OSError):
            pass
    result = await provider.send(
        messages=[{"role": "user", "content": user}], system=system, model=model
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


def parse_questions(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        nl = candidate.find("\n")
        if nl != -1:
            candidate = candidate[nl + 1 :]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
        candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        first = candidate.find("[")
        last = candidate.rfind("]")
        if first >= 0 and last > first:
            try:
                parsed = json.loads(candidate[first : last + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(parsed, list):
        return None
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        mods = item.get("expected_modules", [])
        if not q:
            continue
        if not isinstance(mods, list):
            mods = []
        mods = [str(m).strip() for m in mods if str(m).strip()]
        out.append({"question": q, "expected_modules": mods})
    return out or None


async def main() -> int:
    LLM_CACHE_DIR.mkdir(exist_ok=True)
    units = load_units()
    code_context = build_code_context()
    model = "opus"

    provider = BridgeProvider(model="opus")
    all_rows: List[Dict[str, Any]] = []
    live = 0
    hits = 0

    try:
        for cat_idx, cat in enumerate(CATEGORIES):
            corpus_digest = build_corpus_digest(
                units, chunk_idx=cat_idx, n_chunks=len(CATEGORIES)
            )
            guidance = CATEGORY_GUIDANCE[cat]
            user = (
                f"CATEGORY: {cat}\n\n{guidance}\n\n"
                f"Produce EXACTLY {PER_CATEGORY} {cat} questions.\n\n"
                f"Available file paths for expected_modules:\n"
                + "\n".join(f"  - {m}" for m in ALL_MODULES)
                + '\n  - corpus_sample.jsonl  (the work-unit digest)\n\n'
                "=================== CORPUS DIGEST ===================\n"
                f"{corpus_digest}\n\n"
                "=================== SUBSTRATE CODE ===================\n"
                f"{code_context}\n\n"
                f"Now write EXACTLY {PER_CATEGORY} {cat} questions as the JSON array."
            )
            questions: Optional[List[Dict[str, Any]]] = None
            raw_last = ""
            for attempt in range(2):
                u = user if attempt == 0 else (
                    user
                    + "\n\n[RETRY] Previous output did not parse as a JSON array. "
                    "Output ONLY the JSON array, nothing else."
                )
                text, from_cache = await call_opus(provider, SYSTEM, u, model)
                if from_cache:
                    hits += 1
                else:
                    live += 1
                raw_last = text
                questions = parse_questions(text)
                if questions is not None:
                    break
            if questions is None:
                print(f"  {cat}: PARSE FAIL x2 (raw[:120]={raw_last[:120]!r})")
                await provider.close()
                return 6
            # keep exactly PER_CATEGORY
            questions = questions[:PER_CATEGORY]
            if len(questions) < PER_CATEGORY:
                print(
                    f"  WARNING {cat}: got {len(questions)} < {PER_CATEGORY} questions"
                )
            for q in questions:
                all_rows.append(
                    {"category": cat, "question": q["question"],
                     "expected_modules": q["expected_modules"]}
                )
            print(f"  {cat}: {len(questions)} questions")
    finally:
        try:
            await provider.close()
        except Exception:
            pass

    # Assign qids sequentially q001..
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, row in enumerate(all_rows, start=1):
            out_row = {
                "qid": f"q{i:03d}",
                "category": row["category"],
                "question": row["question"],
                "expected_modules": row["expected_modules"],
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print(f"QUESTIONS DONE: {len(all_rows)} written to {OUT_PATH.name}")
    from collections import Counter
    by_cat = Counter(r["category"] for r in all_rows)
    print(f"  by category: {dict(by_cat)}")
    print(f"  live calls: {live}, cache hits: {hits}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
