#!/usr/bin/env python3
"""Phase 8 contest runner — does equilibration earn its complexity?

Pre-registration: docs/phase8_prereg.md (binding). This runner owns the
Arms table and enforces guards #1, #2, #6, #7 in code (hard aborts).

Modes
-----
--calibrate
    Bare Haiku (arm A prompt) answers ALL of questions.jsonl. Writes
    calibration.jsonl rows {"qid","arm":"bare","prompt","answer"}.

--contest --selected selected_questions.jsonl
    Runs the five arms per prereg on each selected question:
      A bare   : no context.
      B rag    : top-5 of ArtifactIndex.search(q, k=15) by cosine;
                 context = payload problem+resolution per artifact (NO claims).
      C ledger : same k=15 candidates, re-ranked by
                 cosine * (1 + tanh(vote_standing)); top-5. vote_standing =
                 Σ(+1 PRO, -1 CON judge claim for that uid/digest) + outcome
                 signal. NO world, NO equilibration. Context = payloads +
                 verdict claims.
      D256/D64 : same re-rank FORM, but standing from the equilibrated world
                 via infer._infer_artifacts (Σ net_score PRO - Σ net_score CON
                 over claim nodes with that artifact_digest). Context identical
                 to C.
    Char budget 6000 for every context arm; per-artifact proportional truncate.

    Guards (hard aborts; prereg Amendment 2 applied):
      #1: arms B/C/D each have >=3 artifacts with non-None payloads.
      #2: per-row mean context length over {C, D256, D64} in [3000, 6000];
          arm B's per-row context in [3000, 6000] as well (same window).
      #6 (Amendment 2): the 10% per-row band applies ONLY within
          {C, D256, D64}. Arm B is exempt from the cross-arm band — omitting
          the claims section IS the B-vs-C treatment — and instead has a
          hard per-row floor of 2500 chars. Arms are NOT capped to the
          min-filled length (C/D payload richness must not be truncated
          down to B's).
    On any violation: print the row, write an ABORTED marker, sys.exit(1).

--mock
    Deterministic stub (answer = "MOCK") replacing the bridge, but the FULL
    retrieval/standing/context/guards pipeline still runs.

All bridge calls are cached in llm_cache/ keyed by sha256(prompt+system+model)
(guard #7). Progress jsonl is appended per question; resume skips completed
qids.

Usage
-----
    python run_contest.py --calibrate [--mock]
    python run_contest.py --contest --selected selected_questions.jsonl [--mock]
    # sandbox smoke:
    python run_contest.py --contest --selected sandbox/selected_questions.jsonl \
        --mock --outdir sandbox --corpus sandbox/corpus_sample.jsonl \
        --judge sandbox/judge_cache.jsonl --questions sandbox/questions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- sys.path bootstrap to the autonet repo (as phase7) --------------------
_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

from world_model.generalized.serialize import restore_world  # type: ignore  # noqa: E402
from world_model.models.tree import Position  # type: ignore  # noqa: E402

from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    default_usefulness_embedder,
)
from nodes.common.world_model_substrate.artifact_index import (  # type: ignore  # noqa: E402
    ArtifactIndex,
)
from nodes.common.world_model_substrate.infer import (  # type: ignore  # noqa: E402
    _infer_artifacts,
)
from nodes.common.blob_store import BlobStore  # type: ignore  # noqa: E402

try:
    from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001 — mock mode must not require the bridge import
    BridgeProvider = None  # type: ignore


HERE = Path(__file__).resolve().parent

CONTEXT_CHAR_CAP = 6000       # guard #2/#6 upper cap (also per-row context budget)
CONTEXT_CHAR_MIN = 3000       # guard #2 lower bound on the mean
CONTEXT_BAND = 0.10           # guard #6: per-row band within {C, D256, D64}
B_CONTEXT_FLOOR = 2500        # guard #6 Amendment 2: arm B hard per-row floor
SEARCH_K = 15                 # retrieval candidate pool
TOP_K = 5                     # kept per arm
MIN_ARTIFACTS = 3             # guard #1
INVALID_ROW_CEILING = 0.20    # abort if >20% of rows end invalid after retry
CONTEXT_ARMS = ("B", "C", "D256", "D64")
BAND_ARMS = ("C", "D256", "D64")   # Amendment 2: 10% band applies here only
DIM_FOR_ARM = {"B": 256, "C": 256, "D256": 256, "D64": 64}


# The bare (arm A) system prompt — question only, code-grounded, concise.
BARE_SYSTEM = (
    "You are answering a question about the autonet substrate codebase "
    "(the world-model substrate: nodes/common/world_model_substrate/, "
    "world_model/generalized/, the two-plane inference path, epoch close, "
    "Substrate.sol). Answer concisely and precisely. Cite specific files, "
    "functions, or constants where relevant. If you are unsure, say so rather "
    "than inventing file names."
)

# Context arms (B/C/D) share a system prompt: answer using the provided
# material. Symmetric across arms so the ONLY difference is the material.
CONTEXT_SYSTEM = (
    "You are answering a question about the autonet substrate codebase. "
    "Below is retrieved material (work units and, where present, vetted "
    "verdicts about them). Use it to answer concisely and precisely, citing "
    "specific files, functions, or constants where relevant. If the material "
    "does not cover the question, answer from your own knowledge and say so."
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """question rows: {"qid","category","question","expected_modules"}."""
    return _read_jsonl(path)


def load_corpus_by_uid(path: Path) -> Dict[str, Dict[str, Any]]:
    """corpus rows: {"uid","problem","resolution","outcome"} -> keyed by uid."""
    return {r["uid"]: r for r in _read_jsonl(path)}


def load_judge_by_uid(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """judge rows: {"uid","claims":[{"axis","position","text"}]} -> by uid."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not path.exists():
        return out
    for r in _read_jsonl(path):
        out[r["uid"]] = r.get("claims", []) or []
    return out


def _qid_of(q: Dict[str, Any]) -> str:
    # question schema uses "qid"; tolerate legacy "id".
    return str(q.get("qid") or q.get("id"))


def _question_text(q: Dict[str, Any]) -> str:
    return str(q.get("question", ""))


# ---------------------------------------------------------------------------
# Digest <-> uid mapping (the artifact index keys by blob digest; judge
# claims key by uid). We reproduce the blob digest the ingestion produced so
# we can map a retrieved digest back to its corpus uid + judge claims.
# ---------------------------------------------------------------------------


def _payload_digest(store: BlobStore, unit: Dict[str, Any]) -> str:
    """The digest add_artifact would have produced for this unit's payload.

    Must match build_worlds.py's payload construction exactly.
    """
    payload: Dict[str, Any] = {
        "problem": unit.get("problem", "") or "",
        "resolution": unit.get("resolution", "") or "",
    }
    if unit.get("outcome") is not None:
        payload["outcome"] = unit["outcome"]
    return store.add_json(payload)  # content-addressed; idempotent.


def build_digest_maps(
    corpus: Dict[str, Dict[str, Any]],
    judge: Dict[str, List[Dict[str, Any]]],
    store: BlobStore,
) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """Return (digest->uid, digest->judge_claims, digest->unit)."""
    digest_to_uid: Dict[str, str] = {}
    digest_to_claims: Dict[str, List[Dict[str, Any]]] = {}
    digest_to_unit: Dict[str, Dict[str, Any]] = {}
    for uid, unit in corpus.items():
        digest = _payload_digest(store, unit)
        digest_to_uid[digest] = uid
        digest_to_claims[digest] = judge.get(uid, [])
        digest_to_unit[digest] = unit
    return digest_to_uid, digest_to_claims, digest_to_unit


# ---------------------------------------------------------------------------
# Standing computations
# ---------------------------------------------------------------------------


def _outcome_signal(outcome: Any) -> float:
    if outcome is None:
        return 0.0
    if isinstance(outcome, (int, float)):
        return float(outcome)
    if isinstance(outcome, dict):
        return (float(outcome.get("accepted", 0)) + float(outcome.get("kept", 0))
                + float(outcome.get("built_on", 0)) + float(outcome.get("paid", 0)))
    return 0.0


def vote_standing(claims: List[Dict[str, Any]], outcome: Any) -> float:
    """Arm C standing: Σ(+1 PRO, -1 CON) over judge claims + outcome signal."""
    s = 0.0
    for c in claims:
        pos = str(c.get("position", "")).strip().lower()
        if pos in ("con", "against", "negative", "-1", "no"):
            s -= 1.0
        else:
            s += 1.0
    return s + _outcome_signal(outcome)


# ---------------------------------------------------------------------------
# Retrieval + re-rank (shared by B/C/D)
# ---------------------------------------------------------------------------


import math  # noqa: E402


def retrieve_candidates(index: ArtifactIndex, question: str, k: int = SEARCH_K
                        ) -> List[Tuple[str, float]]:
    """Top-k (digest, cosine) candidates. Shared across B/C/D."""
    return index.search(question, k=k)


def rank_B(candidates: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """Arm B: top-5 by cosine."""
    ranked = sorted(candidates, key=lambda dc: (-dc[1], dc[0]))
    return [{"digest": d, "cosine": c, "standing": None, "final": c}
            for d, c in ranked[:TOP_K]]


def rank_C(
    candidates: List[Tuple[str, float]],
    digest_to_claims: Dict[str, List[Dict[str, Any]]],
    digest_to_unit: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Arm C: re-rank by cosine * (1 + tanh(vote_standing)); top-5."""
    scored: List[Dict[str, Any]] = []
    for digest, cosine in candidates:
        claims = digest_to_claims.get(digest, [])
        outcome = (digest_to_unit.get(digest) or {}).get("outcome")
        standing = vote_standing(claims, outcome)
        final = cosine * (1.0 + math.tanh(standing))
        scored.append({"digest": digest, "cosine": cosine,
                       "standing": standing, "final": final})
    scored.sort(key=lambda a: (-a["final"], a["digest"]))
    return scored[:TOP_K]


def rank_D(
    world: Any, index: ArtifactIndex, store: BlobStore, question: str,
) -> List[Dict[str, Any]]:
    """Arm D: re-rank by equilibrated standing via infer._infer_artifacts.

    _infer_artifacts searches k*3 candidates and returns the top-k re-ranked
    by cosine*(1+tanh(standing)) where standing is the equilibrated
    net_score(PRO) - net_score(CON) over claim nodes with that digest. We call
    it with k=TOP_K (so it searches TOP_K*3 = 15 candidates, matching SEARCH_K).
    """
    probe = _infer_artifacts(
        {"question": question}, world=world,
        artifact_index=index, blob_store=store, k=TOP_K,
    )
    out: List[Dict[str, Any]] = []
    for art in probe.get("artifacts", []):
        out.append({
            "digest": art.get("digest"),
            "cosine": art.get("cosine"),
            "standing": art.get("standing"),
            "final": art.get("final"),
        })
    return out


# ---------------------------------------------------------------------------
# Context formatting (identical template for B/C/D; B omits claims section)
# ---------------------------------------------------------------------------


def _artifact_block(
    idx: int,
    problem: str,
    resolution: str,
    claims_lines: Optional[List[str]],
    budget: int,
) -> str:
    """Render one artifact block within `budget` chars (proportional truncate).

    Header is always kept; the problem/resolution (and claims for C/D) are
    truncated proportionally to fit the per-artifact budget.
    """
    header = f"--- artifact [{idx}] ---\n"
    body_parts = [f"PROBLEM:\n{problem}", f"RESOLUTION:\n{resolution}"]
    if claims_lines:
        body_parts.append("VERDICTS:\n" + "\n".join(claims_lines))
    body = "\n\n".join(body_parts)
    avail = max(0, budget - len(header))
    if len(body) > avail:
        body = body[:avail]
    return header + body


def format_context(
    ranked: List[Dict[str, Any]],
    digest_to_unit: Dict[str, Dict[str, Any]],
    digest_to_claims: Dict[str, List[Dict[str, Any]]],
    include_claims: bool,
    world_standing_by_digest: Optional[Dict[str, Any]] = None,
    cap: int = CONTEXT_CHAR_CAP,
) -> str:
    """Format B/C/D context. Identical template; B passes include_claims=False.

    For C/D the verdict section uses the SAME judge-claim texts (verdict
    content is identical across C and D per prereg). D additionally annotates
    each verdict with its equilibrated net_score when available.
    """
    n = max(1, len(ranked))
    per_art = cap // n
    blocks: List[str] = []
    for i, art in enumerate(ranked):
        digest = art["digest"]
        unit = digest_to_unit.get(digest) or {}
        problem = unit.get("problem", "") or ""
        resolution = unit.get("resolution", "") or ""
        claims_lines: Optional[List[str]] = None
        if include_claims:
            claims_lines = []
            for c in digest_to_claims.get(digest, []):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                pos = str(c.get("position", "")).upper()
                axis = str(c.get("axis", ""))
                claims_lines.append(f"  - [{pos} {axis}] {text}")
        blocks.append(_artifact_block(i, problem, resolution, claims_lines, per_art))
    ctx = "\n\n".join(blocks)
    return ctx[:cap]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

# The bridge session is agentic: without explicit framing, haiku responds
# with tool-use preambles ("I'll search for X in Y...") instead of answers
# (observed on the first calibration run — coordinator directive 2026-07-03).
# The framing is prepended to EVERY arm's prompt. It is IDENTICAL across
# arms except that context arms add "and from the provided material" (per
# the coordinator's wording), so it cannot become a confound.
NO_TOOLS_FRAMING_BARE = (
    "You have no tools, no file access, and no search. Do not describe what "
    "you would do — give your complete final answer now, from what you know. "
    "Cite file paths/functions where relevant."
)
NO_TOOLS_FRAMING_CONTEXT = (
    "You have no tools, no file access, and no search. Do not describe what "
    "you would do — give your complete final answer now, from what you know "
    "and from the provided material. Cite file paths/functions where relevant."
)

# Appended on retry after an invalid first attempt (narrated intentions or
# too short). A different prompt string -> different cache key, so the retry
# is a fresh call, and both attempts stay auditable in llm_cache/.
RETRY_NUDGE = "\n\nAnswer now in full; do not narrate intentions."


def bare_prompt(question: str) -> str:
    return f"{NO_TOOLS_FRAMING_BARE}\n\nQuestion: {question}\n\nAnswer:"


def context_prompt(question: str, context: str) -> str:
    return (
        f"{NO_TOOLS_FRAMING_CONTEXT}\n\n"
        "RETRIEVED MATERIAL:\n"
        f"{context}\n\n"
        "----\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Cached bridge call (guard #7)
# ---------------------------------------------------------------------------


def _cache_key(system: str, prompt: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


class LLMCache:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[str]:
        p = self.dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))["answer"]
            except Exception:  # noqa: BLE001
                return None
        return None

    def put(self, key: str, system: str, prompt: str, model: str, answer: str) -> None:
        p = self.dir / f"{key}.json"
        p.write_text(json.dumps({
            "key": key, "model": model, "system": system,
            "prompt": prompt, "answer": answer,
        }, ensure_ascii=False), encoding="utf-8")


async def call_llm(
    provider: Any, cache: LLMCache, system: str, prompt: str,
    model: str, mock: bool,
) -> str:
    key = _cache_key(system, prompt, model)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if mock:
        answer = "MOCK"
    else:
        assert provider is not None, "real run needs a BridgeProvider"
        try:
            result = await provider.send(
                messages=[{"role": "user", "content": prompt}],
                system=system, model=model,
            )
            answer = result.text or ""
        except Exception as e:  # noqa: BLE001
            answer = f"BRIDGE_ERROR: {type(e).__name__}: {e}"
    cache.put(key, system, prompt, model, answer)
    return answer


# ---------------------------------------------------------------------------
# Response validity (agentic-preamble rejection; coordinator directive)
# ---------------------------------------------------------------------------

MIN_ANSWER_CHARS = 200
# Narrated-intention opener referring to searching/opening files.
_INTENT_OPENER = re.compile(r"^\s*(I'll|I will|I need to|Let me)\b", re.IGNORECASE)
_INTENT_ACTION = re.compile(
    r"\b(search|look|find|open|read|examine|check|explore|locate|grep|scan)\b"
    r".{0,80}?\b(file|files|code|codebase|implementation|module|repo|source"
    r"|directory|\.py|\.sol)\b",
    re.IGNORECASE | re.DOTALL,
)


def answer_is_valid(answer: str) -> bool:
    """False for too-short answers or tool-use narration openers."""
    if answer.startswith("BRIDGE_ERROR"):
        return False
    if len(answer.strip()) < MIN_ANSWER_CHARS:
        return False
    head = answer.strip()[:300]
    if _INTENT_OPENER.search(head) and _INTENT_ACTION.search(head):
        return False
    return True


async def call_llm_validated(
    provider: Any, cache: LLMCache, system: str, prompt: str,
    model: str, mock: bool,
) -> Dict[str, Any]:
    """call_llm + validity gate. Invalid first attempt -> one retry with a
    firmer nudge appended (new cache key). Returns
    {"answer", "rejected_first_attempt" (opt), "valid": bool}.
    Mock mode bypasses validity ("MOCK" is deliberately short; validity is
    an LLM-behavior check, not a pipeline check).
    """
    answer = await call_llm(provider, cache, system, prompt, model, mock)
    if mock or answer_is_valid(answer):
        return {"answer": answer, "valid": True}
    retry_answer = await call_llm(
        provider, cache, system, prompt + RETRY_NUDGE, model, mock)
    return {
        "answer": retry_answer,
        "rejected_first_attempt": answer,
        "valid": answer_is_valid(retry_answer),
    }


# ---------------------------------------------------------------------------
# Guards (hard aborts)
# ---------------------------------------------------------------------------


class GuardViolation(Exception):
    pass


def enforce_guards(qid: str, arms: Dict[str, Dict[str, Any]]) -> None:
    """Guards #1, #2, #6 — raise GuardViolation on any breach.

    Prereg Amendment 2 (docs/phase8_prereg.md): guard #6's 10% per-row band
    applies ONLY within {C, D256, D64}. Arm B is exempt from the cross-arm
    band (omitting the claims section IS the B-vs-C treatment); instead B
    has a hard per-row floor of 2500 chars. Arms are NOT capped to the
    min-filled length — C/D payload richness must not be truncated to B's.
    """
    # Guard #1: arms B/C/D >=3 artifacts with non-None payloads.
    for arm in CONTEXT_ARMS:
        probe = arms[arm]["probe"]
        n_payload = sum(1 for a in probe["artifacts"] if a.get("payload_present"))
        if n_payload < MIN_ARTIFACTS:
            raise GuardViolation(
                f"[guard#1] qid={qid} arm={arm}: only {n_payload} artifacts with "
                f"non-None payloads (need >={MIN_ARTIFACTS})."
            )

    lengths = {arm: arms[arm]["context_chars"] for arm in CONTEXT_ARMS}

    # Guard #2: mean context length over {C, D256, D64} in [3000, 6000].
    cd_lengths = {arm: lengths[arm] for arm in BAND_ARMS}
    cd_mean = sum(cd_lengths.values()) / len(cd_lengths)
    if not (CONTEXT_CHAR_MIN <= cd_mean <= CONTEXT_CHAR_CAP):
        raise GuardViolation(
            f"[guard#2] qid={qid}: mean C/D context length {cd_mean:.0f} outside "
            f"[{CONTEXT_CHAR_MIN}, {CONTEXT_CHAR_CAP}]. per-arm={lengths}"
        )

    # Guard #6 (Amendment 2): 10% per-row band within {C, D256, D64} only.
    lo, hi = min(cd_lengths.values()), max(cd_lengths.values())
    if lo <= 0 or (hi - lo) / cd_mean > CONTEXT_BAND:
        raise GuardViolation(
            f"[guard#6] qid={qid}: C/D context lengths not within "
            f"{CONTEXT_BAND:.0%} (min={lo}, max={hi}, mean={cd_mean:.0f}). "
            f"per-arm={cd_lengths}"
        )

    # Guard #6 (Amendment 2): arm B hard per-row floor (and the shared cap).
    b_len = lengths["B"]
    if b_len < B_CONTEXT_FLOOR:
        raise GuardViolation(
            f"[guard#6-B] qid={qid}: arm B context {b_len} below hard floor "
            f"{B_CONTEXT_FLOOR}."
        )
    if b_len > CONTEXT_CHAR_CAP:
        raise GuardViolation(
            f"[guard#6-B] qid={qid}: arm B context {b_len} above cap "
            f"{CONTEXT_CHAR_CAP}."
        )


def enforce_mean_window(arm: str, per_row_lengths: List[int]) -> None:
    """Run-level guard #2 for a single arm: mean over all rows in
    [3000, 6000]. Amendment 2 gives arm B "the same [3000, 6000]
    mean-length window as the others" — B's per-row floor is 2500, so its
    run mean must still clear 3000. Called once after the loop completes.
    """
    if not per_row_lengths:
        return
    mean_len = sum(per_row_lengths) / len(per_row_lengths)
    if not (CONTEXT_CHAR_MIN <= mean_len <= CONTEXT_CHAR_CAP):
        raise GuardViolation(
            f"[guard#2-mean] arm={arm}: run-level mean context length "
            f"{mean_len:.0f} outside [{CONTEXT_CHAR_MIN}, {CONTEXT_CHAR_CAP}] "
            f"over {len(per_row_lengths)} rows."
        )


def abort(outdir: Path, qid: str, exc: Exception, arms: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "!" * 72)
    print(f"  GUARD VIOLATION on qid={qid}")
    print(f"  {exc}")
    for arm in CONTEXT_ARMS:
        p = arms.get(arm, {})
        print(f"    arm {arm}: context_chars={p.get('context_chars')} "
              f"n_artifacts={len(p.get('probe', {}).get('artifacts', []))}")
    print("!" * 72 + "\n")
    marker = outdir / "ABORTED"
    marker.write_text(json.dumps({
        "qid": qid, "reason": str(exc), "ts": time.time(),
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# Probe metadata (persisted per arm)
# ---------------------------------------------------------------------------


def probe_meta(
    ranked: List[Dict[str, Any]],
    digest_to_uid: Dict[str, str],
    digest_to_unit: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    arts = []
    for a in ranked:
        digest = a.get("digest")
        unit = digest_to_unit.get(digest) or {}
        payload_present = bool(unit.get("problem") or unit.get("resolution"))
        arts.append({
            "digest": digest,
            "uid": digest_to_uid.get(digest),
            "cosine": a.get("cosine"),
            "standing": a.get("standing"),
            "final": a.get("final"),
            "payload_present": payload_present,
        })
    return {"region_size": len(arts), "artifacts": arts}


# ---------------------------------------------------------------------------
# Calibrate mode
# ---------------------------------------------------------------------------


async def run_calibrate(args, cache: LLMCache) -> int:
    outdir = Path(args.outdir)
    questions_path = Path(args.questions)
    out_path = outdir / "calibration.jsonl"
    progress_path = outdir / "calibration_progress.jsonl"

    questions = load_questions(questions_path)
    print(f"  calibrate: {len(questions)} questions, mock={args.mock}")

    done: set = set()
    if progress_path.exists():
        for r in _read_jsonl(progress_path):
            done.add(r["qid"])
        print(f"  resume: {len(done)} already done")

    provider = None
    if not args.mock:
        if BridgeProvider is None:
            print("  ERROR: BridgeProvider unavailable and not --mock"); return 1
        provider = BridgeProvider(model="haiku")

    try:
        for i, q in enumerate(questions, start=1):
            qid = _qid_of(q)
            if qid in done:
                continue
            prompt = bare_prompt(_question_text(q))
            result = await call_llm_validated(
                provider, cache, BARE_SYSTEM, prompt, "haiku", args.mock)
            row = {"qid": qid, "arm": "bare", "prompt": prompt,
                   "answer": result["answer"], "valid": result["valid"]}
            if "rejected_first_attempt" in result:
                row["rejected_first_attempt"] = result["rejected_first_attempt"]
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": qid, "ts": time.time()}) + "\n")
            flag = "" if result["valid"] else "  [INVALID after retry]"
            retried = "  [retried]" if "rejected_first_attempt" in result else ""
            print(f"  [{i}/{len(questions)}] {qid}: "
                  f"{len(result['answer'])} chars{retried}{flag}")
    finally:
        if provider is not None:
            try:
                await provider.close()
            except Exception:  # noqa: BLE001
                pass

    # Validity gate: abort if >20% of rows ended invalid after retry.
    all_rows = _read_jsonl(out_path) if out_path.exists() else []
    n_invalid = sum(1 for r in all_rows if r.get("valid") is False)
    if all_rows and n_invalid / len(all_rows) > INVALID_ROW_CEILING:
        print(f"\n  ABORT: {n_invalid}/{len(all_rows)} calibration rows invalid "
              f"after retry (> {INVALID_ROW_CEILING:.0%}).")
        (outdir / "ABORTED").write_text(json.dumps({
            "qid": None,
            "reason": f"calibration invalid-answer rate {n_invalid}/{len(all_rows)}",
            "ts": time.time(),
        }), encoding="utf-8")
        sys.exit(1)
    if n_invalid:
        print(f"  note: {n_invalid}/{len(all_rows)} rows invalid after retry "
              f"(within {INVALID_ROW_CEILING:.0%} ceiling)")

    print(f"  wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Contest mode
# ---------------------------------------------------------------------------


def _load_world(outdir: Path, dim: int) -> Any:
    """Restore the world and re-attach the artifact_digest sidecar.

    snapshot_world() does not persist the custom node.artifact_digest
    attribute, so build_worlds.py writes a node_id -> digest sidecar. Without
    re-attaching it, _infer_artifacts finds zero standing and D collapses to
    cosine-only (the phase-6 silent-empty-context failure). We fail LOUD if
    the sidecar is missing or attaches nothing.
    """
    wp = outdir / f"world_{dim}.json"
    dp = outdir / f"world_{dim}.digests.json"
    if not wp.exists():
        raise FileNotFoundError(f"world snapshot missing: {wp} (run build_worlds.py)")
    if not dp.exists():
        raise FileNotFoundError(f"digest sidecar missing: {dp} (run build_worlds.py)")
    world = restore_world(json.loads(wp.read_text(encoding="utf-8")))
    digest_map = json.loads(dp.read_text(encoding="utf-8"))
    attached = 0
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            d = digest_map.get(node.id)
            if d:
                node.artifact_digest = d
                attached += 1
    if attached == 0:
        raise RuntimeError(
            f"dim={dim}: digest sidecar attached 0 nodes — D standing would be "
            f"uniformly zero (phase-6 silent failure). Rebuild worlds."
        )
    print(f"  dim={dim}: re-attached {attached} artifact digests to world nodes")
    return world


async def run_contest(args, cache: LLMCache) -> int:
    outdir = Path(args.outdir)
    selected_path = Path(args.selected)
    corpus_path = Path(args.corpus)
    judge_path = Path(args.judge)
    out_path = outdir / "contest_rows.jsonl"
    progress_path = outdir / "contest_progress.jsonl"

    selected = load_questions(selected_path)
    corpus = load_corpus_by_uid(corpus_path)
    judge = load_judge_by_uid(judge_path)
    print(f"  contest: {len(selected)} selected questions, "
          f"{len(corpus)} corpus units, mock={args.mock}")

    # Data plane: shared blob store; one ArtifactIndex per dim (256, 64).
    store = BlobStore(data_dir=str(outdir / "blobs"))
    digest_to_uid, digest_to_claims, digest_to_unit = build_digest_maps(
        corpus, judge, store)

    indexes: Dict[int, ArtifactIndex] = {}
    worlds: Dict[int, Any] = {}
    for dim in (256, 64):
        idx_path = outdir / f"artifact_index_{dim}.jsonl"
        if not idx_path.exists():
            print(f"  ERROR: artifact index missing: {idx_path} (run build_worlds.py)")
            return 1
        indexes[dim] = ArtifactIndex(
            store, embedder=default_usefulness_embedder(dim=dim),
            index_path=idx_path, dim=dim)
        worlds[dim] = _load_world(outdir, dim)
        print(f"  dim={dim}: {len(indexes[dim])} artifacts indexed, world restored")

    done: set = set()
    if progress_path.exists():
        for r in _read_jsonl(progress_path):
            done.add(r["qid"])
        print(f"  resume: {len(done)} qids already done")

    provider = None
    if not args.mock:
        if BridgeProvider is None:
            print("  ERROR: BridgeProvider unavailable and not --mock"); return 1
        provider = BridgeProvider(model="haiku")

    try:
        for i, q in enumerate(selected, start=1):
            qid = _qid_of(q)
            if qid in done:
                continue
            question = _question_text(q)

            # --- retrieval (shared candidates from the dim-256 index) ---
            cand_256 = retrieve_candidates(indexes[256], question, k=SEARCH_K)

            ranked: Dict[str, List[Dict[str, Any]]] = {}
            ranked["B"] = rank_B(cand_256)
            ranked["C"] = rank_C(cand_256, digest_to_claims, digest_to_unit)
            ranked["D256"] = rank_D(worlds[256], indexes[256], store, question)
            ranked["D64"] = rank_D(worlds[64], indexes[64], store, question)

            arms: Dict[str, Dict[str, Any]] = {}

            # --- arm A: bare ---
            a_prompt = bare_prompt(question)
            a_result = await call_llm_validated(
                provider, cache, BARE_SYSTEM, a_prompt, "haiku", args.mock)
            arms["A"] = {"prompt": a_prompt, "answer": a_result["answer"],
                         "valid": a_result["valid"],
                         "context_chars": 0,
                         "probe": {"region_size": 0, "artifacts": []}}
            if "rejected_first_attempt" in a_result:
                arms["A"]["rejected_first_attempt"] = a_result["rejected_first_attempt"]

            # --- context arms B/C/D ---
            for arm in CONTEXT_ARMS:
                include_claims = arm != "B"  # B gets no claims section (prereg).
                ctx = format_context(
                    ranked[arm], digest_to_unit, digest_to_claims,
                    include_claims=include_claims)
                prompt = context_prompt(question, ctx)
                result = await call_llm_validated(
                    provider, cache, CONTEXT_SYSTEM, prompt, "haiku", args.mock)
                arms[arm] = {
                    "prompt": prompt,
                    "answer": result["answer"],
                    "valid": result["valid"],
                    "context_chars": len(ctx),
                    "probe": probe_meta(ranked[arm], digest_to_uid, digest_to_unit),
                }
                if "rejected_first_attempt" in result:
                    arms[arm]["rejected_first_attempt"] = result["rejected_first_attempt"]

            # --- guards (hard abort) ---
            try:
                enforce_guards(qid, arms)
            except GuardViolation as gv:
                abort(outdir, qid, gv, arms)
                if provider is not None:
                    try:
                        await provider.close()
                    except Exception:  # noqa: BLE001
                        pass
                sys.exit(1)

            row = {"qid": qid, "arms": arms}
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"qid": qid, "ts": time.time()}) + "\n")

            lens = {a: arms[a]["context_chars"] for a in CONTEXT_ARMS}
            print(f"  [{i}/{len(selected)}] {qid}: ctx={lens}")
    finally:
        if provider is not None:
            try:
                await provider.close()
            except Exception:  # noqa: BLE001
                pass

    # Run-level guard #2 (Amendment 2): each context arm's mean over ALL
    # completed rows (including resumed ones) must sit in [3000, 6000].
    # Read back from contest_rows.jsonl so the check covers the full run.
    all_rows = _read_jsonl(out_path) if out_path.exists() else []

    # Validity gate: a row is invalid if ANY arm's final answer failed
    # validity after its retry. Abort if >20% of rows end invalid.
    invalid_rows = [
        r["qid"] for r in all_rows
        if any(a.get("valid") is False for a in r.get("arms", {}).values())
    ]
    if all_rows and len(invalid_rows) / len(all_rows) > INVALID_ROW_CEILING:
        print(f"\n  ABORT: {len(invalid_rows)}/{len(all_rows)} contest rows have "
              f"an invalid arm answer after retry (> {INVALID_ROW_CEILING:.0%}): "
              f"{invalid_rows}")
        (outdir / "ABORTED").write_text(json.dumps({
            "qid": None,
            "reason": f"contest invalid-answer rows {len(invalid_rows)}/{len(all_rows)}",
            "ts": time.time(),
        }), encoding="utf-8")
        sys.exit(1)
    if invalid_rows:
        print(f"  note: invalid-after-retry rows (within ceiling): {invalid_rows}")
    for arm in CONTEXT_ARMS:
        per_row = [r["arms"][arm]["context_chars"] for r in all_rows
                   if arm in r.get("arms", {})]
        try:
            enforce_mean_window(arm, per_row)
        except GuardViolation as gv:
            print("\n" + "!" * 72)
            print(f"  RUN-LEVEL GUARD VIOLATION: {gv}")
            print("!" * 72 + "\n")
            (outdir / "ABORTED").write_text(json.dumps({
                "qid": None, "reason": str(gv), "ts": time.time(),
            }), encoding="utf-8")
            sys.exit(1)

    print(f"  wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--contest", action="store_true")
    parser.add_argument("--selected", default=None,
                        help="selected_questions.jsonl (contest mode)")
    parser.add_argument("--mock", action="store_true",
                        help="deterministic stub answer='MOCK'; full pipeline runs")
    parser.add_argument("--outdir", default=str(HERE))
    parser.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    parser.add_argument("--corpus", default=str(HERE / "corpus_sample.jsonl"))
    parser.add_argument("--judge", default=str(HERE / "judge_cache.jsonl"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("nodes.common.blob_store").setLevel(logging.ERROR)

    cache = LLMCache(Path(args.outdir) / "llm_cache")

    if args.calibrate:
        return await run_calibrate(args, cache)
    if args.contest:
        if not args.selected:
            print("  --contest requires --selected"); return 1
        return await run_contest(args, cache)
    print("  nothing to do: pass --calibrate or --contest")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
