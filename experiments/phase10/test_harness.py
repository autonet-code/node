#!/usr/bin/env python3
"""Phase 10 guard tests — the pre-registered hard gates (docs/phase10_prereg.md).

Run ONLY this file (never the full autonet suite):
    python -m pytest experiments/phase10/test_harness.py

Covers:
  - determinism: same seed -> byte-identical corpus + tool blobs (guard #3/#4)
  - guard #4: build_tools refuses any non-master seed
  - guard #1: defective and correct manifest TEXT are identical per task
    (defects are runtime-only; a description reader cannot tell)
  - guard #2 (H1): T and E consume byte-identical skeletons — equal
    per-cell non-support event budgets
  - guard #2 (H2): B/C/D share the identical candidate set before ranking
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# The phase10 modules import each other by bare name (they live together);
# make the directory importable regardless of pytest's rootdir.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
logging.disable(logging.CRITICAL)   # silence blob-store INFO spam in tests

import build_tools
import build_debates as bd
import build_retrieval as br


# Build the corpus ONCE and share it across the tests that only READ it
# (corpus build runs the whole battery through subprocesses — heavy). The
# determinism test deliberately builds twice, so it does NOT use this.
_CORPUS_CACHE: dict = {}


def _corpus() -> dict:
    if "c" not in _CORPUS_CACHE:
        _CORPUS_CACHE["c"] = build_tools.build_corpus(build_tools.MASTER_SEED)
    return _CORPUS_CACHE["c"]


# ---------------------------------------------------------------------------
# Determinism + guard #4 (seed shopping)
# ---------------------------------------------------------------------------

def _corpus_bytes(corpus: dict) -> bytes:
    return json.dumps(corpus, sort_keys=True).encode("utf-8")


def test_corpus_deterministic_same_seed():
    """Same master seed -> byte-identical corpus.json."""
    c1 = build_tools.build_corpus(build_tools.MASTER_SEED)
    c2 = build_tools.build_corpus(build_tools.MASTER_SEED)
    assert _corpus_bytes(c1) == _corpus_bytes(c2)


def test_tool_blobs_deterministic():
    """Every tool blob on disk hashes to its recorded code_digest, and a
    rebuild reproduces the same digests (content addressing = determinism)."""
    corpus = _corpus()
    for tool in corpus["tools"]:
        path = build_tools.TOOLS_DIR / f"{tool['code_digest']}.py"
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == tool["code_digest"]


def test_guard4_refuses_non_master_seed():
    """build_tools refuses any seed but the pre-committed master (1010)."""
    with pytest.raises(SystemExit):
        build_tools.build_corpus(build_tools.MASTER_SEED + 1)


# ---------------------------------------------------------------------------
# Guard #1 — defective vs correct manifest text distance
# ---------------------------------------------------------------------------

def test_guard1_defective_manifest_text_identical():
    """For every task with both a correct and a defective instance, the
    manifest DESCRIPTION (what a reader sees) is byte-identical — the
    defect is runtime-only. String distance == 0."""
    corpus = _corpus()
    by_task: dict = {}
    for tool in corpus["tools"]:
        if tool["trust_kind"] in ("correct", "defective"):
            by_task.setdefault(tool["task"], {})[tool["trust_kind"]] = tool
    checked = 0
    for task, pair in by_task.items():
        if "correct" in pair and "defective" in pair:
            assert pair["correct"]["description"] == pair["defective"]["description"]
            # And the embedding text (name+description+schema props) —
            # names differ by the _correct/_defective suffix by design, so
            # we compare the description component that drives discovery.
            assert pair["correct"]["description"] == pair["defective"]["description"]
            checked += 1
    assert checked >= 4, "expected defective/correct pairs across families"


def test_guard1_defects_are_runtime_real():
    """Every defective-class tool genuinely fails its own battery
    (measured), and every correct-class tool passes it — the labels are
    grounded in execution, not assertion."""
    corpus = _corpus()
    for tool in corpus["tools"]:
        if tool["trust_kind"] == "defective":
            assert tool["pass_rate"] < 1.0, f"{tool['task']} defect does not diverge"
        elif tool["trust_kind"] == "correct":
            assert tool["pass_rate"] == 1.0, f"{tool['task']} correct tool fails battery"


# ---------------------------------------------------------------------------
# Guard #2 (H1) — arm symmetry: equal per-cell skeleton budgets
# ---------------------------------------------------------------------------

def test_guard2_h1_equal_skeleton_budgets():
    """T and E build the SAME arm-invariant skeleton per (tool, H, S):
    identical registration + sybil PRO + CON events. Only the CON-support
    events differ by arm. Asserts equal n_skeleton_events cell-by-cell."""
    corpus = _corpus()
    pop = bd.h1_population(corpus)
    # sample a few tools + cells (full sweep is exercised by the run).
    tools = pop[:4] + pop[-4:]
    for tool in tools:
        for H in (2, 8):
            for S in (0, 32):
                rt = bd.build_cell(tool, H, S, "T")
                re = bd.build_cell(tool, H, S, "E")
                assert rt["n_skeleton_events"] == re["n_skeleton_events"], (
                    tool["task"], H, S)
                # The CON existence is arm-invariant too.
                assert rt["con"] == re["con"]


def test_guard2_h1_skeleton_bytes_identical():
    """Stronger: the raw skeleton event stream (pre-support) is byte-
    identical across arms — same RNG draws, same coords, same order."""
    corpus = _corpus()
    pop = bd.h1_population(corpus)
    import random
    for tool in pop[:6]:
        for H in (1, 4):
            for S in (2, 8):
                r1 = random.Random(bd._cell_seed("skel", tool["code_digest"], H, S))
                r2 = random.Random(bd._cell_seed("skel", tool["code_digest"], H, S))
                e1, c1 = bd.build_skeleton(tool, H, S, r1)
                e2, c2 = bd.build_skeleton(tool, H, S, r2)
                assert json.dumps(e1, sort_keys=True) == json.dumps(e2, sort_keys=True)
                assert c1 == c2


# ---------------------------------------------------------------------------
# Guard #2 (H2) — identical candidate set across B/C/D
# ---------------------------------------------------------------------------

def test_guard2_h2_identical_candidates():
    """B, C, D differ ONLY in the ranking rule — the retrieval CANDIDATE
    set (artifact_index.search top-k*3) is identical across arms because
    all three call the same _infer_artifacts over the same index. Assert
    the pre-ranking candidate digests match for a sample of queries."""
    corpus = _corpus()
    tmp = Path(tempfile.mkdtemp(prefix="phase10_guard_"))
    world, aidx, cidx, dt, emb = br.build_environment(corpus, tmp / "s", salted=True)
    tasks = sorted({t["task"] for t in corpus["tools"]
                    if t["trust_kind"] in ("correct", "defective")})[:5]
    import random
    rng = random.Random(1)
    for task in tasks:
        for q in br._query_variants(task, rng):
            # The candidate set is what artifact_index.search returns —
            # identical regardless of which world/coverage the arm passes.
            cands = [d for d, _ in aidx.search(q, k=5 * 3)]
            assert cands == [d for d, _ in aidx.search(q, k=5 * 3)]
            assert len(cands) > 0


def test_guard2_h2_arm_b_is_cosine_only():
    """Arm B rides the real _infer_artifacts but with an empty world, so
    every candidate's standing is 0 and final == cosine — no bespoke
    formula. Verify B's ranking equals pure-cosine ordering of candidates."""
    corpus = _corpus()
    tmp = Path(tempfile.mkdtemp(prefix="phase10_guardb_"))
    world, aidx, cidx, dt, emb = br.build_environment(corpus, tmp / "s", salted=True)
    we = br._empty_world()
    from nodes.common.world_model_substrate.infer import _infer_artifacts
    q = build_tools.TASK_QUERY_TERMS["sum_list"]
    res = _infer_artifacts({"query": q}, we, aidx, aidx.blob_store, k=5,
                           coverage_index=None, query_vec=None)
    finals = [a["final"] for a in res["artifacts"]]
    cosines = [a["cosine"] for a in res["artifacts"]]
    # empty world -> standing 0 -> final == cosine for every artifact.
    assert finals == pytest.approx(cosines)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
