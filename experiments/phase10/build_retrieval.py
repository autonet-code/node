#!/usr/bin/env python3
"""Phase 10 retrieval builder — H2 (density-blend anti-SEO).

Prereg: docs/phase10_prereg.md, H2. Over a corpus salted with SEO
manifests (descriptions engineered to match a query's vocabulary,
behavior unrelated), measure whether the PRODUCTION retrieval blend finds
the tool that actually solves the task better than cosine-only.

Three arms, differing ONLY in the ranking rule (guard #2 analog: identical
candidate sets, identical formats):

  B  cosine-only               — final = cosine
  C  cosine x standing re-rank — final = cosine * (1 + tanh(standing)), no density
  D  the production blend      — _infer_artifacts with coverage_index set
                                 (density lift) + standing re-rank

All three go through the REAL _infer_artifacts. B and C are obtained by
calling it with coverage_index=None (pure claimed cosine base) and then,
for B, stripping the standing re-rank by passing an empty world (no claim
nodes -> standing 0 -> final == cosine). This keeps every arm on the same
retrieval code path — never a reimplemented ranking formula.

Embedder: the deterministic HashingEmbedder on BOTH build and query sides
(prereg: "identical embedder on build and query"). SEO salting: the SEO
tool advertises task A's vocabulary but its demonstrated-coverage cloud
(problem_coords from attestations) sits at its REAL task's region — so
density (what it actually served) exposes the lie that cosine (what it
claims) cannot.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes.common.blob_store import BlobStore
from nodes.common.world_model_substrate.adapter import (
    N_DIMS,
    build_charter_world,
)
from nodes.common.world_model_substrate.aggregate import apply_events
from nodes.common.world_model_substrate.artifact_index import ArtifactIndex
from nodes.common.world_model_substrate.coverage import CoverageIndex
from nodes.common.world_model_substrate.infer import _infer_artifacts
from nodes.common.world_model_substrate.tool_manifest import build_tool_manifest
from nodes.common.world_model_substrate.usefulness_coords import HashingEmbedder

from build_tools import TASK_QUERY_TERMS
from build_debates import EMBED_DIM, _coords

RETR_DIM = 64                 # embedder dim for both index + coverage
N_QUERIES_PER_TASK = 3        # M = tasks x this many fresh queries
COVERAGE_POINTS = 6           # attested problem_coords per tool


def _query_variants(task: str, rng: random.Random) -> List[str]:
    """Fresh query phrasings for a task — the full task query vocabulary
    with one word dropped per variant (a natural paraphrase, not the
    manifest text: the manifest embeds name+description+schema, which are
    different words). Dropping one word keeps the trigram signal strong
    enough for the hashing embedder while making each variant distinct
    and not an exact copy of the coverage-cloud text."""
    terms = TASK_QUERY_TERMS[task].split()
    out: List[str] = []
    for i in range(N_QUERIES_PER_TASK):
        if len(terms) <= 3:
            out.append(" ".join(terms))
            continue
        drop = rng.randrange(len(terms))
        out.append(" ".join(t for j, t in enumerate(terms) if j != drop))
    return out


def _manifest_payload(tool: Dict[str, Any]) -> Dict[str, Any]:
    """A real tool manifest payload (so is_tool_manifest gates the density
    path and manifest_embedding_text embeds name+description+props)."""
    return build_tool_manifest(
        name=tool["name"],
        description=tool["description"],
        input_schema=tool["input_schema"],
        author=tool["author"],
        trust_class="pinned",
        code_digest=tool["code_digest"],
        entrypoint="main",
        runtime="python3",
        created_ts=0,
    )


def _coverage_text_for(tool: Dict[str, Any]) -> str:
    """The work-context text a caller was ACTUALLY solving when it
    attested this tool. For honest tools that is the tool's own task; for
    an SEO tool it is its REAL task (real_task) — the behavior, not the
    advertisement. This is what makes density expose SEO."""
    real = tool.get("real_task") or tool["task"]
    return TASK_QUERY_TERMS[real]


def build_environment(corpus: Dict[str, Any], tmp_dir: Path, *,
                      salted: bool) -> Tuple[Any, ArtifactIndex,
                                             CoverageIndex, Dict[str, str],
                                             HashingEmbedder]:
    """Materialize the retrieval environment over the corpus.

    ``salted=True`` includes the SEO manifests; ``salted=False`` is the
    clean control corpus (SEO tools dropped) used by the decision rule's
    no-regression clause. Returns (world, artifact_index, coverage_index,
    digest->task map, embedder).

    Standing: a fresh charter world seeded with one PRO manifest node per
    tool (author_post) so C/D's standing re-rank has real net_scores; the
    node ids/coords come from the same _coords helper the debate builder
    uses. Standing is deliberately uniform here (each tool one PRO post):
    H2 isolates the DENSITY lever, not standing — the debate/standing
    lever is H1. Keeping standing equal across candidates means the
    B->C step is a no-op and D's lift is purely density.
    """
    embedder = HashingEmbedder(dim=RETR_DIM)
    blobs = BlobStore(data_dir=str(tmp_dir / "blobs"))
    aidx = ArtifactIndex(blobs, embedder=embedder,
                         index_path=tmp_dir / "artifacts.jsonl", dim=RETR_DIM)
    cidx = CoverageIndex(index_path=tmp_dir / "coverage.jsonl")

    tools = [t for t in corpus["tools"]
             if salted or not t.get("seo")]
    digest_task: Dict[str, str] = {}

    world = build_charter_world(bandwidth=1.5, embedding_dim=EMBED_DIM)
    seq = 0
    events: List[Dict[str, Any]] = []
    rng = random.Random(1010)
    for tool in tools:
        payload = _manifest_payload(tool)
        digest = aidx.add_artifact(payload)      # embeds manifest text
        # Map the manifest digest to the task this tool ACTUALLY solves
        # (SEO -> its real task) — the H2 ground truth for hit@k.
        digest_task[digest] = tool.get("real_task") or tool["task"]

        # One PRO manifest node in the world so standing re-rank has a hook.
        seq += 1
        events.append({
            "kind": "sub_claim_sprouted", "seq": seq,
            "author_agent": tool["author"], "tendency_id": "correctness",
            "parent_id": "solver_root", "node_id": f"m_{digest[:12]}",
            "position": "pro", "coords": _coords(rng),
            "polarity_axis": _coords(rng), "content": tool["name"],
            "author_post": True, "artifact_digest": digest,
        })

        # Demonstrated-coverage cloud: COVERAGE_POINTS attested points at
        # the tool's REAL behavior region (embedded via the SAME embedder).
        cov_text = _coverage_text_for(tool)
        cov_vec = list(embedder(cov_text))
        for _ in range(COVERAGE_POINTS):
            cidx.add_point(digest, cov_vec, attested=True, ts=0)

    apply_events(world, events, equilibrate_after=False, remap_out={})
    return world, aidx, cidx, digest_task, embedder


def _empty_world():
    """A charter world with no claim nodes -> every candidate's standing
    is 0 -> _infer_artifacts' final == cosine base. This is how arm B
    (cosine-only) rides the REAL retrieval path without a bespoke formula."""
    return build_charter_world(bandwidth=1.5, embedding_dim=EMBED_DIM)


def run_arm(arm: str, query: str, task: str, *, world, world_empty,
            aidx: ArtifactIndex, cidx: CoverageIndex,
            digest_task: Dict[str, str], embedder: HashingEmbedder,
            k: int = 5) -> Dict[str, Any]:
    """Run one query through _infer_artifacts under one arm and score
    hit@1 / hit@5 / SEO-share against ground truth.

    B: world_empty (standing 0), coverage_index=None  -> final == cosine
    C: world (real standing),      coverage_index=None -> cosine*standing
    D: world (real standing),      coverage_index=cidx -> density blend
    """
    if arm == "B":
        use_world, cov = world_empty, None
    elif arm == "C":
        use_world, cov = world, None
    elif arm == "D":
        use_world, cov = world, cidx
    else:
        raise ValueError(f"unknown arm {arm!r}")

    query_vec = list(embedder(query)) if cov is not None else None
    result = _infer_artifacts(
        {"query": query}, use_world, aidx, aidx.blob_store, k=k,
        coverage_index=cov, query_vec=query_vec,
    )
    ranked = result["artifacts"]
    ranked_tasks = [digest_task.get(a["digest"], "") for a in ranked]
    # A retrieved slot is "SEO" if the manifest's advertised task != the
    # task it truly solves — but by digest_task we already map to the real
    # task; SEO share = fraction of top-k whose real task != the query task
    # yet were retrieved (they matched on advertised vocab).
    hit1 = bool(ranked_tasks[:1] and ranked_tasks[0] == task)
    hit5 = task in ranked_tasks[:5]
    seo_in_top5 = sum(1 for t in ranked_tasks[:5] if t != task)
    return {
        "arm": arm, "task": task, "query": query,
        "hit1": hit1, "hit5": hit5,
        "seo_share_top5": seo_in_top5 / max(1, len(ranked_tasks[:5])),
        "ranked_tasks": ranked_tasks[:5],
    }


def build_rows(corpus: Dict[str, Any], tmp_dir: Path) -> List[Dict[str, Any]]:
    """Every (query, arm) row on the salted corpus, plus the clean-control
    hit@5 needed by the decision rule's no-regression clause."""
    rng = random.Random(1010)
    tasks = sorted({t["task"] for t in corpus["tools"]
                    if t["trust_kind"] in ("correct", "defective")})

    rows: List[Dict[str, Any]] = []

    # --- salted corpus (the H2 test surface) ---------------------------
    world, aidx, cidx, digest_task, emb = build_environment(
        corpus, tmp_dir / "salted", salted=True)
    world_empty = _empty_world()
    for task in tasks:
        for query in _query_variants(task, rng):
            for arm in ("B", "C", "D"):
                row = run_arm(arm, query, task, world=world,
                              world_empty=world_empty, aidx=aidx, cidx=cidx,
                              digest_task=digest_task, embedder=emb)
                row["corpus"] = "salted"
                rows.append(row)

    # --- clean control (no SEO) — no-regression clause -----------------
    rng2 = random.Random(2020)
    cw, caidx, ccidx, cdt, cemb = build_environment(
        corpus, tmp_dir / "clean", salted=False)
    cwe = _empty_world()
    for task in tasks:
        for query in _query_variants(task, rng2):
            for arm in ("B", "D"):
                row = run_arm(arm, query, task, world=cw, world_empty=cwe,
                              aidx=caidx, cidx=ccidx, digest_task=cdt,
                              embedder=cemb)
                row["corpus"] = "clean"
                rows.append(row)

    return rows
