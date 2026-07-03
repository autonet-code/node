"""Inference: query the global world for the region relevant to a
request, return its structure.

Two query modes:

  1. Charter-alignment query (default for turn-shaped inputs):
     evaluate the input against the four charter tendencies, return
     which root principle dominates and the softmax of root scores.
     This is the original alignment-scoring path.

  2. General query (any other input):
     locate the query in the graph, render the located region's
     structure, return the rendered output. The decoder (LLM at the
     I/O boundary, or the calling pipeline) consumes this structure
     to produce its final answer in whatever format it needs.

The two paths share the same backing graph and the same locate
primitive. Charter-alignment is just locate restricted to the four
root tendencies plus a softmax readout.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from world_model.generalized import (
    Observation,
    default_locator,
    equilibrate,
    render,
)
from world_model.models.tree import Position

from .adapter import (
    CHARTER,
    build_charter_world,
    deserialize_world,
    turn_to_observation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]


def _looks_like_turn(input_data: Dict[str, Any]) -> bool:
    """Heuristic: is this input a charter-alignment query (i.e., does
    it look like an agent turn we'd score against the charter roots)?

    Markers: presence of 'turn', 'turns', or any of the
    *_impact fields anywhere in the dict.
    """
    if not isinstance(input_data, dict):
        return False
    if "turn" in input_data or "turns" in input_data:
        return True
    impact_keys = {
        "life_impact", "self_pres_impact",
        "intelligence_impact", "evolution_impact",
        "correctness_impact", "simplicity_impact",
    }
    if any(k in input_data for k in impact_keys):
        return True
    # Inner inspection: 'tool_calls', 'tool_call', 'tool', 'command'
    # are turn-shaped. 'query', 'question', 'request' are general.
    if any(k in input_data for k in ("tool_calls", "tool_call", "tool", "command")):
        return True
    return False


# ---------------------------------------------------------------------------
# Charter-alignment inference (original path)
# ---------------------------------------------------------------------------


def _infer_alignment(
    input_data: Dict[str, Any],
    global_model_payload: Optional[Dict[str, Any]] = None,
    world: Optional[Any] = None,
) -> Dict[str, Any]:
    if "turns" in input_data:
        turns = input_data["turns"]
    elif "turn" in input_data:
        turns = [input_data["turn"]]
    else:
        turns = [input_data]

    if world is None:
        if global_model_payload is not None:
            world = deserialize_world(global_model_payload)
        else:
            world = build_charter_world()

    for i, turn in enumerate(turns):
        world.add_observation(turn_to_observation(turn, turn_index=i))
    equilibrate(world, max_rounds=8, tolerance=1e-3)

    scores = world.root_scores()
    ordered = [scores.get(entry["id"], 0.0) for entry in CHARTER]
    probs = _softmax(ordered)
    pred = max(range(len(ordered)), key=lambda i: ordered[i])

    return {
        "mode": "alignment",
        "predictions": [pred],
        "probabilities": [probs],
        "root_scores": dict(scores),
        "aligned_principle": CHARTER[pred]["id"],
        "principle_thesis": CHARTER[pred]["thesis"],
    }


# ---------------------------------------------------------------------------
# General inference (locate + render)
# ---------------------------------------------------------------------------


def _infer_general(
    input_data: Dict[str, Any],
    global_model_payload: Optional[Dict[str, Any]] = None,
    world: Optional[Any] = None,
) -> Dict[str, Any]:
    if world is None:
        if global_model_payload is not None:
            world = deserialize_world(global_model_payload)
        else:
            world = build_charter_world()

    # Equilibrate once so existing structure has settled scores
    equilibrate(world, max_rounds=4, tolerance=1e-3)

    locator = default_locator()
    region = locator(world, input_data)
    structure = render(world, region, descendants_per_node=3)

    return {
        "mode": "general",
        "structure": structure,
        # Compatibility fields so downstream callers expecting the
        # alignment shape don't crash. They're empty/default for
        # general queries.
        "predictions": [],
        "probabilities": [],
        "root_scores": world.root_scores(),
    }


# ---------------------------------------------------------------------------
# Artifact retrieval inference (data plane)
# ---------------------------------------------------------------------------


def _query_text(input_data: Dict[str, Any]) -> str:
    """Extract the query text from an artifact request.

    Accepts ``text``, ``query``, or ``question`` (first present).
    Raises ValueError if none is provided.
    """
    for key in ("text", "query", "question"):
        if key in input_data and input_data[key] is not None:
            return str(input_data[key])
    raise ValueError(
        "artifacts inference requires 'text', 'query', or 'question' in input_data"
    )


def _artifact_standing(world: Any) -> Dict[str, List[Any]]:
    """Walk every node across all tendencies once, grouping claim nodes
    by their ``artifact_digest`` attribute.

    Returns a map ``digest -> [claim nodes]``. Nodes without an
    ``artifact_digest`` (or with an empty one) are skipped. The
    attribute is read via ``getattr(node, "artifact_digest", "")`` —
    it is set by the ingestion agent and may be absent.
    """
    by_digest: Dict[str, List[Any]] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            digest = getattr(node, "artifact_digest", "")
            if not digest:
                continue
            by_digest.setdefault(digest, []).append(node)
    return by_digest


def _standing_of(nodes: List[Any]) -> float:
    """Standing = Σ net_score(PRO-positioned) − Σ net_score(CON-positioned)."""
    standing = 0.0
    for node in nodes:
        if node.position == Position.CON:
            standing -= node.net_score
        elif node.position == Position.PRO:
            standing += node.net_score
    return standing


def _infer_artifacts(
    input_data: Dict[str, Any],
    world: Any,
    artifact_index: Any,
    blob_store: Any,
    k: int = 5,
) -> Dict[str, Any]:
    """Retrieve artifacts by embedding similarity, re-ranked by graph standing.

    See ``docs/two_plane_inference.md`` section 4. The claim graph is a
    verdict layer: artifacts are retrieved from the embedding index and
    re-priced by the net_score of the PRO/CON claim nodes that reference
    each artifact digest.
    """
    query = _query_text(input_data)

    candidates = artifact_index.search(query, k=k * 3)
    by_digest = _artifact_standing(world)

    scored: List[Dict[str, Any]] = []
    for digest, cosine in candidates:
        claim_nodes = by_digest.get(digest, [])
        standing = _standing_of(claim_nodes)
        final = cosine * (1.0 + math.tanh(standing))
        claims = sorted(
            (
                {
                    "content": node.content,
                    "position": node.position.value,
                    "net_score": node.net_score,
                }
                for node in claim_nodes
            ),
            key=lambda c: c["content"],
        )
        scored.append(
            {
                "digest": digest,
                "cosine": cosine,
                "standing": standing,
                "final": final,
                "payload": blob_store.get_json(digest),
                "claims": claims,
            }
        )

    # Deterministic: highest final first, tie-break on digest ascending.
    scored.sort(key=lambda a: (-a["final"], a["digest"]))
    artifacts = scored[:k]

    return {
        "mode": "artifacts",
        "artifacts": artifacts,
        # Compatibility fields so downstream callers expecting the
        # alignment shape don't crash.
        "predictions": [],
        "probabilities": [],
    }


# ---------------------------------------------------------------------------
# Context rendering (phase8 arm B format — docs/ledger_pricing.md Change 3)
# ---------------------------------------------------------------------------


def _render_block(idx: int, problem: str, resolution: str, budget: int) -> str:
    """Render one artifact block within ``budget`` chars.

    Mirrors ``experiments/phase8/run_contest.py:_artifact_block`` (arm B
    shape): a file-style header plus PROBLEM/RESOLUTION text, no claims,
    no digests, no standing numbers. Proportionally truncated to fit the
    per-artifact budget; the header itself is never truncated.
    """
    header = f"--- artifact [{idx}] ---\n"
    body = f"PROBLEM:\n{problem}\n\nRESOLUTION:\n{resolution}"
    avail = max(0, budget - len(header))
    if len(body) > avail:
        body = body[:avail]
    return header + body


def render_context(result: Dict[str, Any], char_budget: int = 6000) -> str:
    """Render an ``_infer_artifacts`` result into a phase8 arm-B-style
    context string: standing-ranked payloads only.

    Per docs/ledger_pricing.md Change 3 (phase8: C−B = −0.28 — verdicts
    in the prompt hurt): renders each artifact's payload problem +
    resolution text behind a small index header. NO claims, NO digest
    hashes, NO standing numbers — nothing that reads as verdict text.
    Claims/standing stay available in the structured result for
    programmatic consumers; this function is prompt-facing only.

    Artifacts already arrive in standing-ranked order from
    ``_infer_artifacts`` (sorted by ``final`` descending); this function
    preserves that order. Artifacts whose ``payload`` is ``None`` (blob
    not yet replicated) are skipped. Proportional truncation across
    artifacts mirrors ``run_contest.py``'s ``format_context``/
    ``_artifact_block``: the budget is divided evenly by artifact count,
    and each block is truncated independently to fit its share.
    Deterministic: pure function of ``result`` and ``char_budget``.
    """
    artifacts = [a for a in result.get("artifacts", []) if a.get("payload") is not None]
    if not artifacts:
        return ""

    per_art = char_budget // len(artifacts)
    blocks = []
    for i, art in enumerate(artifacts):
        payload = art["payload"] or {}
        problem = payload.get("problem", "") or ""
        resolution = payload.get("resolution", "") or ""
        blocks.append(_render_block(i, problem, resolution, per_art))
    ctx = "\n\n".join(blocks)
    return ctx[:char_budget]


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def infer_with_world_model(
    input_data: Dict[str, Any],
    global_model_payload: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    *,
    world: Optional[Any] = None,
    artifact_index: Optional[Any] = None,
    blob_store: Optional[Any] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """Run inference. Three modes auto-selected unless overridden:

      mode='alignment' or input looks turn-shaped:
        Score against the charter tendencies. Return the dominant
        principle and softmax probabilities. Original path.

      mode='artifacts' or (artifact_index + blob_store provided and
      input is not turn-shaped):
        Retrieve artifacts from the embedding index and re-rank by
        graph standing. Data-plane retrieval. See
        ``docs/two_plane_inference.md``.

      mode='general' or input is anything else:
        locate(input) -> render(region). Return the rendered region's
        structure for downstream rendering.

    When ``world`` is passed it is used directly by every mode instead
    of deserializing ``global_model_payload`` or building a fresh
    charter world. When ``world`` is None behavior is unchanged.
    """
    if mode is None:
        if _looks_like_turn(input_data):
            mode = "alignment"
        elif artifact_index is not None and blob_store is not None:
            mode = "artifacts"
        else:
            mode = "general"

    if mode == "alignment":
        return _infer_alignment(input_data, global_model_payload, world=world)
    elif mode == "general":
        return _infer_general(input_data, global_model_payload, world=world)
    elif mode == "artifacts":
        if artifact_index is None or blob_store is None or world is None:
            raise ValueError(
                "artifacts mode requires world, artifact_index, and blob_store"
            )
        return _infer_artifacts(input_data, world, artifact_index, blob_store, k=k)
    else:
        raise ValueError(f"unknown inference mode: {mode!r}")
