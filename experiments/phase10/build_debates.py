#!/usr/bin/env python3
"""Phase 10 debate builder — arms T (text CON) and E (evidence CON).

Prereg: docs/phase10_prereg.md, H1. For each pinned tool in the H1
population (the correct + defective classes) and each participation cell
(H honest observers x S author-side sybil PRO posters), build a debate
graph as a canonical EventBatch stream, in two arms:

  Arm T (text CON): a critic sprouts one CON on the tool. Honest
    observers CANNOT verify a text claim, so each observer posts support
    on the CON with a FIXED pre-registered prior:
      - genuinely defective tool: p = 0.6
      - correct tool (false CON):  p = 0.3
    (docs/phase10_prereg.md "Text-arm support prior"). The RNG is a fixed
    stream folded from the master seed + cell coordinates.

  Arm E (evidence CON): the critic's CON carries a replayable failing
    invocation (args + expected). Each honest observer REPLAYS it against
    the pinned code (batteries.replay_invocation, the real subprocess
    contract) and posts support IFF it reproduces. Deterministic — no
    prior. Against a correct tool the attached invocation does NOT
    reproduce, so evidence recruits nothing: evidence PROTECTS correct
    tools.

False CONs appear in BOTH arms at the SAME rate (25% of tools carry a
false CON regardless of true defectiveness) so arm E must demonstrate
both separations: recruit against real defects, recruit nothing against
correct tools.

GUARD #2 (arm symmetry): T and E consume byte-identical event streams
EXCEPT the CON-support events. The registration sprout, the CON sprout,
and the S sybil PRO posts are produced by shared helpers so the per-cell
event budget differs ONLY by which observers ended up supporting. The
builder records the exact per-cell event counts for the guard test to
assert equality of the non-support skeleton.

Standing is read from the REAL ledger replay (build_charter_world +
apply_events, exactly the pricing="ledger" path inside
federated_epoch_close) via _artifact_standing / _standing_of — never a
reimplemented formula.
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root FIRST on the path: a pip-installed `autonet` in site-packages
# would otherwise shadow the local nodes.* source we are testing against
# (the installed copy predates _artifact_standing).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.world_model_substrate.adapter import N_DIMS, build_charter_world
from nodes.common.world_model_substrate.aggregate import apply_events
from nodes.common.world_model_substrate.infer import (
    _artifact_standing,
    _standing_of,
)

from batteries import replay_invocation

# Pre-registered constants (docs/phase10_prereg.md).
PRIOR_DEFECTIVE = 0.6      # text CON against a real defect recruits at 0.6
PRIOR_CORRECT = 0.3        # false text CON against a correct tool recruits at 0.3
FALSE_CON_RATE = 0.25      # 25% of CONs are false (against correct tools)

# Sweep grid.
H_VALUES = [1, 2, 4, 8]
S_VALUES = [0, 2, 8, 32]

HERE = Path(__file__).resolve().parent
TOOLS_DIR = HERE / "tools"

EMBED_DIM = 64             # cheap replay world; mint math is width-blind


def _cell_seed(*parts: Any) -> int:
    """Deterministic per-cell seed folded from parts (FNV-1a, like the
    sim's _cell_seed). hash() is salted per-process, so we fold a stable
    byte hash instead."""
    h = 1469598103934665603
    for p in parts:
        for b in repr(p).encode():
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _coords(rng: random.Random, axis: int = 4) -> List[float]:
    """Distinct-per-call coords: charter axis mass + random embedding tail.
    Distinctness matters — identical coords collapse to one content-
    addressed node, merging separate claims' standing (sim lesson)."""
    out = [0.0] * (N_DIMS + EMBED_DIM)
    out[axis] = 0.6 + rng.random() * 0.3
    for j in range(N_DIMS, N_DIMS + EMBED_DIM):
        out[j] = rng.uniform(-1.0, 1.0)
    return out


def _reg_event(seq: int, digest: str, author: str,
               rng: random.Random) -> Dict[str, Any]:
    return {
        "kind": "sub_claim_sprouted", "seq": seq, "author_agent": author,
        "tendency_id": "correctness", "parent_id": "solver_root",
        "node_id": f"tool_{digest[:12]}", "position": "pro",
        "coords": _coords(rng), "polarity_axis": _coords(rng),
        "content": f"tool {digest[:8]}", "author_post": True,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": author},
    }


def _con_event(seq: int, digest: str, critic: str,
               rng: random.Random) -> Dict[str, Any]:
    """A CON on the tool. Carries the artifact_digest so its net_score
    subtracts from the manifest's standing (the anti-tool verdict)."""
    return {
        "kind": "sub_claim_sprouted", "seq": seq, "author_agent": critic,
        "tendency_id": "correctness", "parent_id": f"tool_{digest[:12]}",
        "node_id": f"con_{digest[:12]}", "position": "con",
        "coords": _coords(rng), "polarity_axis": _coords(rng),
        "content": f"con on {digest[:8]}", "author_post": True,
        "artifact_digest": digest,
    }


def _sybil_pro_event(seq: int, digest: str, sybil: str,
                     rng: random.Random) -> Dict[str, Any]:
    """An author-side sybil PRO post directly on the manifest node —
    props up the tool's standing (worst case: sybils evade the owner map,
    prereg H1 sweep). Carries the digest (adds to PRO standing)."""
    return {
        "kind": "sub_claim_sprouted", "seq": seq, "author_agent": sybil,
        "tendency_id": "correctness", "parent_id": f"tool_{digest[:12]}",
        "node_id": f"syb_{digest[:12]}_{seq}", "position": "pro",
        "coords": _coords(rng), "polarity_axis": _coords(rng),
        "content": "endorse", "author_post": True,
        "artifact_digest": digest,
    }


def _support_con_event(seq: int, digest: str, observer: str,
                       rng: random.Random) -> Dict[str, Any]:
    """An honest observer's PRO post ON THE CON — strengthens the CON's
    net_score, which propagates as a stronger subtraction from the tool's
    standing. Does NOT carry the digest (it's meta-debate on the CON, not
    a direct verdict on the artifact)."""
    return {
        "kind": "sub_claim_sprouted", "seq": seq, "author_agent": observer,
        "tendency_id": "correctness", "parent_id": f"con_{digest[:12]}",
        "node_id": f"sup_{digest[:12]}_{seq}", "position": "pro",
        "coords": _coords(rng), "polarity_axis": _coords(rng),
        "content": "reproduced", "author_post": True,
    }


def _batch(events: List[Dict[str, Any]], kp: Keypair) -> EventBatch:
    return EventBatch(
        rpb_address="rpb_phase10", sender_pubkey=kp.public_key,
        batch_seq=1, events=events, prev_batch_hash=b"",
        timestamp=1_700_000_000.0,
    )


def _false_con(digest: str) -> bool:
    """Deterministic 25% of tools carry a false CON (a CON is filed even
    though the tool is actually correct). Folded from the digest so it is
    stable and independent of true defectiveness."""
    h = int(hashlib.sha256(("falsecon:" + digest).encode()).hexdigest(), 16)
    return (h % 100) < int(FALSE_CON_RATE * 100)


def _con_fires(tool: Dict[str, Any]) -> bool:
    """Is there a CON on this tool at all? Every genuinely defective tool
    gets a CON (a critic found the defect); correct tools get a CON only
    when the deterministic false-CON draw fires. This keeps false CONs at
    the pre-registered 25% rate among CORRECT tools while ensuring real
    defects are always challenged."""
    if tool["defective"]:
        return True
    return _false_con(tool["code_digest"])


# Arm-E reproduction decision is arm/cell-INVARIANT (pure function of the
# tool's code + battery), so cache it per digest: the subprocess battery
# scan runs ONCE per tool instead of once per (H,S) cell. This is a pure
# memoization — no behavior change, only speed (keeps the sweep well under
# the guard-#5 runtime budget).
_REPRO_CACHE: Dict[str, bool] = {}


def _failing_invocation(tool: Dict[str, Any]) -> Dict[str, Any] | None:
    """The evidence a CON attaches in arm E: the first battery case on
    which THIS tool's output diverges from ground truth (a replayable
    failing invocation). None if the tool passes its whole battery — then
    the CON is a false CON and its attached invocation (a happy case) will
    NOT reproduce against a correct tool."""
    from batteries import run_tool, _norm
    code_path = TOOLS_DIR / f"{tool['code_digest']}.py"
    for case in tool["battery"]:
        r = run_tool(code_path, case["args"])
        if not (r["ok"] and r["output"] == _norm(case["expected"])):
            return {"args": case["args"], "expected": case["expected"]}
    # Correct tool: attach the FIRST battery case as the (non-reproducing)
    # invocation a false CON would cite — evidence must fail to recruit.
    if tool["battery"]:
        c = tool["battery"][0]
        return {"args": c["args"], "expected": c["expected"]}
    return None


def _evidence_reproduces(tool: Dict[str, Any]) -> bool:
    """Does the CON's attached invocation reproduce against this tool's
    pinned code? Cached per digest (arm/cell-invariant)."""
    digest = tool["code_digest"]
    if digest in _REPRO_CACHE:
        return _REPRO_CACHE[digest]
    invocation = _failing_invocation(tool)
    reproduces = False
    if invocation is not None:
        code_path = TOOLS_DIR / f"{digest}.py"
        reproduces = replay_invocation(code_path, invocation)
    _REPRO_CACHE[digest] = reproduces
    return reproduces


def _standing_from_events(events: List[Dict[str, Any]],
                          digest: str) -> float:
    """Standing of the manifest node via the REAL ledger replay path.

    Mirrors federated_epoch_close(pricing="ledger") exactly: fresh charter
    world, apply_events per canonical batch with equilibrate_after=False,
    then read _artifact_standing / _standing_of. No reimplemented formula.
    """
    kp = Keypair.generate()
    canon = canonical_order([_batch(events, kp)])
    world = build_charter_world(bandwidth=1.5, embedding_dim=EMBED_DIM)
    for batch in canon.ordered_batches:
        apply_events(world, batch.events, equilibrate_after=False,
                     remap_out={})
    nodes = _artifact_standing(world).get(digest, [])
    return _standing_of(nodes)


def build_skeleton(tool: Dict[str, Any], H: int, S: int,
                   rng: random.Random) -> Tuple[List[Dict[str, Any]], int]:
    """The arm-INVARIANT event skeleton for one (tool, H, S) cell:
    registration + S sybil PRO posts + (CON if one fires). Returns
    (events, con_seq) where con_seq is the CON's seq (or -1 if no CON).

    GUARD #2: this skeleton is byte-identical across arms T and E (same
    RNG draws, same event count). Only the SUPPORT events differ by arm.
    """
    digest = tool["code_digest"]
    author = tool["author"]
    events: List[Dict[str, Any]] = [_reg_event(1, digest, author, rng)]
    seq = 1
    for i in range(S):
        seq += 1
        events.append(_sybil_pro_event(seq, digest, f"sybil_{i}", rng))
    con_seq = -1
    if _con_fires(tool):
        seq += 1
        con_seq = seq
        events.append(_con_event(seq, digest, "critic", rng))
    return events, con_seq


def build_cell(tool: Dict[str, Any], H: int, S: int, arm: str,
               ) -> Dict[str, Any]:
    """Build one (tool, H, S, arm) debate and compute the tool's standing
    from the real ledger replay.

    Returns a row: {code_digest, defective, trust_kind, family, H, S, arm,
    standing, con, con_supporters, n_events, n_skeleton_events}.
    """
    digest = tool["code_digest"]
    # ONE rng per (tool, H, S) shared across arms for the skeleton so the
    # skeleton is byte-identical; the support decisions draw from a
    # SEPARATE per-(tool,H,S) stream so arm T's prior draws are stable and
    # arm E doesn't consume them.
    skel_rng = random.Random(_cell_seed("skel", digest, H, S))
    events, con_seq = build_skeleton(tool, H, S, skel_rng)
    n_skeleton = len(events)

    con_supporters = 0
    if con_seq >= 0:
        sup_rng = random.Random(_cell_seed("support", digest, H, S))
        seq = con_seq
        if arm == "T":
            # Text CON: each observer supports by the fixed prior.
            p = PRIOR_DEFECTIVE if tool["defective"] else PRIOR_CORRECT
            for _ in range(H):
                if sup_rng.random() < p:
                    seq += 1
                    events.append(
                        _support_con_event(seq, digest, f"obs_{seq}", sup_rng))
                    con_supporters += 1
        elif arm == "E":
            # Evidence CON: each observer replays; posts iff it reproduces.
            # The reproduction decision is arm/cell-invariant (cached).
            if _evidence_reproduces(tool):
                for _ in range(H):
                    seq += 1
                    events.append(
                        _support_con_event(seq, digest, f"obs_{seq}", sup_rng))
                    con_supporters += 1
        else:
            raise ValueError(f"unknown arm {arm!r}")

    standing = _standing_from_events(events, digest)
    return {
        "code_digest": digest,
        "defective": bool(tool["defective"]),
        "trust_kind": tool["trust_kind"],
        "family": tool["family"],
        "H": H, "S": S, "arm": arm,
        "standing": standing,
        "con": con_seq >= 0,
        "con_supporters": con_supporters,
        "n_events": len(events),
        "n_skeleton_events": n_skeleton,
    }


def h1_population(corpus: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The H1 classification set: pinned correct + defective classes.
    Wash/SEO adversaries are excluded (they belong to H2/H3) — H1 measures
    standing separation on tools with KNOWN implanted defects vs correct
    siblings."""
    return [t for t in corpus["tools"]
            if t["trust_kind"] in ("correct", "defective")]


def build_all(corpus: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build every (tool, H, S, arm) cell. Returns {"T": [...], "E": [...]}
    of standing rows. Deterministic; pure function of the corpus + the
    folded cell seeds."""
    pop = h1_population(corpus)
    out: Dict[str, List[Dict[str, Any]]] = {"T": [], "E": []}
    for arm in ("T", "E"):
        for H in H_VALUES:
            for S in S_VALUES:
                for tool in pop:
                    out[arm].append(build_cell(tool, H, S, arm))
    return out
