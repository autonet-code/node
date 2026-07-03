#!/usr/bin/env python3
"""Phase 8 world builder — serialize the two substrate worlds (D256, D64).

Ingests corpus_sample.jsonl + judge_cache.jsonl via the committed two-plane
path (usefulness_training.py pattern):

  For each corpus unit:
    1. ArtifactIndex.add_artifact({problem, resolution, [outcome]}) -> digest.
       (BlobStore at phase8/blobs/, index at phase8/artifact_index_{dim}.jsonl,
        embedder = default_usefulness_embedder(dim).)
    2. Sprout the work-unit claim node under a bootstrap root tendency;
       set node.artifact_digest = digest.
    3. Sprout each judge sub-claim (from judge_cache) under the work-unit node
       with its PRO/CON position; set claim_node.artifact_digest = digest too.
    4. equilibrate(world) per unit (as usefulness_training does).

Each world is serialized with the repo serializer (snapshot_world; NEVER
sort_keys — it corrupts snapshots) to phase8/world_{dim}.json.

O(N^2)-slow (~15 s/unit, ~50 min for 200 units): per-unit progress is
printed, a periodic snapshot is written for resume, and the two dims run
sequentially.

Schemas built against (owned by a concurrent agent):
  corpus_sample.jsonl : {"uid","problem","resolution","outcome"}
  judge_cache.jsonl   : {"uid","claims":[{"axis","position","text"}]}

Usage:
    python build_worlds.py                       # both dims (256 then 64)
    python build_worlds.py --dim 64              # one dim
    python build_worlds.py --dim 64 --limit 15   # smoke: first 15 units
    python build_worlds.py --corpus sandbox/corpus_sample.jsonl \
        --judge sandbox/judge_cache.jsonl --outdir sandbox --dim 64 --limit 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- sys.path bootstrap to the autonet repo (as phase7) --------------------
_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

from world_model.generalized import (  # type: ignore  # noqa: E402
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.generalized.serialize import snapshot_world  # type: ignore  # noqa: E402
from world_model.models.tree import Position  # type: ignore  # noqa: E402

from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    _l2_normalize,
    coords_for_problem_resolution,
    default_usefulness_embedder,
)
from nodes.common.world_model_substrate.artifact_index import (  # type: ignore  # noqa: E402
    ArtifactIndex,
)
from nodes.common.blob_store import BlobStore  # type: ignore  # noqa: E402


HERE = Path(__file__).resolve().parent

DIMS = (256, 64)
BANDWIDTH = 0.5             # matches build_usefulness_world default
EQUILIBRATE_ROUNDS = 4
SNAPSHOT_EVERY = 10         # write a resume snapshot every N units


# ---------------------------------------------------------------------------
# Loaders (build against the concurrent agent's schemas)
# ---------------------------------------------------------------------------


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    """corpus rows: {"uid","problem","resolution","outcome"}."""
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_judge(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """judge rows: {"uid","claims":[{"axis","position","text"}]}.

    Returns uid -> list of claim dicts. Missing file => empty (no claims).
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["uid"]] = row.get("claims", []) or []
    return out


# ---------------------------------------------------------------------------
# Position parsing (judge claims carry "PRO"/"CON" text)
# ---------------------------------------------------------------------------


def _parse_position(raw: Any) -> Position:
    s = str(raw or "").strip().lower()
    if s in ("con", "against", "negative", "-1", "no"):
        return Position.CON
    return Position.PRO  # default PRO (matches usefulness path's PRO default)


# ---------------------------------------------------------------------------
# World bootstrap (two utility roots, NOT the charter — usefulness_training)
# ---------------------------------------------------------------------------


def build_bootstrap_world(dim: int) -> World:
    world = World()
    pos_anchor = tuple([1.0] + [0.0] * (dim - 1))
    neg_anchor = tuple([-1.0] + [0.0] * (dim - 1))
    world.add_tendency(GeneralizedTendency(
        id="good_resolution",
        thesis="This approach worked for this kind of problem.",
        anchor=pos_anchor, polarity_axis=pos_anchor,
        budget=1.0, bandwidth=BANDWIDTH, smooth_promotion=True,
    ))
    world.add_tendency(GeneralizedTendency(
        id="novel_resolution",
        thesis="This approach is new to the network.",
        anchor=neg_anchor, polarity_axis=neg_anchor,
        budget=1.0, bandwidth=BANDWIDTH, smooth_promotion=True,
    ))
    return world


def _best_root(world: World, coords) -> Optional[GeneralizedTendency]:
    best = None
    best_dot = -2.0
    for tendency in world.tendencies.values():
        if not tendency.anchor:
            continue
        n = min(len(coords), len(tendency.anchor))
        dot = sum(coords[i] * tendency.anchor[i] for i in range(n))
        if dot > best_dot:
            best_dot = dot
            best = tendency
    return best


def _outcome_pro_signal(outcome: Any) -> float:
    """Best-effort positive signal from an outcome payload.

    outcome may be a dict (accepted/kept/...), a number, or absent. Positive
    => PRO work-unit node, else CON (mirrors usefulness_training's pos_signal).
    """
    if outcome is None:
        return 1.0
    if isinstance(outcome, (int, float)):
        return float(outcome)
    if isinstance(outcome, dict):
        return float(outcome.get("accepted", 0)) + float(outcome.get("kept", 0)) \
            + float(outcome.get("built_on", 0)) + float(outcome.get("paid", 0))
    return 1.0


# ---------------------------------------------------------------------------
# Build one world for a given dim
# ---------------------------------------------------------------------------


def _collect_digests(world: World) -> Dict[str, str]:
    """node_id -> artifact_digest for every node carrying one.

    Read via getattr (the attribute is set post-sprout and absent on most
    nodes). Restored worlds re-attach these onto their nodes so
    _infer_artifacts can price artifacts by equilibrated standing.
    """
    out: Dict[str, str] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            digest = getattr(node, "artifact_digest", "")
            if digest:
                out[node.id] = digest
    return out


def build_world_for_dim(
    corpus: List[Dict[str, Any]],
    judge: Dict[str, List[Dict[str, Any]]],
    dim: int,
    outdir: Path,
    limit: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> Path:
    log = log or logging.getLogger("build_worlds")
    units = corpus[:limit] if limit else corpus
    n_units = len(units)

    blobs_dir = outdir / "blobs"
    index_path = outdir / f"artifact_index_{dim}.jsonl"
    world_path = outdir / f"world_{dim}.json"
    # Digest sidecar: the repo serializer (snapshot_world) does NOT persist the
    # custom node.artifact_digest attribute, so restore_world would drop it and
    # _infer_artifacts would find zero standing (D collapses to cosine-only —
    # exactly the phase-6 silent-empty-context failure). We persist a
    # node_id -> artifact_digest map next to the snapshot; run_contest.py
    # re-attaches it after restore. This keeps the committed serializer
    # unmodified (no hand-rolled world json, no sort_keys).
    digests_path = outdir / f"world_{dim}.digests.json"
    snapshot_path = outdir / f"world_{dim}.snapshot.json"
    progress_path = outdir / f"build_{dim}_progress.json"

    # Fresh index for this dim (the blob store is content-addressed and
    # shared safely; the per-dim index must not mix embeddings of two dims).
    if index_path.exists():
        index_path.unlink()

    embedder = default_usefulness_embedder(dim=dim)
    store = BlobStore(data_dir=str(blobs_dir))
    index = ArtifactIndex(store, embedder=embedder, index_path=index_path, dim=dim)

    world = build_bootstrap_world(dim)

    log.info("dim=%d: building world from %d units -> %s", dim, n_units, world_path)
    started = time.time()

    for i, unit in enumerate(units, start=1):
        uid = unit["uid"]
        problem = unit.get("problem", "") or ""
        resolution = unit.get("resolution", "") or ""
        outcome = unit.get("outcome")

        # --- data plane: full payload -> artifact index -> digest ---
        payload: Dict[str, Any] = {"problem": problem, "resolution": resolution}
        if outcome is not None:
            payload["outcome"] = outcome
        digest = index.add_artifact(payload)

        # --- consensus plane: work-unit observation + node ---
        coords = coords_for_problem_resolution(problem, resolution, embedder=embedder)
        obs = Observation(
            id=f"u_{uid}", coords=coords,
            label=f"uid={uid} | problem: {problem[:60]}",
        )
        world.add_observation(obs)

        root = _best_root(world, coords)
        pos_signal = _outcome_pro_signal(outcome)
        position = Position.PRO if pos_signal >= 0 else Position.CON
        axis_list = list(coords)
        axis = _l2_normalize(axis_list) if any(c != 0 for c in axis_list) \
            else root.polarity_axis
        work_node = root.sprout_child(
            parent_node_id=root.tree.root_node.id,
            position=position,
            anchor=coords,
            polarity_axis=tuple(axis),
            observation=obs,
            content=problem[:80],
            world=world,
        )
        # Price the artifact on the work-unit node.
        work_node.artifact_digest = digest

        # --- judge sub-claims: sprout under the work-unit node ---
        for claim in judge.get(uid, []):
            text = (claim.get("text") or "").strip()
            if not text:
                continue
            claim_pos = _parse_position(claim.get("position"))
            claim_coords = coords_for_problem_resolution(text, "", embedder=embedder)
            c_axis_list = list(claim_coords)
            c_axis = _l2_normalize(c_axis_list) if any(c != 0 for c in c_axis_list) \
                else root.polarity_axis
            try:
                claim_node = root.sprout_child(
                    parent_node_id=work_node.id,
                    position=claim_pos,
                    anchor=claim_coords,
                    polarity_axis=tuple(c_axis),
                    content=text[:200],
                    world=world,
                )
                # Same digest so the verdict prices the same artifact.
                claim_node.artifact_digest = digest
            except Exception as e:  # noqa: BLE001
                log.warning("dim=%d uid=%s: judge claim sprout failed: %s", dim, uid, e)

        # Equilibrate per unit (as usefulness_training does).
        equilibrate(world, max_rounds=EQUILIBRATE_ROUNDS, tolerance=1e-3)

        elapsed = time.time() - started
        rate = elapsed / i
        eta = rate * (n_units - i)
        n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
        print(
            f"  dim={dim} [{i:>3}/{n_units}] uid={uid[:12]:<12} "
            f"nodes={n_nodes:<5} arts={len(index):<4} "
            f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
            flush=True,
        )

        # Periodic resume snapshot.
        if i % SNAPSHOT_EVERY == 0 or i == n_units:
            snap = snapshot_world(world)  # NEVER sort_keys.
            snapshot_path.write_text(json.dumps(snap), encoding="utf-8")
            (outdir / f"world_{dim}.snapshot.digests.json").write_text(
                json.dumps(_collect_digests(world)), encoding="utf-8")
            progress_path.write_text(
                json.dumps({"dim": dim, "completed": i, "total": n_units,
                            "ts": time.time()}),
                encoding="utf-8",
            )

    # Final serialize + digest sidecar.
    snap = snapshot_world(world)  # NEVER sort_keys — corrupts snapshots.
    world_path.write_text(json.dumps(snap), encoding="utf-8")
    digest_map = _collect_digests(world)
    digests_path.write_text(json.dumps(digest_map), encoding="utf-8")
    log.info("dim=%d: wrote digest sidecar (%d priced nodes) -> %s",
             dim, len(digest_map), digests_path)
    log.info("dim=%d: wrote %s (%d units, %.0fs)", dim, world_path, n_units,
             time.time() - started)
    return world_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(HERE / "corpus_sample.jsonl"))
    parser.add_argument("--judge", default=str(HERE / "judge_cache.jsonl"))
    parser.add_argument("--outdir", default=str(HERE))
    parser.add_argument("--dim", type=int, default=None,
                        help="single dim; omit to build both (256 then 64)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N corpus units (smoke)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("nodes.common.blob_store").setLevel(logging.ERROR)
    log = logging.getLogger("build_worlds")

    corpus_path = Path(args.corpus)
    judge_path = Path(args.judge)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        log.error("corpus not found: %s", corpus_path)
        return 1

    corpus = load_corpus(corpus_path)
    judge = load_judge(judge_path)
    log.info("loaded %d corpus units, %d units with judge claims",
             len(corpus), len(judge))

    dims: Tuple[int, ...] = (args.dim,) if args.dim else DIMS
    for dim in dims:
        build_world_for_dim(corpus, judge, dim, outdir, limit=args.limit, log=log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
