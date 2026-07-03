#!/usr/bin/env python3
"""Phase 7 contest runner — the corrected phase 6 rerun.

Phase 6 (phase6/run_contest.py:290) called infer_with_world_model WITHOUT
passing the built world, so the "substrate" arm was silently bare haiku
(probe_region_size was 0 on every row). Phase 7 uses the new two-plane
inference path (docs/two_plane_inference.md): artifacts live in a blob
store + embedding index, claim-graph nodes are debated verdicts about
them, and retrieval is embedding similarity re-ranked by graph standing.
The world is threaded through explicitly.

Three arms per held-out test problem (N=10, all tendencies):
  1. none      — no context (explicit baseline; what phase 6 measured)
  2. rag       — top-k by embedding + each entry's judge claims (fair)
  3. substrate — infer_with_world_model(mode="artifacts", world=..., ...)

Scores via phase 5's doctest harness. Writes:
  - contest7_N10.jsonl     (one row per problem, all three arms)
  - contest7_progress.jsonl (appended; mid-run readable)
  - aggregate7.json        (per-arm means + pairwise deltas)

Usage:
    python run_contest.py --out contest7_N10.jsonl          # real run
    python run_contest.py --mock --out contest7_mock.jsonl  # offline smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

# --- sys.path bootstrapping (copied from phase6) --------------------------
_AUTONET = Path(r"C:\code\autonet")
if str(_AUTONET) not in sys.path:
    sys.path.insert(0, str(_AUTONET))

sys.path.insert(0, str(Path(
    r"D:\videos\SF\manifesting\from_endstate\new physics\substrate_experiment\phase5"
)))

from world_model.generalized import (  # type: ignore  # noqa: E402
    GeneralizedTendency, Observation, World, equilibrate,
)
from world_model.models.tree import Position  # type: ignore  # noqa: E402

from nodes.common.world_model_substrate.usefulness_coords import (  # type: ignore  # noqa: E402
    default_usefulness_embedder,
)
from nodes.common.world_model_substrate.infer import (  # type: ignore  # noqa: E402
    infer_with_world_model,
)
from nodes.common.world_model_substrate.artifact_index import (  # type: ignore  # noqa: E402
    ArtifactIndex,
)
from nodes.common.blob_store import BlobStore  # type: ignore  # noqa: E402
from atn.providers.bridge import BridgeProvider  # type: ignore  # noqa: E402

from doctest_harness import grade_implementation  # type: ignore  # noqa: E402


HERE = Path(__file__).resolve().parent
# Corpus + judge cache are reused from phase6 (do NOT regenerate).
PHASE6 = HERE.parent / "phase6"
CORPUS_PATH = PHASE6 / "corpus.json"
CACHE_PATH = PHASE6 / "judge_cache.jsonl"

PROGRESS_PATH = HERE / "contest7_progress.jsonl"
AGGREGATE_PATH = HERE / "aggregate7.json"
BLOBS_DIR = HERE / "blobs"
INDEX_PATH = HERE / "artifact_index.jsonl"


TENDENCIES = [
    "correctness", "simplicity", "robustness", "purity",
    "laziness", "composability", "type_flexibility", "error_clarity",
    "efficiency", "documentation_fidelity",
]

N = 10  # phase 7 runs the full tendency set only.

EMBEDDING_DIM = 64
BANDWIDTH = 1.5
TOP_K = 5
PROBE_K = 5
CONTEXT_CHAR_CAP = 6000  # cap total context chars for rag + substrate


CONTESTANT_SYSTEM = """You are completing the body of a Python function.

You will be given:
  - A function signature with a sparse docstring containing 1-2 examples.
  - Optionally, related material to consult.

Your job: write ONLY the function body that satisfies the docstring's examples.

Rules:
  - Output ONLY the Python source for the function (def ...:\\n    body).
  - Do NOT include explanatory text, markdown, or commentary.
  - The function must be named exactly as the signature specifies.
  - Use only Python stdlib (and `itertools`).
"""


# ---------------------------------------------------------------------------
# Corpus / cache loading (reused from phase6)
# ---------------------------------------------------------------------------


def load_cache() -> Dict[str, Dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out[row["name"]] = row
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def coords_for_text(text: str, n_axes: int, embedder, axis_idx: int = -1) -> Tuple[float, ...]:
    tail = embedder(text)
    tail_t = tuple(float(x) for x in tail)[:EMBEDDING_DIM]
    if len(tail_t) < EMBEDDING_DIM:
        tail_t = tail_t + (0.0,) * (EMBEDDING_DIM - len(tail_t))
    head = tuple(1.0 if i == axis_idx else 0.0 for i in range(n_axes))
    return head + tail_t


# ---------------------------------------------------------------------------
# Substrate build (two-plane: claim graph verdicts + artifact index)
# ---------------------------------------------------------------------------


def build_world(
    corpus: Dict[str, Any], embedder,
) -> Tuple[World, Dict[str, Any], ArtifactIndex, BlobStore]:
    """Build the N=10 substrate with the two-plane data layer.

    Returns (world, train_index, artifact_index, blob_store).
    train_index maps name -> {obs_node_id, embedding (tail), digest, entry}.
    """
    active = TENDENCIES[:N]
    total_dim = N + EMBEDDING_DIM
    world = World()
    for i, tid in enumerate(active):
        anchor = tuple(1.0 if j == i else 0.0 for j in range(total_dim))
        world.add_tendency(GeneralizedTendency(
            id=tid, thesis=f"Charter axis: {tid}",
            anchor=anchor, polarity_axis=anchor, budget=1.0,
            bandwidth=BANDWIDTH, smooth_promotion=True,
        ))

    # Data plane: blob store + embedding index (daemon-local, rebuildable).
    store = BlobStore(data_dir=str(BLOBS_DIR))
    index = ArtifactIndex(store, index_path=INDEX_PATH, dim=EMBEDDING_DIM)

    cache = load_cache()
    train = corpus["train"]
    train_index: Dict[str, Dict[str, Any]] = {}

    for entry in train:
        name = entry["name"]
        docstring = entry.get("docstring", "")
        impl = entry.get("impl_full_source", "")

        # --- data plane: full payload -> artifact index -> digest ---
        payload = {
            "problem": f"{name}\n\n{docstring}",
            "resolution": impl,
            "text": name,
        }
        digest = index.add_artifact(payload)

        # --- consensus plane: observation node (as phase6) ---
        text = f"{name}\n\n{docstring}\n\n{impl}"
        obs_coords = coords_for_text(text, N, embedder, axis_idx=-1)
        obs = Observation(id=f"f_{name}", coords=obs_coords, label=name)
        world.add_observation(obs)

        sprouter = world.tendencies[active[0]]
        obs_node = sprouter.sprout_child(
            parent_node_id=sprouter.tree.root_node.id,
            position=Position.PRO,
            anchor=obs_coords,
            polarity_axis=sprouter.polarity_axis,
            observation=obs,
            content=name,
            world=world,
        )
        # Price the artifact: set digest on the work-unit node.
        obs_node.artifact_digest = digest

        tail_t = obs_coords[N:]
        train_index[name] = {
            "obs_node_id": obs_node.id,
            "embedding": tail_t,
            "digest": digest,
            "entry": entry,
        }

        # --- judge sub-claims: sprout under work-unit, price the artifact ---
        parsed = (cache.get(name) or {}).get("parsed") or {}
        for axis_idx, axis in enumerate(active):
            for claim_text in parsed.get(axis, []) or []:
                if not claim_text:
                    continue
                claim_coords = coords_for_text(claim_text, N, embedder, axis_idx=axis_idx)
                try:
                    claim_node = world.tendencies[axis].sprout_child(
                        parent_node_id=obs_node.id,
                        position=Position.PRO,
                        anchor=claim_coords,
                        polarity_axis=world.tendencies[axis].polarity_axis,
                        content=claim_text,
                        world=world,
                    )
                    # Same digest so verdicts price the same artifact.
                    claim_node.artifact_digest = digest
                except Exception:
                    pass

    equilibrate(world, max_rounds=4, tolerance=1e-3)
    return world, train_index, index, store


# ---------------------------------------------------------------------------
# RAG arm (fair: includes judge claims as distilled knowledge)
# ---------------------------------------------------------------------------


def topk_by_embedding(test_entry, train_index, embedder, k=TOP_K) -> List[Dict[str, Any]]:
    q_tail = embedder(f"{test_entry['name']}\n\n{test_entry['sparse_docstring']}")
    q_arr = np.array(q_tail, dtype=np.float32)
    q_norm = np.linalg.norm(q_arr) + 1e-9
    scored: List[Tuple[float, str]] = []
    for name, idx in train_index.items():
        e_arr = np.array(idx["embedding"], dtype=np.float32)
        cos = float(np.dot(q_arr, e_arr) / (q_norm * (np.linalg.norm(e_arr) + 1e-9)))
        scored.append((cos, name))
    scored.sort(reverse=True)
    return [{"name": n, "similarity": s} for s, n in scored[:k]]


def _judge_claims_text(name: str, cache: Dict[str, Dict[str, Any]]) -> List[str]:
    """Flatten a cached entry's judge sub-claims (all axes) into text lines."""
    parsed = (cache.get(name) or {}).get("parsed") or {}
    lines: List[str] = []
    for axis in TENDENCIES:
        for claim in parsed.get(axis, []) or []:
            if claim:
                lines.append(f"  - [{axis}] {claim}")
    return lines


def format_rag_context(
    top: List[Dict[str, Any]],
    train_by_name: Dict[str, Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
) -> str:
    lines = ["RELATED FUNCTIONS (most similar), with vetted notes:", ""]
    for item in top:
        e = train_by_name[item["name"]]
        lines.append(f"--- {e['name']}({e['signature']}) ---")
        lines.append(e.get("docstring", ""))
        lines.append("")
        lines.append(e.get("impl_full_source", ""))
        claims = _judge_claims_text(e["name"], cache)
        if claims:
            lines.append("")
            lines.append("Notes:")
            lines.extend(claims)
        lines.append("")
    ctx = "\n".join(lines)
    return ctx[:CONTEXT_CHAR_CAP]


# ---------------------------------------------------------------------------
# Substrate arm (two-plane artifact retrieval)
# ---------------------------------------------------------------------------


def format_substrate_context(probe: Dict[str, Any]) -> str:
    lines = ["VETTED ARTIFACTS (substrate probe), most relevant first:", ""]
    for i, art in enumerate(probe.get("artifacts", [])):
        payload = art.get("payload") or {}
        problem = payload.get("problem") or ""
        resolution = payload.get("resolution") or ""
        standing = art.get("standing", 0.0)
        final = art.get("final", 0.0)
        cosine = art.get("cosine", 0.0)
        lines.append(
            f"--- artifact [{i}]  (cosine={cosine:.3f} standing={standing:.2f} "
            f"final={final:.3f}) ---"
        )
        lines.append(problem)
        lines.append("")
        lines.append(resolution)
        claims = art.get("claims") or []
        if claims:
            lines.append("")
            lines.append("Vetted notes (net_score / position):")
            for c in claims:
                content = c.get("content", "")
                pos = c.get("position", "")
                ns = c.get("net_score", 0.0)
                lines.append(f"  - [{pos} {ns:+.2f}] {content}")
        lines.append("")
    ctx = "\n".join(lines)
    return ctx[:CONTEXT_CHAR_CAP]


def substrate_probe_meta(probe: Dict[str, Any]) -> Dict[str, Any]:
    """Per-artifact metadata for the output row."""
    arts = probe.get("artifacts", [])
    return {
        "region_size": len(arts),
        "artifacts": [
            {
                "digest": a.get("digest"),
                "cosine": a.get("cosine"),
                "standing": a.get("standing"),
                "final": a.get("final"),
                "payload_present": a.get("payload") is not None,
                "n_claims": len(a.get("claims") or []),
            }
            for a in arts
        ],
    }


# ---------------------------------------------------------------------------
# Prompts / parsing (from phase6)
# ---------------------------------------------------------------------------


def build_user_prompt(test_entry: Dict[str, Any], context: str = "") -> str:
    prompt = (
        f"Implement this function:\n\n"
        f"def {test_entry['name']}({test_entry['signature']}):\n"
        f'    """{test_entry["sparse_docstring"]}"""\n'
        f"    # YOUR CODE HERE\n"
    )
    if context:
        prompt = context + "\n\n" + prompt
    prompt += "\nReturn ONLY the complete function definition. No prose, no markdown fences."
    return prompt


def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Contestant call (real bridge, or deterministic mock)
# ---------------------------------------------------------------------------


def mock_impl(name: str) -> str:
    """Deterministic offline stub — scores 0, exercises the pipeline."""
    return f"def {name}(*args, **kwargs):\n    raise NotImplementedError()"


async def call_contestant(
    provider: Optional[BridgeProvider], test_entry: Dict[str, Any], user: str, mock: bool,
) -> Tuple[str, float]:
    started = time.time()
    if mock:
        return mock_impl(test_entry["name"]), 0.0
    assert provider is not None
    try:
        result = await provider.send(
            messages=[{"role": "user", "content": user}],
            system=CONTESTANT_SYSTEM,
            model="haiku",
        )
        text = result.text or ""
    except Exception as e:
        text = f"BRIDGE_ERROR: {type(e).__name__}: {e}"
    return text, time.time() - started


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


ARMS = ("none", "rag", "substrate")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mock", action="store_true",
                        help="deterministic offline stub; zero bridge/API calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    # blob_store logs at INFO on every add; silence it (per-artifact noise).
    logging.getLogger("nodes.common.blob_store").setLevel(logging.ERROR)

    # Fresh data-plane state each run so the smoke is reproducible.
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    test_entries = corpus["test"]
    train_by_name = {e["name"]: e for e in corpus["train"]}
    cache = load_cache()
    print(f"  N = {N}, test entries: {len(test_entries)}, mock={args.mock}")

    embedder = default_usefulness_embedder(dim=EMBEDDING_DIM)
    world, train_index, index, store = build_world(corpus, embedder)
    n_nodes = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    print(f"  substrate built: {n_nodes} nodes, {len(index)} artifacts indexed")

    provider: Optional[BridgeProvider] = None if args.mock else BridgeProvider(model="haiku")

    rows: List[Dict[str, Any]] = []
    started = time.time()

    # Smoke-invariant tracking.
    probe_min_arts = None
    probe_ctx_lens: List[int] = []
    probe_invariant_failed = False
    sample_ctx_printed = False

    try:
        for i, t in enumerate(test_entries, start=1):
            row: Dict[str, Any] = {
                "n": N,
                "name": t["name"],
                "signature": t["signature"],
                "sparse_docstring": t["sparse_docstring"],
                "doctests": t["doctests"],
            }

            # --- arm 1: none (no context) ---
            impl_none, dt_none = await call_contestant(
                provider, t, build_user_prompt(t, ""), args.mock)
            row["none"] = {"impl": strip_code_fences(impl_none), "raw": impl_none,
                           "elapsed_s": dt_none}

            # --- arm 2: rag (fair — includes judge claims) ---
            top = topk_by_embedding(t, train_index, embedder)
            rag_ctx = format_rag_context(top, train_by_name, cache)
            impl_rag, dt_rag = await call_contestant(
                provider, t, build_user_prompt(t, rag_ctx), args.mock)
            row["rag"] = {"impl": strip_code_fences(impl_rag), "raw": impl_rag,
                          "elapsed_s": dt_rag, "rag_top": top,
                          "context_chars": len(rag_ctx)}

            # --- arm 3: substrate (two-plane artifact retrieval) ---
            query_text = f"{t['name']} {t['sparse_docstring']}"
            try:
                probe = infer_with_world_model(
                    {"text": query_text},
                    mode="artifacts",
                    world=world,
                    artifact_index=index,
                    blob_store=store,
                    k=PROBE_K,
                )
            except Exception as e:
                probe = {"error": f"{type(e).__name__}: {e}", "artifacts": []}

            arts = probe.get("artifacts", [])
            # --- SMOKE INVARIANT: non-empty artifacts, non-None payloads ---
            n_arts = len(arts)
            payloads_ok = all(a.get("payload") is not None for a in arts)
            probe_min_arts = n_arts if probe_min_arts is None else min(probe_min_arts, n_arts)
            if n_arts == 0 or not payloads_ok:
                probe_invariant_failed = True
                print("  " + "!" * 68)
                print(f"  FAIL: substrate probe invariant violated for {t['name']!r}: "
                      f"n_artifacts={n_arts} payloads_ok={payloads_ok}")
                print("  This is the exact phase 6 bug (empty substrate context).")
                print("  " + "!" * 68)

            substrate_ctx = format_substrate_context(probe)
            probe_ctx_lens.append(len(substrate_ctx))
            if not sample_ctx_printed and n_arts > 0:
                print("\n  --- SAMPLE substrate context (first ~24 lines) ---")
                for ln in substrate_ctx.splitlines()[:24]:
                    print("  | " + ln)
                print("  --- end sample ---\n")
                sample_ctx_printed = True

            impl_sub, dt_sub = await call_contestant(
                provider, t, build_user_prompt(t, substrate_ctx), args.mock)
            row["substrate"] = {"impl": strip_code_fences(impl_sub), "raw": impl_sub,
                                "elapsed_s": dt_sub,
                                "context_chars": len(substrate_ctx),
                                "probe": substrate_probe_meta(probe)}

            # --- score all three arms ---
            for arm in ARMS:
                result = grade_implementation(
                    contestant_source=row[arm]["impl"],
                    fn_name=row["name"],
                    signature=row["signature"],
                    doctests=row["doctests"],
                )
                row[arm]["score"] = result["score"]
                row[arm]["passed"] = result["n_passed"]
                row[arm]["total"] = result["n_doctests"]
                row[arm]["compile_error"] = result["compile_error"]

            with PROGRESS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "n": N, "name": t["name"],
                    "none_score": row["none"]["score"],
                    "rag_score": row["rag"]["score"],
                    "substrate_score": row["substrate"]["score"],
                    "probe_region_size": row["substrate"]["probe"]["region_size"],
                    "ts": time.time(),
                }, ensure_ascii=False) + "\n")

            rows.append(row)
            elapsed = time.time() - started
            print(f"  [{i:>2}/{len(test_entries)}] {t['name']:>28}  "
                  f"none={row['none']['score']:.2f} rag={row['rag']['score']:.2f} "
                  f"sub={row['substrate']['score']:.2f}  "
                  f"probe={row['substrate']['probe']['region_size']}  ({elapsed:.0f}s)")
    finally:
        if provider is not None:
            try:
                await provider.close()
            except Exception:
                pass

    # --- write per-problem output ---
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- aggregate ---
    def mean(arm: str) -> float:
        return sum(r[arm]["score"] for r in rows) / max(len(rows), 1)

    means = {arm: mean(arm) for arm in ARMS}

    # Paired significance (ttest_rel like phase6), only if scipy present.
    def paired_t(a: str, b: str) -> Optional[Dict[str, float]]:
        try:
            from scipy import stats  # type: ignore
        except Exception:
            return None
        xa = [r[a]["score"] for r in rows]
        xb = [r[b]["score"] for r in rows]
        if len(xa) < 2:
            return None
        t, p = stats.ttest_rel(xa, xb)
        return {"t": float(t), "p": float(p)}

    significance = {
        "substrate_vs_none": paired_t("substrate", "none"),
        "substrate_vs_rag": paired_t("substrate", "rag"),
        "rag_vs_none": paired_t("rag", "none"),
    }
    scipy_available = any(v is not None for v in significance.values())

    # Flag bridge errors and empty-artifact problems.
    bridge_errors = [
        {"name": r["name"], "arm": arm}
        for r in rows for arm in ARMS
        if str(r[arm].get("raw", "")).startswith("BRIDGE_ERROR")
    ]
    empty_artifact_problems = [
        r["name"] for r in rows
        if r["substrate"]["probe"]["region_size"] == 0
        or not all(a["payload_present"] for a in r["substrate"]["probe"]["artifacts"])
    ]

    aggregate = {
        "n": N,
        "mock": args.mock,
        "n_problems": len(rows),
        "means": means,
        "deltas": {
            "substrate_minus_none": means["substrate"] - means["none"],
            "substrate_minus_rag": means["substrate"] - means["rag"],
            "rag_minus_none": means["rag"] - means["none"],
        },
        "significance_ttest_rel": significance,
        "scipy_available": scipy_available,
        "bridge_errors": bridge_errors,
        "empty_artifact_problems": empty_artifact_problems,
        "probe": {
            "min_artifacts_per_query": probe_min_arts,
            "mean_context_chars": (sum(probe_ctx_lens) / len(probe_ctx_lens))
                                   if probe_ctx_lens else 0.0,
            "invariant_failed": probe_invariant_failed,
        },
    }
    AGGREGATE_PATH.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print()
    print(f"  means: none={means['none']:.3f}  rag={means['rag']:.3f}  "
          f"substrate={means['substrate']:.3f}")
    print(f"  deltas: sub-none={aggregate['deltas']['substrate_minus_none']:+.3f}  "
          f"sub-rag={aggregate['deltas']['substrate_minus_rag']:+.3f}  "
          f"rag-none={aggregate['deltas']['rag_minus_none']:+.3f}")
    print(f"  probe: min_artifacts_per_query={probe_min_arts}  "
          f"mean_context_chars={aggregate['probe']['mean_context_chars']:.0f}")
    if scipy_available:
        for k, v in significance.items():
            if v is not None:
                print(f"  ttest_rel {k}: t={v['t']:+.3f} p={v['p']:.4f}")
    else:
        print("  (scipy not available — paired significance skipped)")
    if bridge_errors:
        print(f"  BRIDGE ERRORS: {len(bridge_errors)} -> {bridge_errors}")
    if empty_artifact_problems:
        print(f"  EMPTY-ARTIFACT PROBLEMS: {empty_artifact_problems}")

    if probe_invariant_failed:
        print("\n  ****** SMOKE FAILED: substrate probe invariant violated ******")
        return 1
    print("\n  substrate probe invariant held on all problems.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
