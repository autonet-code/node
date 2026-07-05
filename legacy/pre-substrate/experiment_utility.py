#!/usr/bin/env python3
"""Utility experiment: does the substrate's locate beat baselines?

Setup
-----

Synthetic work-unit corpus across 6 known categories, each with
10-20 problem/resolution templates. Train the substrate on N units,
then for held-out queries measure precision@K — is the substrate's
locate returning resolutions from the SAME category as the query.

Baselines
---------

  random     : pick K resolutions uniformly at random from training set.
  keyword    : KeywordLocator on raw problem/resolution text.
  substrate  : substrate locate after training on N work units.

Metric
------

  precision@5 = fraction of top-5 returned resolutions whose category
                matches the query's category.

We run at training sizes 50, 200, 500, 1000 to see whether the
substrate's precision climbs with scale.
"""

from __future__ import annotations

import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from world_model.generalized import (
    KeywordLocator,
    CoordinateLocator,
    World,
    equilibrate,
)
from world_model.models.tree import Position

from nodes.common.world_model_substrate import (
    HashingEmbedder,
    Outcome,
    aggregate_contributions,
    apply_events,
    build_usefulness_world,
    coords_for_query,
    train_world_model_on_usefulness,
)


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

CATEGORIES = {
    "auth": [
        ("oauth callback returns invalid_grant", "verified redirect_uri matches client config"),
        ("session token expires too quickly", "extended session ttl in auth middleware"),
        ("user can't sign in with social login", "fixed missing scope in oauth provider config"),
        ("password reset email never arrives", "configured smtp credentials in auth service"),
        ("two factor code rejected", "synced server clock with ntp for totp validation"),
        ("logout doesn't clear cookies", "cleared httpOnly cookie on logout endpoint"),
        ("auth header dropped on cors preflight", "added authorization to allowed headers"),
        ("jwt signature invalid after restart", "made jwt signing key persistent across restarts"),
        ("refresh token never rotates", "implemented refresh token rotation on use"),
        ("api accepts expired bearer token", "added exp claim validation in token middleware"),
        ("login redirect strips query params", "preserved query string through login redirect"),
        ("saml assertion rejected by idp", "regenerated metadata with current cert chain"),
    ],
    "perf": [
        ("api responses are slow above 100 rps", "added redis cache for hot read paths"),
        ("homepage takes 8 seconds to load", "lazy loaded below the fold images"),
        ("database query timeouts under load", "added composite index on filter columns"),
        ("memory grows unbounded over time", "fixed retain cycle in event listener"),
        ("cpu usage spikes during cron", "moved heavy job to background queue"),
        ("ssr renders are slow", "cached compiled templates in memory"),
        ("websocket pings drop under traffic", "increased nginx proxy buffer size"),
        ("search latency over 2 seconds", "moved search index to elasticsearch"),
        ("pagination on large tables is slow", "switched to cursor pagination"),
        ("startup time is 30 seconds", "lazy initialized heavy dependencies"),
        ("mobile app feels sluggish", "deferred non critical javascript"),
        ("graphql queries n+1 problem", "added dataloader for nested resolvers"),
    ],
    "test": [
        ("flaky test in ci pipeline", "added retry with exponential backoff"),
        ("test coverage below 50 percent", "added unit tests for service layer"),
        ("integration tests take 20 minutes", "parallelized test runs by suite"),
        ("snapshot tests break on minor changes", "switched to inline snapshots for clarity"),
        ("e2e tests fail on cold start", "added wait for database ready in setup"),
        ("mocks drift from real api", "generated mocks from openapi schema"),
        ("test data fixtures are stale", "regenerated fixtures from production sample"),
        ("unit tests rely on database", "isolated tests behind in memory adapter"),
        ("coverage report missing edge cases", "added property based tests with hypothesis"),
        ("ci runs sequentially", "ran linting and tests in parallel jobs"),
        ("flaky time dependent assertion", "froze time with mock library"),
        ("test fails only on monday", "fixed timezone assumption in date logic"),
    ],
    "deploy": [
        ("deployment fails after dependency update", "pinned versions in package lock"),
        ("rollback takes 15 minutes", "moved to blue green deployment"),
        ("config drift between environments", "centralized config in env specific yaml"),
        ("secrets exposed in container logs", "redirected secrets to vault driver"),
        ("docker image size is 2gb", "switched to multi stage build"),
        ("kubernetes pods restart loop", "increased memory limits and probe timeout"),
        ("cdn serves stale assets", "added cache busting hash to filenames"),
        ("dns propagation takes hours", "lowered ttl before cutover"),
        ("ssl cert expired in production", "automated renewal with letsencrypt"),
        ("load balancer health check flaps", "increased success threshold"),
        ("zero downtime release breaks websocket", "added connection draining"),
        ("staging diverges from production", "automated weekly sync of staging data"),
    ],
    "data": [
        ("schema migration locks production for hours", "added concurrent index creation"),
        ("data export takes 6 hours", "streamed export with cursor instead of in memory"),
        ("etl pipeline drops rows on error", "added per row error logging and continue"),
        ("analytics dashboard shows stale data", "fixed clock skew in event timestamps"),
        ("duplicates in deduplicated table", "added unique constraint and cleaned data"),
        ("backup restore takes too long", "switched to incremental snapshots"),
        ("query plan is suboptimal", "ran analyze on relevant tables"),
        ("foreign key violations on import", "imported in topological order"),
        ("partitioning slows down queries", "rebuilt partitions on better key"),
        ("large blob columns slow down listing", "moved blobs out of main table"),
        ("event sourcing replay diverges from snapshot", "fixed nondeterminism in handler"),
        ("data warehouse jobs miss sla", "moved heavy joins to staging tables"),
    ],
    "ui": [
        ("button click does nothing on safari", "added preventDefault on touchstart"),
        ("modal closes on accidental swipe", "added confirm dialog before close"),
        ("dropdown shows behind navbar", "fixed z-index stacking context"),
        ("input loses focus while typing", "fixed parent rerender on every keystroke"),
        ("checkbox state lost on refresh", "persisted form state in localstorage"),
        ("dark mode breaks images", "added dark mode aware image variants"),
        ("layout breaks on iphone se", "added small screen breakpoint"),
        ("tooltip clipped at viewport edge", "switched to floating ui placement"),
        ("animations stutter on low end devices", "respected prefers reduced motion"),
        ("text overflows in narrow columns", "added text wrapping rules"),
        ("focus trap broken in dialog", "added explicit focus handling"),
        ("keyboard navigation skips toggle", "added tabindex to interactive elements"),
    ],
}

ALL_CATEGORIES = sorted(CATEGORIES.keys())


def build_corpus() -> List[Tuple[str, str, str]]:
    """Returns flat list of (category, problem, resolution)."""
    out = []
    for cat, pairs in CATEGORIES.items():
        for problem, resolution in pairs:
            out.append((cat, problem, resolution))
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def baseline_random(
    train: List[Tuple[str, str, str]],
    query: Tuple[str, str, str],
    k: int,
    rng: random.Random,
) -> List[str]:
    """Return K random training categories."""
    sample = rng.sample(train, min(k, len(train)))
    return [t[0] for t in sample]


def baseline_keyword(
    train: List[Tuple[str, str, str]],
    query: Tuple[str, str, str],
    k: int,
) -> List[str]:
    """Score each training item by Jaccard overlap with the query
    problem; return top-K categories.
    """
    from nodes.common.world_model_substrate.usefulness_coords import _tokenize
    qtokens = set(_tokenize(query[1]))
    if not qtokens:
        return []
    scored: List[Tuple[float, str]] = []
    for cat, problem, resolution in train:
        ttokens = set(_tokenize(problem + " " + resolution))
        if not ttokens:
            continue
        overlap = len(qtokens & ttokens) / len(qtokens | ttokens)
        scored.append((overlap, cat))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:k]]


def baseline_substrate(
    train_world: World,
    train_index: Dict[str, str],   # node_id -> category
    query: str,
    k: int,
    embedder: Any,
    locator: CoordinateLocator,
) -> List[str]:
    """Use substrate locate to find similar training resolutions; map
    back to categories via train_index. Considers the located nodes'
    ancestors and descendants to weight the prediction (richer than
    direct leaf-only lookup).
    """
    qcoords = coords_for_query(query, embedder=embedder)
    region = locator(train_world, qcoords)
    # Score each category by both direct hits AND descendant hits.
    # A node whose subtree includes a category-tagged leaf supports
    # that category.
    cat_scores: Dict[str, float] = {}
    seen_nodes: set = set()
    for tendency_id, node_id, dist in region:
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        weight = 1.0 / (1.0 + dist)
        # Direct category for this node
        if node_id in train_index:
            cat_scores[train_index[node_id]] = (
                cat_scores.get(train_index[node_id], 0.0) + weight
            )
            continue
        # Walk descendants
        tendency = train_world.tendencies.get(tendency_id)
        if tendency is None:
            continue
        node = tendency.tree.get_node(node_id)
        if node is None:
            continue
        queue = list(node.pro_children) + list(node.con_children)
        while queue:
            child = queue.pop()
            if child.id in train_index:
                cat = train_index[child.id]
                cat_scores[cat] = cat_scores.get(cat, 0.0) + weight * 0.5
            queue.extend(child.pro_children)
            queue.extend(child.con_children)
    ranked = sorted(cat_scores.items(), key=lambda x: -x[1])
    return [cat for cat, _ in ranked[:k]]


# ---------------------------------------------------------------------------
# Substrate training for the experiment
# ---------------------------------------------------------------------------


def train_substrate(
    train: List[Tuple[str, str, str]],
    embedder: Any,
    dim: int,
) -> Tuple[World, Dict[str, str]]:
    """Build a substrate trained on the train set. Returns the world
    plus a node_id -> category index for top-K -> category mapping.

    The index is populated from the events emitted by the training
    function: each observation_added event has an obs_id; we tag each
    (obs_id, category) pair so we can resolve nodes back to categories.
    """
    work_units = []
    obs_id_to_cat: Dict[str, str] = {}

    from nodes.common.world_model_substrate.usefulness_training import _obs_id

    for cat, problem, resolution in train:
        outcome = Outcome(accepted=1.0, kept=1.0)
        work_units.append((problem, resolution, outcome))
        oid = _obs_id(problem, resolution)
        obs_id_to_cat[oid] = cat

    contribution, _metrics = train_world_model_on_usefulness(
        work_units=work_units,
        dim=dim,
        bandwidth=0.5,
        epochs=1,
        agent_id="exp",
        embedder=embedder,
    )

    # Build the live world from the events
    world = build_usefulness_world(dim=dim, bandwidth=0.5)
    apply_events(world, contribution["events"])

    # Map node_ids back to categories. Sub-claims sprouted under
    # observations carry observation_id; abstract sub-claims may not.
    node_id_to_cat: Dict[str, str] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            obs_id = getattr(node, "observation_id", None)
            if obs_id and obs_id in obs_id_to_cat:
                node_id_to_cat[node.id] = obs_id_to_cat[obs_id]

    return world, node_id_to_cat


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_at_size(
    corpus: List[Tuple[str, str, str]],
    n_train: int,
    n_eval: int,
    k: int,
    rng: random.Random,
    embedder: Any,
    dim: int,
) -> Dict[str, float]:
    """Run one configuration, return precision@k for each variant."""
    rng.shuffle(corpus)
    train = corpus[:n_train]
    eval_set = corpus[n_train:n_train + n_eval]

    if not train or not eval_set:
        return {"random": 0.0, "keyword": 0.0, "substrate": 0.0}

    # Train substrate once
    t0 = time.time()
    world, node_index = train_substrate(train, embedder=embedder, dim=dim)
    t_train = time.time() - t0
    locator = CoordinateLocator(max_distance=2.0, max_results=64)

    # Evaluate
    hits = {"random": 0, "keyword": 0, "substrate": 0}
    total = len(eval_set)

    t0 = time.time()
    for q in eval_set:
        cat = q[0]
        rand_cats = baseline_random(train, q, k, rng)
        if cat in rand_cats:
            hits["random"] += 1
        kw_cats = baseline_keyword(train, q, k)
        if cat in kw_cats:
            hits["keyword"] += 1
        sub_cats = baseline_substrate(
            world, node_index, q[1], k, embedder=embedder, locator=locator
        )
        if cat in sub_cats:
            hits["substrate"] += 1
    t_eval = time.time() - t0

    return {
        "random": hits["random"] / total,
        "keyword": hits["keyword"] / total,
        "substrate": hits["substrate"] / total,
        "n_train": n_train,
        "n_eval": total,
        "train_seconds": t_train,
        "eval_seconds": t_eval,
        "n_nodes": sum(len(t.tree.all_nodes()) for t in world.tendencies.values()),
    }


def main() -> int:
    print()
    print("=" * 70)
    print("UTILITY EXPERIMENT: substrate locate vs random / keyword baselines")
    print("=" * 70)

    corpus = build_corpus()
    print(f"\n  corpus: {len(corpus)} work units across {len(ALL_CATEGORIES)} categories")
    print(f"  categories: {', '.join(ALL_CATEGORIES)}")

    # Use sentence-transformers if available -- it's the real test
    # of the substrate's locate. Falls back to HashingEmbedder.
    dim = 16
    from nodes.common.world_model_substrate import default_usefulness_embedder
    embedder = default_usefulness_embedder(dim=dim)
    print(f"\n  embedder: {type(embedder).__name__} (dim={dim})")

    # Vary training size; eval set fixed at 20.
    rng = random.Random(7)
    train_sizes = [12, 24, 48, 60]   # cap by corpus size minus eval
    n_eval = 12
    k = 1   # strict: was the TOP returned resolution in the right category?
    n_seeds = 5  # average across seeds to smooth out small-eval-set variance

    print(f"\n  precision@{k} across training sizes (eval set = {n_eval}, "
          f"averaged over {n_seeds} seeds):\n")
    print(f"  {'n_train':>9}  {'nodes':>6}  {'random':>8}  {'keyword':>8}  {'substrate':>10}  {'train_s':>8}  {'eval_s':>8}")
    print(f"  {'-' * 70}")

    results = []
    for n_train in train_sizes:
        rand_acc = kw_acc = sub_acc = 0.0
        train_secs = eval_secs = 0.0
        n_nodes = 0
        for seed in range(n_seeds):
            rng_run = random.Random(7 + n_train * 100 + seed)
            corpus_copy = list(corpus)
            result = run_at_size(
                corpus_copy, n_train, n_eval, k, rng_run,
                embedder=embedder, dim=dim,
            )
            rand_acc += result["random"]
            kw_acc += result["keyword"]
            sub_acc += result["substrate"]
            train_secs += result["train_seconds"]
            eval_secs += result["eval_seconds"]
            n_nodes = result["n_nodes"]   # last is fine; deterministic by size
        rand_acc /= n_seeds
        kw_acc /= n_seeds
        sub_acc /= n_seeds
        train_secs /= n_seeds
        eval_secs /= n_seeds
        results.append({
            "n_train": n_train,
            "n_nodes": n_nodes,
            "random": rand_acc,
            "keyword": kw_acc,
            "substrate": sub_acc,
            "train_seconds": train_secs,
            "eval_seconds": eval_secs,
        })
        print(
            f"  {n_train:>9d}  {n_nodes:>6d}  "
            f"{rand_acc:>7.1%}  {kw_acc:>7.1%}  "
            f"{sub_acc:>9.1%}  "
            f"{train_secs:>7.2f}  {eval_secs:>7.2f}"
        )

    print()
    print("=" * 70)
    print("HONEST READING")
    print("=" * 70)

    last = results[-1]
    print(f"\n  At n_train={last['n_train']}:")
    print(f"    random     : {last['random']:.1%}")
    print(f"    keyword    : {last['keyword']:.1%}")
    print(f"    substrate  : {last['substrate']:.1%}")

    print(f"\n  What this says about the substrate at this scale:")

    if last["substrate"] >= last["keyword"]:
        print(f"  OK: substrate matches/beats keyword retrieval by "
              f"{(last['substrate'] - last['keyword']) * 100:.1f}pp.")
        print(f"      The graph is adding value over direct embedding lookup.")
    else:
        gap = (last["keyword"] - last["substrate"]) * 100
        print(f"  -- substrate is below keyword retrieval by {gap:.1f}pp.")
        print(f"      At this scale, the substrate's leaf-node graph is no")
        print(f"      better than a direct nearest-neighbor lookup. The")
        print(f"      structural depth that would distinguish it (sub-claims,")
        print(f"      cross-influence, training-agent debate) hasn't grown")
        print(f"      yet because we're running 1 epoch with no judges.")

    first = results[0]
    sub_scaling = last["substrate"] - first["substrate"]
    if sub_scaling > 0.02:
        print(f"\n  Substrate precision did climb with scale: "
              f"{first['substrate']:.1%} -> {last['substrate']:.1%} "
              f"(+{sub_scaling * 100:.1f}pp).")
    else:
        print(f"\n  Substrate precision is flat across scale "
              f"({first['substrate']:.1%} -> {last['substrate']:.1%}).")
        print(f"  More work units alone don't help when no depth is forming.")

    print(f"\n  What would change the substrate's number:")
    print(f"    - multi-epoch training so cross-influence can develop")
    print(f"    - judge-agent posts that build the graph's depth structure")
    print(f"    - inference query that walks ancestors+descendants, not")
    print(f"      just nearest leaves")
    print(f"    - real corpus larger than 72 work units, where structural")
    print(f"      generalization matters more than direct match")

    print(f"\n  Per-instance training cost: {last['train_seconds'] / last['n_train']:.3f}s.")
    print(f"  Per-instance eval cost: {last['eval_seconds'] / n_eval:.4f}s.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
