#!/usr/bin/env python3
"""Test usefulness coordinate space."""

from __future__ import annotations

import math
import sys

from nodes.common.world_model_substrate.usefulness_coords import (
    HashingEmbedder,
    coords_for_problem_resolution,
    coords_for_query,
    default_usefulness_embedder,
)


def cos_sim(a: tuple, b: tuple) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main() -> int:
    cases_passed = 0
    cases_total = 0

    embedder = HashingEmbedder(dim=16, seed=42)

    # 1. Determinism
    cases_total += 1
    a = embedder("fix bug in authentication module")
    b = embedder("fix bug in authentication module")
    if a == b:
        cases_passed += 1
        print("[PASS] determinism")
    else:
        print(f"[FAIL] determinism: {a} != {b}")

    # 2. L2 normalization (after _l2_normalize, magnitudes should be 1)
    cases_total += 1
    norm = math.sqrt(sum(x * x for x in a))
    if abs(norm - 1.0) < 1e-6:
        cases_passed += 1
        print(f"[PASS] L2-normalized (norm={norm:.4f})")
    else:
        print(f"[FAIL] L2-normalized: norm={norm}")

    # 3. Empty input -> zero vector
    cases_total += 1
    z = embedder("")
    if all(x == 0.0 for x in z):
        cases_passed += 1
        print("[PASS] empty input zero vector")
    else:
        print(f"[FAIL] empty input: {z}")

    # 4. Same topic -> high cos sim
    cases_total += 1
    auth1 = embedder("fix authentication bug oauth login flow")
    auth2 = embedder("repair the oauth login flow in authentication")
    s = cos_sim(auth1, auth2)
    if s > 0.2:
        cases_passed += 1
        print(f"[PASS] same topic similarity: {s:.3f}")
    else:
        print(f"[FAIL] same topic similarity too low: {s:.3f}")

    # 5. Different topics -> lower cos sim
    cases_total += 1
    auth = embedder("fix authentication bug oauth login flow")
    db = embedder("optimize database query performance for reporting")
    s = cos_sim(auth, db)
    auth_self = cos_sim(auth, auth)
    if s < auth_self:
        cases_passed += 1
        print(f"[PASS] different topics less similar than self: {s:.3f} < {auth_self:.3f}")
    else:
        print(f"[FAIL] cross-topic too similar: {s:.3f}")

    # 6. coords_for_problem_resolution end-to-end
    cases_total += 1
    coords = coords_for_problem_resolution(
        problem="fix the bug in oauth callback",
        resolution="changed redirect_uri validation in handler",
        embedder=embedder,
    )
    if len(coords) == 16 and any(c != 0 for c in coords):
        cases_passed += 1
        print(f"[PASS] coords_for_problem_resolution shape and content")
    else:
        print(f"[FAIL] coords_for_problem_resolution: {coords}")

    # 7. coords_for_query
    cases_total += 1
    qcoords = coords_for_query("oauth bug", embedder=embedder)
    if len(qcoords) == 16:
        cases_passed += 1
        print("[PASS] coords_for_query shape")
    else:
        print(f"[FAIL] coords_for_query: {qcoords}")

    # 8. default_usefulness_embedder returns something usable
    cases_total += 1
    emb = default_usefulness_embedder(dim=16)
    out = emb("test input")
    if len(out) == 16:
        cases_passed += 1
        print(f"[PASS] default factory returns dim-16 vector via {type(emb).__name__}")
    else:
        print(f"[FAIL] default factory shape: {out}")

    print(f"\n{cases_passed}/{cases_total} cases passed")
    return 0 if cases_passed == cases_total else 1


if __name__ == "__main__":
    sys.exit(main())
