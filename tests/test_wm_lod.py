#!/usr/bin/env python3
"""Smoke test for the LOD scaffolding.

Validates the dual contract:
  - Epistemic:  reduce drops fields per the protocol's first-appearance map;
                expand introduces them with defaults.
  - Structural: every payload at LOD K serializes to exactly the level's
                budget_bytes, padded with NUL.

Uses trivial synthetic tendencies (not personality data) so the tests
exercise the LOD machinery itself, not domain-specific calibration.
"""

from __future__ import annotations

from world_model import (
    LODPayload,
    TrivialLODProtocol,
    Tendency,
    to_level,
)


def banner(text: str) -> None:
    print()
    print("-" * 60)
    print(text)
    print("-" * 60)


def make_full_synthetic(name: str, allocation: float) -> LODPayload:
    proto = TrivialLODProtocol()
    return LODPayload(
        level=proto.level(3),
        fields={
            "id": name,
            "allocation": allocation,
            "recent_validation_rate": 0.42,
            "birth_epoch": 7,
            "claim_mutations_count": 2,
            "claim": f"{name} matters because reasons",
            "description": f"the {name} drive",
            "metadata": {"origin": "synthetic"},
        },
    )


def test_strict_fixed_size_at_every_level() -> None:
    banner("test: every LOD payload serializes to exactly its budget")
    proto = TrivialLODProtocol()
    payload = make_full_synthetic("alpha", 0.5)
    for k in range(proto.max_level() + 1):
        p = to_level(payload, k, proto)
        raw = p.to_bytes()
        assert len(raw) == proto.level(k).budget_bytes, (
            f"LOD {k}: got {len(raw)} bytes, expected {proto.level(k).budget_bytes}"
        )
        print(f"  LOD {k} ({proto.level(k).name:9s}): "
              f"{len(raw):4d} bytes  fields={list(p.fields)}")


def test_round_trip_through_bytes() -> None:
    banner("test: pack then unpack preserves fields")
    proto = TrivialLODProtocol()
    payload = make_full_synthetic("beta", 0.25)
    p2 = to_level(payload, 2, proto)
    raw = p2.to_bytes()
    p2_back = LODPayload.from_bytes(raw, proto.level(2))
    assert p2_back.fields == p2.fields, f"mismatch: {p2.fields} vs {p2_back.fields}"
    print(f"  fields survived: {p2_back.fields}")


def test_reduce_is_deterministic_and_lossy() -> None:
    banner("test: reduce drops fields by the protocol's first-level map")
    proto = TrivialLODProtocol()
    payload = make_full_synthetic("gamma", 0.1)

    p3_to_p2 = proto.reduce(payload)
    assert p3_to_p2.level.level == 2
    assert "claim" not in p3_to_p2.fields
    assert "description" not in p3_to_p2.fields
    assert "metadata" not in p3_to_p2.fields
    assert "birth_epoch" in p3_to_p2.fields  # introduced at level 2

    p2_to_p1 = proto.reduce(p3_to_p2)
    assert p2_to_p1.level.level == 1
    assert "birth_epoch" not in p2_to_p1.fields
    assert "recent_validation_rate" in p2_to_p1.fields  # introduced at level 1

    p1_to_p0 = proto.reduce(p2_to_p1)
    assert p1_to_p0.level.level == 0
    assert "recent_validation_rate" not in p1_to_p0.fields
    assert set(p1_to_p0.fields) == {"id", "allocation"}

    # Reduce twice from same input -> identical output
    p3_to_p2_again = proto.reduce(payload)
    assert p3_to_p2.fields == p3_to_p2_again.fields
    print(f"  LOD 3 -> 2 -> 1 -> 0: {p1_to_p0.fields}")
    print(f"  reduce is deterministic: same input -> same output")


def test_expand_introduces_default_fields() -> None:
    banner("test: expand introduces missing fields with defaults")
    proto = TrivialLODProtocol()
    minimal = LODPayload(
        level=proto.level(0),
        fields={"id": "delta", "allocation": 0.3},
    )
    p1 = proto.expand(minimal)
    assert p1.level.level == 1
    assert "recent_validation_rate" in p1.fields
    assert p1.fields["recent_validation_rate"] == 0.0  # default

    # Expand with context override
    p1b = proto.expand(minimal, context={"recent_validation_rate": 0.7})
    assert p1b.fields["recent_validation_rate"] == 0.7

    # Walk up the ladder
    p3 = to_level(minimal, 3, proto)
    assert p3.level.level == 3
    assert set(p3.fields).issuperset(
        {"id", "allocation", "recent_validation_rate", "birth_epoch",
         "claim_mutations_count", "claim", "description", "metadata"}
    )
    print(f"  LOD 0 -> 3 introduces: "
          f"{sorted(set(p3.fields) - set(minimal.fields))}")


def test_tendency_at_lod() -> None:
    banner("test: Tendency.at_lod() produces a new tendency at target LOD")
    proto = TrivialLODProtocol()
    payload = make_full_synthetic("epsilon", 0.4)
    t_full = Tendency(
        id="epsilon",
        allocation=0.4,
        description="the epsilon drive",
        metadata={"origin": "synthetic"},
        protocol=proto,
        payload=payload,
    )
    assert t_full.current_lod == 3

    t_id = t_full.at_lod(0)
    assert t_id.current_lod == 0
    assert t_id.id == "epsilon"
    assert t_id.allocation == 0.4

    t_back = t_id.at_lod(3)
    assert t_back.current_lod == 3
    # Going back up loses claim/description because expand fills with defaults.
    # This is expected -- expand is reconstructive, not recovery.
    assert t_back.id == "epsilon"
    print(f"  full -> identity -> full: id preserved, content reconstructed-with-defaults")
    print(f"  full claim:   '{t_full.metadata.get('origin')}' / "
          f"reconstructed: '{t_back.metadata.get('origin', '<empty>')}'")


def test_byte_size_uniform_across_tendencies_at_same_lod() -> None:
    banner("test: at LOD K, byte size is uniform regardless of content")
    proto = TrivialLODProtocol()
    # All ids fit comfortably within the LOD-0 budget. Pathologically
    # long ids are tested separately by test_budget_overflow_raises.
    payloads = [
        make_full_synthetic("a", 0.1),
        make_full_synthetic("medium_name", 0.5),
        make_full_synthetic("z", 0.99),
    ]
    for k in range(proto.max_level() + 1):
        sizes = {
            payload.fields["id"]: to_level(payload, k, proto).to_bytes()
            for payload in payloads
        }
        # All same byte length
        unique_lengths = {len(b) for b in sizes.values()}
        assert len(unique_lengths) == 1, (
            f"LOD {k}: payloads have different sizes: "
            f"{ {k: len(v) for k, v in sizes.items()} }"
        )
        print(f"  LOD {k}: all payloads exactly {unique_lengths.pop()} bytes "
              f"(over {len(payloads)} different content samples)")


def test_budget_overflow_raises() -> None:
    banner("test: oversized payload raises rather than truncating silently")
    proto = TrivialLODProtocol()
    # LOD 0 budget is 32 bytes. Stuff a string that exceeds this.
    bloated = LODPayload(
        level=proto.level(0),
        fields={"id": "x" * 200, "allocation": 0.5},
    )
    try:
        bloated.to_bytes()
    except ValueError as e:
        print(f"  raised as expected: {e}")
        return
    raise AssertionError("expected ValueError on budget overflow")


if __name__ == "__main__":
    test_strict_fixed_size_at_every_level()
    test_round_trip_through_bytes()
    test_reduce_is_deterministic_and_lossy()
    test_expand_introduces_default_fields()
    test_tendency_at_lod()
    test_byte_size_uniform_across_tendencies_at_same_lod()
    test_budget_overflow_raises()
    print()
    print("All LOD smoke tests passed.")
