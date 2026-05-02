#!/usr/bin/env python3
"""Test outcome signal extraction heuristics."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from nodes.common.world_model_substrate.outcomes import (
    Outcome,
    extract_acceptance,
    extract_built_on,
    extract_kept,
    extract_paid,
    outcome_from_conversation,
)


def write_conv(path: Path, messages: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")


def main() -> int:
    cases_passed = 0
    cases_total = 0

    # 1. Acceptance: positive marker
    cases_total += 1
    msgs = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "Fixed."},
        {"role": "user", "content": "Thanks, looks good"},
    ]
    if extract_acceptance(msgs) == 1.0:
        cases_passed += 1
        print("[PASS] acceptance: positive marker")
    else:
        print(f"[FAIL] acceptance: positive marker -> {extract_acceptance(msgs)}")

    # 2. Acceptance: rejection marker
    cases_total += 1
    msgs = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "Did this."},
        {"role": "user", "content": "No, that's wrong, redo it"},
    ]
    if extract_acceptance(msgs) == -1.0:
        cases_passed += 1
        print("[PASS] acceptance: rejection marker")
    else:
        print(f"[FAIL] acceptance: rejection -> {extract_acceptance(msgs)}")

    # 3. Acceptance: no follow-up
    cases_total += 1
    msgs = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "Done."},
    ]
    if extract_acceptance(msgs) == 0.0:
        cases_passed += 1
        print("[PASS] acceptance: no follow-up")
    else:
        print(f"[FAIL] acceptance: no follow-up -> {extract_acceptance(msgs)}")

    # 4. Kept: default true
    cases_total += 1
    msgs = [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "ok"}]
    if extract_kept(msgs) == 1.0:
        cases_passed += 1
        print("[PASS] kept: default true")
    else:
        print(f"[FAIL] kept: default -> {extract_kept(msgs)}")

    # 5. Kept: revert detected
    cases_total += 1
    msgs = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "revert that, it broke things"},
    ]
    if extract_kept(msgs) == -1.0:
        cases_passed += 1
        print("[PASS] kept: revert detected")
    else:
        print(f"[FAIL] kept: revert -> {extract_kept(msgs)}")

    # 6. Built-on: file path referenced later
    cases_total += 1
    earlier = [
        {"role": "user", "content": "edit nodes/common/world_model_substrate/adapter.py"},
        {"role": "assistant", "content": "Modified the adapter."},
    ]
    later = [
        {"role": "user", "content": "now also update nodes/common/world_model_substrate/adapter.py to handle X"},
    ]
    score = extract_built_on(earlier, later)
    if score >= 0.5:
        cases_passed += 1
        print(f"[PASS] built_on: file path reference -> {score}")
    else:
        print(f"[FAIL] built_on: file path -> {score}")

    # 7. Built-on: no relation
    cases_total += 1
    earlier = [{"role": "assistant", "content": "did some stuff"}]
    later = [{"role": "user", "content": "totally different topic"}]
    score = extract_built_on(earlier, later)
    if score == 0.0:
        cases_passed += 1
        print("[PASS] built_on: no relation")
    else:
        print(f"[FAIL] built_on: no relation -> {score}")

    # 8. Paid: placeholder zero
    cases_total += 1
    if extract_paid([]) == 0.0:
        cases_passed += 1
        print("[PASS] paid: placeholder zero")
    else:
        print(f"[FAIL] paid: placeholder")

    # 9. End-to-end on a temp conversation
    cases_total += 1
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.jsonl"
        write_conv(path, [
            {"role": "user", "content": "edit foo/bar/baz_module.py to add temporal awareness"},
            {"role": "assistant", "content": "Added it."},
            {"role": "user", "content": "Thanks, exactly what I wanted"},
        ])
        outcome = outcome_from_conversation(path)
        if outcome.accepted == 1.0 and outcome.kept == 1.0:
            cases_passed += 1
            print(f"[PASS] e2e: {outcome.to_coords()}")
        else:
            print(f"[FAIL] e2e: {outcome.to_coords()}")

    # 10. Outcome.signal_strength
    cases_total += 1
    o = Outcome(accepted=1.0, kept=1.0, built_on=0.0, paid=0.0)
    if o.signal_strength == 2.0:
        cases_passed += 1
        print("[PASS] signal_strength")
    else:
        print(f"[FAIL] signal_strength: {o.signal_strength}")

    print(f"\n{cases_passed}/{cases_total} cases passed")
    return 0 if cases_passed == cases_total else 1


if __name__ == "__main__":
    sys.exit(main())
