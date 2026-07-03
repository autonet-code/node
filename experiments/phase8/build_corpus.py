#!/usr/bin/env python3
"""Phase 8 corpus builder (prereg Amendment 1).

Amendment 1 (2026-07-03, autonet commit 191d804): the original source
(work_units_autonet.jsonl, 17 units) was too thin. Amended source:
fresh extraction from the C--code-autonet and C--code-world-model
session transcripts (phase8/work_units_fresh.jsonl, produced by the
parent dir's extract_sessions.py pipeline on 2026-07-03), merged with
the legacy 17 and deduplicated by uid.

Amendment 3 (2026-07-03): the C--code-world-model project dir no
longer exists under ~/.claude/projects (deleted since the May 3
extraction), so fresh extraction covers C--code-autonet only, now WITH
--include-subagents (194 units). The surviving May 3 extraction of the
world-model transcripts (parent dir's work_units_world-model.jsonl,
13 units) is included as the third source. One near-duplicate pair
(same problem/resolution, drifted outcome from session growth) is
accepted per the amendment.

Each unit is normalized to the phase-8 schema:

    {"uid": <sha256 hex of the unit's canonical JSON>,
     "problem": str,
     "resolution": str,
     "outcome": {...} or {}}

Deterministic sample per prereg (docs/phase8_prereg.md, Corpus section):
sort ALL units by uid ascending, take first 200. If fewer than 200
exist, take all and note it.

Source schema (from extract_sessions.py):
    session_path, problem, resolution, outcome (a 4-float list =
    [accepted, kept, built_on, paid] coords from outcomes.py), timestamp,
    n_messages.

The `outcome` is a list of coords, not a dict. We normalize it to a named
dict so downstream (arms C/D "outcome signal") has stable keys. An
all-zero / empty list becomes {} per the "{...} or {}" contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY_SOURCE = HERE.parent / "work_units_autonet.jsonl"
FRESH_SOURCE = HERE / "work_units_fresh.jsonl"
LEGACY_WORLD_MODEL = HERE.parent / "work_units_world-model.jsonl"
INCLUDE_LEGACY_WORLD_MODEL = True  # Amendment 3
OUT = HERE / "corpus_sample.jsonl"

SAMPLE_N = 200
OUTCOME_KEYS = ["accepted", "kept", "built_on", "paid"]


def canonical_uid(problem: str, resolution: str, outcome) -> str:
    """sha256 hex of the unit's canonical JSON.

    Keyed on the identity-bearing content (problem, resolution, outcome)
    with sort_keys for determinism. session_path/timestamp are excluded
    so the uid is stable to where/when the session lived on disk and is a
    pure function of the work content.
    """
    payload = json.dumps(
        {"problem": problem, "resolution": resolution, "outcome": outcome},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_outcome(raw) -> dict:
    """4-float coord list -> named dict; drop all-zero to {}."""
    if not isinstance(raw, list) or not raw:
        return {}
    coords = [float(x) for x in raw[: len(OUTCOME_KEYS)]]
    if all(c == 0.0 for c in coords):
        return {}
    out = {}
    for k, v in zip(OUTCOME_KEYS, coords):
        out[k] = v
    return out


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def main() -> int:
    if not LEGACY_SOURCE.exists():
        raise SystemExit(f"legacy source not found: {LEGACY_SOURCE}")
    if not FRESH_SOURCE.exists():
        raise SystemExit(
            f"fresh source not found: {FRESH_SOURCE}; run extract_sessions.py "
            "(parent dir) with --project-match C--code-autonet --out phase8/work_units_fresh.jsonl"
        )

    fresh = read_jsonl(FRESH_SOURCE)
    legacy = read_jsonl(LEGACY_SOURCE)
    wm = read_jsonl(LEGACY_WORLD_MODEL) if INCLUDE_LEGACY_WORLD_MODEL else []
    # Fresh first: on a uid collision the fresh copy wins (identical
    # content anyway, since uid is content-derived).
    raw_units = fresh + legacy + wm
    print(f"fresh units read: {len(fresh)} ({FRESH_SOURCE.name})")
    print(f"legacy units read: {len(legacy)} ({LEGACY_SOURCE.name})")
    if INCLUDE_LEGACY_WORLD_MODEL:
        print(f"legacy world-model units read: {len(wm)} ({LEGACY_WORLD_MODEL.name})")

    normalized = []
    seen_uids = set()
    dedupe_collisions = 0
    for u in raw_units:
        problem = str(u.get("problem", ""))
        resolution = str(u.get("resolution", ""))
        outcome = normalize_outcome(u.get("outcome"))
        # uid over the RAW outcome list so identical work content collides
        # deterministically; use raw list (pre-normalization) for stability
        # to the source encoding.
        uid = canonical_uid(problem, resolution, u.get("outcome"))
        if uid in seen_uids:
            # exact-duplicate work content (uid is content-derived);
            # skip the dup so the sample has distinct units.
            dedupe_collisions += 1
            continue
        seen_uids.add(uid)
        normalized.append(
            {
                "uid": uid,
                "problem": problem,
                "resolution": resolution,
                "outcome": outcome,
            }
        )

    normalized.sort(key=lambda r: r["uid"])
    total = len(normalized)
    sample = normalized[:SAMPLE_N]

    with OUT.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    note = ""
    if total < SAMPLE_N:
        note = (
            f"  NOTE: only {total} distinct units exist (< {SAMPLE_N}); "
            f"took all {total}."
        )
    print(f"dedupe collisions (uid already seen): {dedupe_collisions}")
    print(f"distinct normalized units: {total}")
    print(f"sampled (first {SAMPLE_N} by uid asc): {len(sample)}")
    if note:
        print(note)
    print(f"wrote {OUT}")
    # quick stats
    n_with_outcome = sum(1 for r in sample if r["outcome"])
    mean_prob = sum(len(r["problem"]) for r in sample) / max(len(sample), 1)
    mean_res = sum(len(r["resolution"]) for r in sample) / max(len(sample), 1)
    print(
        f"  units with non-empty outcome: {n_with_outcome}/{len(sample)}; "
        f"mean problem chars={mean_prob:.0f}, mean resolution chars={mean_res:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
