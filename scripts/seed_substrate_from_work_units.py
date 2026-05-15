"""Seed the daemon's substrate from extracted work units.

Pre-launch tool. Reads ``work_units_*.jsonl`` produced by
``substrate_experiment/extract_sessions.py`` and feeds each unit into
the daemon's ``WorldService`` so the daemon starts up with a real
substrate instead of an empty one.

Two scoring paths:

  * **Default (fast, free)** — calls ``WorldService.submit_work_units``
    which uses the local usefulness embedder. Charter head (the 4
    alignment + 2 usefulness *axis* dims) is zeroed; the embedding
    tail carries topical structure.

  * **``--llm-charter`` (slow, real signal)** — for each unit, builds a
    synthetic turn dict and scores it on the full 6-axis charter via
    ``llm_score_turn`` against the Claude Max bridge. The resulting
    charter head is then merged with the usefulness embedder tail and
    submitted via ``submit_observation`` directly (bypassing the
    submit_work_units zero-charter-head shortcut). Result is a
    substrate where every node carries real alignment signal — what
    the network will produce in production.

Operational shape:

  * Constructs a fresh ``WorldService`` pointed at the daemon's data
    directory. The persistence layer picks up any existing state and
    appends to it.
  * **The daemon MUST be stopped** while this runs — both processes
    writing to the same on-disk substrate state would corrupt it.
  * Attributes every observation to a single ``agent_id`` (default
    ``"root"``). This keeps the seeded substrate honest — "my
    historical work" rather than synthetic activity from a phantom
    network.

LLM-charter mode details:

  * Spawns N independent ``BridgeProvider`` instances (default 4).
    Each owns its own bridge subprocess and asyncio.Lock, so calls
    parallelize across providers. One worker per provider pulls work
    units from a queue, scores via the LLM, and submits the result.
  * On-disk cache at ``--llm-cache-path`` is shared across providers
    (keyed by ``(provider_name, model, obs_id)``). Crash-safe: restart
    resumes from cache. Cache writes are append-only JSONL.
  * The Claude Max bridge has built-in prompt caching keyed by session.
    Each provider builds up its session naturally on its first call.

Usage:

    # smoke test — default heuristic, 5 units, dry-run
    python scripts/seed_substrate_from_work_units.py --dry-run --limit 5

    # full run, fast heuristic mode
    python scripts/seed_substrate_from_work_units.py

    # LLM-charter smoke (10 units, single provider, sanity check the LLM path)
    python scripts/seed_substrate_from_work_units.py \\
        --llm-charter --limit 10 --concurrency 1

    # overnight full run, 4-way parallel haiku scoring
    python scripts/seed_substrate_from_work_units.py \\
        --llm-charter --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Default input — matches the extract_sessions.py output location the
# user described. Users override with --input.
DEFAULT_INPUT = Path(
    r"D:/videos/SF/manifesting/from_endstate/new physics/substrate_experiment/"
    r"work_units_filtered.jsonl"
)

# Cache for LLM-scored charter heads. Lives next to the daemon data so
# repeated seed runs (with different --limit values) share work.
DEFAULT_LLM_CACHE = REPO_ROOT / "scripts" / ".seed_llm_cache.jsonl"

# Charter head is 6 dims (4 alignment + 2 usefulness).
N_CHARTER_DIMS = 6


log = logging.getLogger("seed_substrate")


# ---------------------------------------------------------------------------
# Work-unit reader (shared by both paths)
# ---------------------------------------------------------------------------


def iter_work_units(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each work-unit dict from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed line %d in %s", line_no, path)


def work_unit_to_tuple(unit: dict[str, Any]) -> tuple[str, str, list[float]] | None:
    """Convert a work-unit dict to ``(problem, resolution, outcome)``.

    Returns None when required fields are missing/empty.
    """
    problem = (unit.get("problem") or "").strip()
    resolution = (unit.get("resolution") or "").strip()
    outcome = unit.get("outcome") or [0.0, 0.0, 0.0, 0.0]
    if not problem or not resolution:
        return None
    if not isinstance(outcome, list) or len(outcome) != 4:
        outcome = [float(x) for x in (list(outcome) + [0.0] * 4)[:4]]
    else:
        outcome = [float(x) for x in outcome]
    return (problem, resolution, outcome)


def work_unit_to_turn(problem: str, resolution: str) -> dict[str, Any]:
    """Convert a work unit into a 'turn' dict ``llm_score_turn`` accepts.

    The scorer's serializer reads a few well-known fields. We fold the
    problem + resolution into the ``content`` field so the LLM sees one
    coherent text. We label it with role=`pair` for traceability — the
    scorer doesn't care, it just renders any string fields.
    """
    return {
        "role": "pair",
        "content": f"PROBLEM:\n{problem}\n\nRESOLUTION:\n{resolution}",
    }


# ---------------------------------------------------------------------------
# WorldService construction
# ---------------------------------------------------------------------------


def build_world_service(data_dir: Path | None, rpb_address: str):
    """Construct a WorldService that shares the daemon's persistence
    layer. Lazy import so non-seeding environments don't need nodes."""
    from nodes.common.world_service import WorldService  # noqa: WPS433
    if data_dir is not None:
        return WorldService(rpb_address=rpb_address, data_root=data_dir)
    return WorldService(rpb_address=rpb_address)


# ---------------------------------------------------------------------------
# Default path: heuristic embedder via submit_work_units
# ---------------------------------------------------------------------------


def run_heuristic(
    *,
    units: list[tuple[str, str, list[float]]],
    ws,
    agent_id: str,
    batch_size: int,
) -> int:
    """Feed units through WorldService.submit_work_units in batches.

    Returns the total event count appended.
    """
    total_events = 0
    started_at = time.time()
    n_batches = (len(units) + batch_size - 1) // batch_size
    for i in range(0, len(units), batch_size):
        batch = units[i:i + batch_size]
        receipt = ws.submit_work_units(batch, agent_id=agent_id)
        total_events += int(receipt.get("events_appended", 0))
        log.info(
            "batch %d/%d: units=%d events_total=%d",
            (i // batch_size) + 1, n_batches, len(batch), total_events,
        )
    duration = time.time() - started_at
    log.info(
        "heuristic done: %d units → %d events in %.1fs (%.1f units/s)",
        len(units), total_events, duration,
        len(units) / duration if duration > 0 else 0.0,
    )
    return total_events


# ---------------------------------------------------------------------------
# LLM-charter path: parallel scoring + serial submission
# ---------------------------------------------------------------------------


async def _score_one(
    *,
    problem: str,
    resolution: str,
    outcome: list[float],
    provider,
    model: str,
    cache_path: Path,
) -> tuple[tuple[float, ...], str]:
    """Score one work unit and return (charter_head_6d, obs_id)."""
    from nodes.common.world_model_substrate.llm_score_turn import (  # noqa: WPS433
        llm_score_turn,
    )
    from nodes.common.world_model_substrate.adapter import (  # noqa: WPS433
        _obs_id_from_turn,
    )

    turn = work_unit_to_turn(problem, resolution)
    obs_id = _obs_id_from_turn(turn)
    head = await llm_score_turn(
        turn, provider=provider, model=model, cache_path=cache_path,
    )
    return head, obs_id


def _submit_observation(
    *,
    ws,
    problem: str,
    resolution: str,
    charter_head: tuple[float, ...],
    agent_id: str,
) -> int:
    """Build an Observation with [charter_head | usefulness_tail] and
    submit it directly. Returns the event count applied (usually 1)."""
    from nodes.common.world_model_substrate.usefulness_coords import (  # noqa: WPS433
        default_usefulness_embedder,
        coords_for_problem_resolution,
    )
    from nodes.common.world_model_substrate.adapter import (  # noqa: WPS433
        _obs_id_from_turn, Observation,
    )

    embedder = default_usefulness_embedder(dim=ws.embedding_dim)
    tail = tuple(coords_for_problem_resolution(
        problem, resolution, embedder=embedder,
    ))
    # Pad/truncate the tail to ws.embedding_dim (mirrors submit_work_units).
    tail_t = tail[: ws.embedding_dim]
    if len(tail_t) < ws.embedding_dim:
        tail_t = tail_t + (0.0,) * (ws.embedding_dim - len(tail_t))
    coords = tuple(charter_head) + tail_t

    turn = work_unit_to_turn(problem, resolution)
    obs_id = "wu_" + _obs_id_from_turn(turn)[3:]
    label = (problem or "")[:80]
    obs = Observation(id=obs_id, coords=coords, label=label)

    receipt = ws.submit_observation(
        obs,
        agent_id=agent_id,
        sprout_under_charter=True,
        sprout_rootless=True,
    )
    return int(receipt.get("events_applied", 0))


async def run_llm_charter(
    *,
    units: list[tuple[str, str, list[float]]],
    ws,
    agent_id: str,
    concurrency: int,
    model: str,
    cache_path: Path,
) -> int:
    """Score units with the LLM in parallel, submit serially.

    Spawns ``concurrency`` BridgeProvider instances. Each runs as a
    worker pulling units off a shared queue, scoring via the bridge,
    then dropping the scored result on a result queue. A separate
    submitter consumes the result queue and writes to WorldService —
    serially, since WorldService holds an internal lock anyway.
    """
    from atn.providers.bridge import BridgeProvider  # noqa: WPS433

    in_queue: asyncio.Queue[tuple[int, tuple[str, str, list[float]]] | None] = (
        asyncio.Queue()
    )
    out_queue: asyncio.Queue[tuple[int, tuple[str, str, list[float]], tuple[float, ...]] | None] = (
        asyncio.Queue()
    )

    # Producer: drop all units onto the in_queue, then signal end to each worker.
    for idx, unit in enumerate(units):
        in_queue.put_nowait((idx, unit))
    for _ in range(concurrency):
        in_queue.put_nowait(None)

    providers = [BridgeProvider(model=model) for _ in range(concurrency)]
    log.info("spawned %d BridgeProvider instance(s) for model=%s",
             concurrency, model)

    total_events = 0
    started_at = time.time()
    submitted = 0
    scored = 0

    async def worker(worker_id: int, provider) -> None:
        nonlocal scored
        while True:
            item = await in_queue.get()
            if item is None:
                await out_queue.put(None)
                return
            idx, (problem, resolution, outcome) = item
            try:
                head, _obs_id = await _score_one(
                    problem=problem, resolution=resolution, outcome=outcome,
                    provider=provider, model=model, cache_path=cache_path,
                )
            except Exception:
                log.exception("worker %d: scoring failed for unit %d; "
                              "falling back to zero charter head", worker_id, idx)
                head = (0.0,) * N_CHARTER_DIMS
            await out_queue.put((idx, (problem, resolution, outcome), head))
            scored += 1

    async def submitter() -> None:
        nonlocal total_events, submitted
        end_signals = 0
        while end_signals < concurrency:
            item = await out_queue.get()
            if item is None:
                end_signals += 1
                continue
            idx, (problem, resolution, _outcome), head = item
            try:
                applied = _submit_observation(
                    ws=ws, problem=problem, resolution=resolution,
                    charter_head=head, agent_id=agent_id,
                )
                total_events += applied
            except Exception:
                log.exception("submission failed for unit %d", idx)
            submitted += 1
            if submitted % 25 == 0 or submitted == len(units):
                elapsed = time.time() - started_at
                rate = submitted / elapsed if elapsed > 0 else 0.0
                eta = (len(units) - submitted) / rate if rate > 0 else 0.0
                log.info(
                    "progress: scored=%d submitted=%d/%d events=%d "
                    "elapsed=%.0fs rate=%.2f/s eta=%.0fs",
                    scored, submitted, len(units), total_events,
                    elapsed, rate, eta,
                )

    workers = [
        asyncio.create_task(worker(i, p))
        for i, p in enumerate(providers)
    ]
    submitter_task = asyncio.create_task(submitter())

    await asyncio.gather(*workers)
    await submitter_task

    # Drain providers — each owns a subprocess.
    for p in providers:
        try:
            close = getattr(p, "close", None)
            if close:
                if asyncio.iscoroutinefunction(close):
                    await close()
                else:
                    close()
        except Exception:
            log.warning("provider close failed (non-fatal)", exc_info=True)

    duration = time.time() - started_at
    log.info(
        "llm-charter done: %d units → %d events in %.1fs (%.2f units/s)",
        len(units), total_events, duration,
        len(units) / duration if duration > 0 else 0.0,
    )
    return total_events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Path to work_units_*.jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--agent-id", default="root",
        help="Agent attribution for every minted observation (default: root)",
    )
    parser.add_argument(
        "--rpb-address", default="default",
        help="WorldService rpb_address — must match the daemon's (default: default)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override WorldService data root. Default: WorldPersistence's "
             "built-in default (~/.atn/world).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="(heuristic mode) units per submit_work_units call (default 100)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only process the first N units (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count + preview, no WorldService writes",
    )
    parser.add_argument(
        "--llm-charter", action="store_true",
        help="Use Claude Max bridge to score the 6-axis charter head per unit.",
    )
    parser.add_argument(
        "--llm-model", default="haiku",
        help="(llm-charter mode) bridge model (default: haiku)",
    )
    parser.add_argument(
        "--llm-cache-path", type=Path, default=DEFAULT_LLM_CACHE,
        help=f"(llm-charter mode) JSONL cache path "
             f"(default: {DEFAULT_LLM_CACHE})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="(llm-charter mode) number of BridgeProvider instances "
             "running in parallel (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input.is_file():
        log.error("input file not found: %s", args.input)
        return 2

    # Pass 1: count + validate.
    valid: list[tuple[str, str, list[float]]] = []
    skipped = 0
    for unit in iter_work_units(args.input):
        t = work_unit_to_tuple(unit)
        if t is None:
            skipped += 1
            continue
        valid.append(t)
        if args.limit and len(valid) >= args.limit:
            break
    log.info("loaded %d valid work units (%d skipped) from %s",
             len(valid), skipped, args.input)
    if valid:
        log.info("  first problem: %s", valid[0][0][:80])
        log.info("  first outcome: %s", valid[0][2])

    if args.dry_run:
        log.info("dry-run: not constructing WorldService, exiting")
        return 0
    if not valid:
        log.warning("nothing to seed; exiting")
        return 0

    log.info("building WorldService (rpb=%s, data_root=%s)",
             args.rpb_address, args.data_root)
    ws = build_world_service(args.data_root, args.rpb_address)
    log.info("WorldService ready; embedding_dim=%d", ws.embedding_dim)

    if args.llm_charter:
        log.info("mode=llm-charter model=%s concurrency=%d cache=%s",
                 args.llm_model, args.concurrency, args.llm_cache_path)
        total = asyncio.run(run_llm_charter(
            units=valid, ws=ws, agent_id=args.agent_id,
            concurrency=args.concurrency, model=args.llm_model,
            cache_path=args.llm_cache_path,
        ))
    else:
        log.info("mode=heuristic batch_size=%d", args.batch_size)
        total = run_heuristic(
            units=valid, ws=ws, agent_id=args.agent_id,
            batch_size=args.batch_size,
        )

    log.info("done: %d events appended", total)
    log.info("restart the daemon to see the seeded substrate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
