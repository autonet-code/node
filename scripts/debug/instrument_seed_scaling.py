"""Instrumented seed scaling diagnostic.

Reads work units from work_units_filtered.jsonl, seeds a FRESH world (a
disposable temp data root — does not touch ~/.autonet/world/default),
and instruments per-event hot-path costs to determine *why* batch wall
time grows super-linearly even with 64-dim + scoped equilibrate in place.

What we want to answer:

  Q1. Is scoped equilibrate actually scoping? Print the `scope` set size
      and the number of tendencies in the world per event. If scope size
      grows with N (because tendency anchors crowd the embedding space),
      scope isn't local.

  Q2. Even when scope is small, does tendency.act() touch all-T anyway?
      Specifically the cross-tendency post loop at tendency.py:537-563.
      Count world.tendencies iterations per act() call.

  Q3. Total event count growth: is N (number of nodes) growing
      linearly with events? If each event spawns multiple sub-claims,
      we get amplified cost growth.

  Q4. Per-call timings: act(), apply_stakes(), update_novelty(),
      snapshot/persistence — which dominates as N grows?

Run:

    python scripts/instrument_seed_scaling.py --units 100 --report-every 10

This is read-only against your input file and writes to a temp dir
(removed on exit unless --keep-temp). The daemon's substrate at
~/.autonet/world/default is NOT touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = Path(
    r"D:/videos/SF/manifesting/from_endstate/new physics/substrate_experiment/"
    r"work_units_filtered.jsonl"
)


def iter_work_units(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--units", type=int, default=100,
                   help="Number of work units to seed (default 100)")
    p.add_argument("--report-every", type=int, default=10,
                   help="Print summary every N events (default 10)")
    p.add_argument("--keep-temp", action="store_true",
                   help="Keep the temp data dir for inspection")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    # --- Load valid units ---
    valid: list[tuple[str, str, list[float]]] = []
    for u in iter_work_units(args.input):
        prob = (u.get("problem") or "").strip()
        res = (u.get("resolution") or "").strip()
        if not prob or not res:
            continue
        oc = u.get("outcome") or [0.0, 0.0, 0.0, 0.0]
        if not isinstance(oc, list) or len(oc) != 4:
            oc = [0.0, 0.0, 0.0, 0.0]
        valid.append((prob, res, [float(x) for x in oc]))
        if len(valid) >= args.units:
            break
    print(f"loaded {len(valid)} units from {args.input.name}")

    # --- Temp data root so we don't touch the real substrate ---
    tmp_root = Path(tempfile.mkdtemp(prefix="instrument_seed_"))
    print(f"temp data root: {tmp_root}")

    try:
        # --- Monkey-patch instrumentation BEFORE constructing WorldService ---
        from world_model.generalized import tendency as tendency_mod
        from world_model.generalized import world as world_mod
        # The package __init__ re-exports `equilibrate` (the function),
        # shadowing the submodule attribute. Pull the module from sys.modules.
        import importlib
        eq_mod = importlib.import_module("world_model.generalized.equilibrate")

        counters: dict[str, Any] = {
            "act_calls": 0,
            "act_cross_tendency_iterations": 0,
            "act_own_obs_iterations": 0,
            "apply_stakes_calls": 0,
            "equilibrate_calls": 0,
            "equilibrate_rounds_total": 0,
            "scope_sizes": [],
            "world_tendency_count_at_event": [],
            "world_node_count_at_event": [],
            "world_obs_count_at_event": [],
            # Cumulative timing
            "t_act": 0.0,
            "t_apply_stakes": 0.0,
            "t_update_novelty": 0.0,
            "t_equilibrate_total": 0.0,
            "t_submit_observation_total": 0.0,
            # Per-event timings
            "per_event_times": [],   # list of dicts
        }

        # Patch tendency.act to count cross-tendency iterations and time itself.
        _orig_act = tendency_mod.GeneralizedTendency.act

        def _instrumented_act(self, world):
            counters["act_calls"] += 1
            t0 = time.perf_counter()
            # We can't easily inject a counter inside the loop without
            # re-implementing the function, so we count externally:
            # The cross-tendency loop iterates over (T-1) tendencies and
            # for each, iterates over every node in that tendency's tree.
            n_others = max(0, len(world.tendencies) - 1)
            n_other_nodes = 0
            for tid, t in world.tendencies.items():
                if tid != self.id:
                    n_other_nodes += len(t.tree.all_nodes())
            counters["act_cross_tendency_iterations"] += n_other_nodes
            counters["act_own_obs_iterations"] += len(world.observations)
            _orig_act(self, world)
            counters["t_act"] += time.perf_counter() - t0

        tendency_mod.GeneralizedTendency.act = _instrumented_act

        # Patch World.apply_stakes
        _orig_apply_stakes = world_mod.World.apply_stakes

        def _instrumented_apply_stakes(self, *a, **kw):
            counters["apply_stakes_calls"] += 1
            t0 = time.perf_counter()
            r = _orig_apply_stakes(self, *a, **kw)
            counters["t_apply_stakes"] += time.perf_counter() - t0
            return r

        world_mod.World.apply_stakes = _instrumented_apply_stakes

        # Patch update_novelty
        _orig_update_novelty = tendency_mod.GeneralizedTendency.update_novelty

        def _instrumented_update_novelty(self, *a, **kw):
            t0 = time.perf_counter()
            r = _orig_update_novelty(self, *a, **kw)
            counters["t_update_novelty"] += time.perf_counter() - t0
            return r

        tendency_mod.GeneralizedTendency.update_novelty = _instrumented_update_novelty

        # Patch equilibrate
        _orig_equilibrate = eq_mod.equilibrate

        def _instrumented_equilibrate(world, *a, **kw):
            counters["equilibrate_calls"] += 1
            scope = kw.get("scope")
            if scope is not None:
                counters["scope_sizes"].append(len(scope))
            else:
                counters["scope_sizes"].append(-1)  # full
            t0 = time.perf_counter()
            r = _orig_equilibrate(world, *a, **kw)
            counters["t_equilibrate_total"] += time.perf_counter() - t0
            counters["equilibrate_rounds_total"] += r
            return r

        # Replace in BOTH places it's imported.
        eq_mod.equilibrate = _instrumented_equilibrate
        # The WorldService imports `equilibrate` directly into its module
        # namespace, so patch there too.
        import nodes.common.world_service as ws_mod
        ws_mod.equilibrate = _instrumented_equilibrate

        # --- Build the WorldService against the temp dir ---
        from nodes.common.world_service import WorldService

        ws = WorldService(rpb_address="default", data_root=tmp_root)
        print(f"WorldService ready; embedding_dim={ws.embedding_dim}, "
              f"scoped_equilibrate_enabled="
              f"{os.environ.get('AUTONET_SCOPED_EQUILIBRATE', '1') != '0'}")

        # Get an initial snapshot of tendency count
        with ws._lock:
            initial_tendencies = len(ws._world.tendencies)
        print(f"initial tendencies in world: {initial_tendencies}")

        # --- Seed one event at a time ---
        print()
        print("=" * 100)
        print(f"{'event':>5} {'wall_s':>8} {'cum_s':>8} "
              f"{'tendencies':>10} {'nodes':>7} {'obs':>5} "
              f"{'scope':>5} {'act_calls':>10} {'cross_it':>10} "
              f"{'rounds':>6}")
        print("=" * 100)

        t_seed_start = time.perf_counter()

        last_event_counters = {
            "act_calls": 0,
            "act_cross_tendency_iterations": 0,
            "equilibrate_rounds_total": 0,
        }

        for i, (problem, resolution, outcome) in enumerate(valid):
            t_event_start = time.perf_counter()
            ws.submit_work_units([(problem, resolution, outcome)],
                                 agent_id="root")
            t_event = time.perf_counter() - t_event_start
            t_cum = time.perf_counter() - t_seed_start

            with ws._lock:
                T = len(ws._world.tendencies)
                # Total nodes across all tendencies (excluding root nodes)
                N = sum(len(t.tree.all_nodes()) for t in ws._world.tendencies.values())
                O = len(ws._world.observations)

            counters["world_tendency_count_at_event"].append(T)
            counters["world_node_count_at_event"].append(N)
            counters["world_obs_count_at_event"].append(O)
            counters["t_submit_observation_total"] += t_event

            # Per-event deltas
            d_act = counters["act_calls"] - last_event_counters["act_calls"]
            d_cross = (counters["act_cross_tendency_iterations"]
                       - last_event_counters["act_cross_tendency_iterations"])
            d_rounds = (counters["equilibrate_rounds_total"]
                        - last_event_counters["equilibrate_rounds_total"])

            scope_sz = counters["scope_sizes"][-1] if counters["scope_sizes"] else "-"

            counters["per_event_times"].append({
                "i": i,
                "wall": t_event,
                "T": T,
                "N": N,
                "O": O,
                "scope": scope_sz,
                "act": d_act,
                "cross": d_cross,
                "rounds": d_rounds,
            })

            if (i + 1) % args.report_every == 0 or i == 0 or i == len(valid) - 1:
                print(f"{i + 1:>5} {t_event:>8.3f} {t_cum:>8.1f} "
                      f"{T:>10} {N:>7} {O:>5} "
                      f"{scope_sz:>5} {d_act:>10} {d_cross:>10} "
                      f"{d_rounds:>6}")

            last_event_counters["act_calls"] = counters["act_calls"]
            last_event_counters["act_cross_tendency_iterations"] = \
                counters["act_cross_tendency_iterations"]
            last_event_counters["equilibrate_rounds_total"] = \
                counters["equilibrate_rounds_total"]

        # --- Summary ---
        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        total_wall = time.perf_counter() - t_seed_start
        print(f"total wall time:                   {total_wall:8.2f}s")
        print(f"submit_observation cumulative:     {counters['t_submit_observation_total']:8.2f}s")
        print(f"equilibrate cumulative:            {counters['t_equilibrate_total']:8.2f}s")
        print(f"  - act() cumulative:              {counters['t_act']:8.2f}s")
        print(f"  - apply_stakes() cumulative:     {counters['t_apply_stakes']:8.2f}s")
        print(f"  - update_novelty() cumulative:   {counters['t_update_novelty']:8.2f}s")
        print(f"  - other (rounds, conv, etc):     "
              f"{counters['t_equilibrate_total'] - counters['t_act'] - counters['t_apply_stakes'] - counters['t_update_novelty']:8.2f}s")
        print()
        print(f"act() calls:                       {counters['act_calls']}")
        print(f"apply_stakes() calls:              {counters['apply_stakes_calls']}")
        print(f"equilibrate() calls:               {counters['equilibrate_calls']}")
        print(f"equilibrate rounds total:          {counters['equilibrate_rounds_total']}")
        print(f"cross-tendency iterations total:   "
              f"{counters['act_cross_tendency_iterations']}")
        print()

        # Scope analysis
        scoped = [s for s in counters["scope_sizes"] if s >= 0]
        full = [s for s in counters["scope_sizes"] if s < 0]
        print(f"equilibrate calls with scope:      {len(scoped)} "
              f"(min={min(scoped) if scoped else '-'}, "
              f"max={max(scoped) if scoped else '-'}, "
              f"avg={sum(scoped) / len(scoped):.1f}"
              f")" if scoped else "(none)")
        print(f"equilibrate calls full-scope:      {len(full)}")
        print()

        # Per-event growth — show every Kth event's per-event times
        print("PER-EVENT GROWTH (sampled)")
        print(f"{'event':>5} {'wall_s':>8} {'T':>5} {'N':>6} "
              f"{'O':>5} {'scope':>5} {'act/ev':>8} {'cross/ev':>10} {'rounds':>6}")
        n_events = len(counters["per_event_times"])
        sample_indices = list(range(0, n_events, max(1, n_events // 20)))
        if sample_indices[-1] != n_events - 1:
            sample_indices.append(n_events - 1)
        for idx in sample_indices:
            ev = counters["per_event_times"][idx]
            print(f"{ev['i'] + 1:>5} {ev['wall']:>8.3f} {ev['T']:>5} "
                  f"{ev['N']:>6} {ev['O']:>5} {str(ev['scope']):>5} "
                  f"{ev['act']:>8} {ev['cross']:>10} {ev['rounds']:>6}")

        # Save raw timings as JSON for plotting
        out_path = REPO_ROOT / "scripts" / ".instrument_seed_results.json"
        with out_path.open("w") as f:
            json.dump({
                "units": len(valid),
                "embedding_dim": ws.embedding_dim,
                "initial_tendencies": initial_tendencies,
                "total_wall_s": total_wall,
                "counters": {k: v for k, v in counters.items()
                             if k != "per_event_times"
                             and not isinstance(v, list)},
                "per_event_times": counters["per_event_times"],
                "scope_sizes": counters["scope_sizes"],
                "tendency_count_progression": counters["world_tendency_count_at_event"],
                "node_count_progression": counters["world_node_count_at_event"],
            }, f, indent=2)
        print(f"\nraw results saved to: {out_path}")

    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"cleaned up: {tmp_root}")
        else:
            print(f"kept temp dir: {tmp_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
