# Substrate experiments (not installed with the daemon)

Imported from the external experiment workspace on 2026-07-03 so the
evidence travels with the code. Excluded from packaging (`pyproject`
includes only `atn*`, `nodes*`, `bridge`, `world_model*`).

- `phase7/` — corrected small-LLM contest on the renamed-toolz corpus.
  Invalidated phase 6 (its substrate arm had been bare haiku) and
  showed context *hurts* on a saturated domain. See PHASE7_PLAN.md.
- `phase8/` — the pre-registered equilibration-vs-vote-ledger contest
  on an unknown domain (real autonet session corpus). Design:
  `docs/phase8_prereg.md` (+3 amendments); results:
  `docs/phase8_results.md`. Verdict: retrieval helps (+0.37),
  verdicts-in-context hurt (−0.28), equilibration +0.13 — below the
  pre-registered bar, demoted.
- Phase 9 (equilibration at depth, hand-built graph) is designed but
  not run: `docs/phase9_depth_experiment.md`.
- `robots/`: three runs with simulated TurtleBot3s in Gazebo (2026-11-11):
  safety packs at the actuation boundary (11/11), the poll rail (9/9), and
  scene summaries with the automatic trigger on position claims (6/6). Setup,
  results, limits and logs in `robots/README.md`; design in `docs/robots.md`.

Rebuildable artifacts (world snapshots, artifact indexes) are excluded;
regenerate with `phase8/build_worlds.py`. `llm_cache/` is the audit
trail of raw model calls (prompt-hash keyed) backing every reported
number.
