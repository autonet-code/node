# Ledger pricing: post-phase8 changes

Ratified 2026-07-03 (phase8 gate outcome, user-approved): mint pricing
moves from score-movement-under-geometric-equilibration to explicit
debate standing. Equilibration becomes an opt-in experimental kernel.
Emission rate/schedule untouched (still config-gated, still unset).

**v3 note (2026-07-08, docs/tool_substrate.md Decision):** on the live
tool rail, standing has since been retired from mint entirely
(`mint = usage_term`) and Change 2's violator-pays gate is dormant —
work units no longer mint at all, and tool mint is not debate-gated.
Change 1 (ledger replay determinism) and Change 3 (arm-B render) stand.

## Change 1 — pricing mode in the epoch close (owner: reconcile agent)

`federated_epoch_close(..., pricing="ledger" | "equilibrated")`,
default **"ledger"**; "equilibrated" preserves today's behavior
bit-for-bit for the experimental kernel.

Ledger mode:
- Replay applies causal events (sprouts, posts) WITHOUT equilibrate
  rounds — no geometric propagation, no derived-sprout capture.
  Per-node score = the existing `net_score` tree recursion (posts +
  signed children). Snapshots record these.
- Mint formula shape unchanged: rise × survival × I(close>0); the
  novelty factor reads node.n where present else 1.0.
- Attribution uses causal events only (author-tagged sprouts).
- Determinism: net_score memoization over co-parented cycles is
  eval-order sensitive (see test_act_memoization) — ledger mode must
  fix an evaluation order (sorted node ids) and prove bit-identical
  double-close in tests.
- Plumb the mode through WorldService epoch close the same way
  epoch_emission_rate is plumbed. O(N²) equilibrate leaves the launch
  hot path.

## Change 2 — violator-pays mint gate (owner: reconcile agent)

Replace apply_mint_gate's uniform global ratio (mint_gate.py:207-218):
retain the per-(node, agent) attribution breakdown in reconcile,
scale it per-node by (1 − gate_strength × violation), THEN aggregate
per agent. With the emission pool, suppressed mint now genuinely
redistributes from flagged nodes' authors to everyone else — the
docstring's promise ("winning a CON debate increases your share")
becomes true. Tests must show: violator down, honest agents up
post-pool, and no change when no violations.

## Change 3 — inference render per phase8 arm B (owner: infer agent)

Phase8: verdicts in the prompt HURT (C−B = −0.28); standing must rank,
not fill context.
- `_infer_artifacts` standing must work on non-equilibrated worlds
  (net_score recursion — it already does; add a test constructed
  without calling equilibrate()).
- New `render_context(result, char_budget=6000) -> str` in infer.py:
  arm-B format — standing-ranked payloads only (problem+resolution
  text, proportional truncation), NO claims/verdict text. Claims stay
  in the structured result for programmatic consumers.
- `WorldService.infer_artifacts` gains `render=False` kwarg returning
  {"result": ..., "context": render_context(...)} when True.

## Out of scope

Defender loop (farmability), decay/novelty redesign, emission params,
Phase 9. Docs sweep (CLAUDE.md training-loop section, README stalenesss)
is a third, code-free task.
