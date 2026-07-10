# Docs index

Autonet's design specs, experiment records, and reference material. For
agent/machine onboarding start at the repo-root `CLAUDE.md`; for the
one-paragraph "what is this" start at the repo-root `README.md`.

Status conventions used below:
- **Living spec** — current design or built feature; safe to edit as the design evolves.
- **Historical record — DO NOT EDIT** — a pre-registered contract or a
  results doc. These were committed before/after a run as an integrity
  record; amend only via a follow-up doc, never by rewriting.
- **Stale (pre-substrate)** — describes the dissolved FedAvg / VL-JEPA /
  role-split paradigm. Kept for history; do not treat as current.

## Living specs (current design + built features)

| Doc | What it covers |
|-----|----------------|
| `tool_substrate.md` | v2 canonical: tools as the primary substrate item — three tiers, manifest, mint formula, combo damper, vetting greenlight, adoption rail. Phase-10 amendment inline. |
| `services_market.md` | The remote-API market rail (`ServiceMarket.sol`): registry + EIP-712 payment channels, behavioral trust, ATN-only pricing (2.5% fee recycled at settlement), no substrate standing. |
| `charter_anchor.md` | `CharterAnchor.sol` — governed anchor for the charter *version* (values stay off-chain); drift detection built, migration deferred. |
| `local_e2e.md` | The two economy proof-of-life scripts (`scripts/local_e2e_tool_economy.py`, `local_e2e_venture_loop.py`); also the reference walk-through of the venture loop. |
| `ledger_pricing.md` | Post-phase8 default: ledger (net_score tree recursion) replaces geometric equilibration for mint pricing; the arm-B context-render rule. |
| `two_plane_inference.md` | Claim graph demoted to a verdict layer; artifacts (full payloads) live in blob store + `ArtifactIndex`, referenced by sha256. |
| `epoch_economics.md` | Fixed emission (implemented, config-gated) + candle close (local path implemented; federated designed). |
| `agentic_loop.md` | Hardening spec for `send_orchestrate` (the anthropic/openai_compat/ollama agent loop) and its adapters. |
| `cross_platform_isolation_design.md` | Isolation + vault port plan — IMPLEMENTED + Linux-verified (POSIX AF_UNIX / SO_PEERCRED, tracked-PID kill). |
| `epoch_economics.md` · `ledger_pricing.md` | (see above) the economic core as shipped. |
| `auto_update_design.md` | Daemon auto-update: stage-on-poll, apply-on-next-boot (running daemon never self-restarts). Implementing. |
| `anti_tamper_design.md` | Consensus-node anti-tamper — the "other half" of auto-update. Designed, pre-implementation. |
| `unified_agent_design.md` | Fractal agent unification (orchestrator ≡ child agent) — the definitive design that drove the `atn/` runtime shape. |

## Experiment records (pre-registered — DO NOT EDIT)

| Doc | Status |
|-----|--------|
| `phase8_prereg.md` | **DO NOT EDIT** — prereg committed before any contest call ("does equilibration earn its complexity?"). |
| `phase8_results.md` | **DO NOT EDIT** — results: equilibration beat vote-count by +0.127, short of the 0.25 bar → demoted. |
| `phase9_depth_experiment.md` | **DO NOT EDIT** — the pre-committed, still-unrun final test of equilibration at graph depth. |
| `phase10_prereg.md` | **DO NOT EDIT** — prereg: "does executable ground truth deliver what prose debate couldn't?" (H1 declared FINAL). |
| `phase10_results.md` | **DO NOT EDIT** — results: H1 REFUTED; evidence-backed standing still separated defective from correct tools (AUC 1.000). |

## Reference

| Doc | What it covers |
|-----|----------------|
| `jurisdiction_wiring_map.md` | Read-only survey (2026-07-05): how the tokenized-project → on-chain surface wires together. Nothing built by the doc itself. |
| `value_props_head_to_head.md` | Pre-merge comparison: text-claim substrate vs tool substrate — what each promised and delivered. |
| `DORG_CHAT_INTERFACE_DESIGN.md` | dOrg deployment + chat-interface layer design (mostly not yet built). |
| `DRAFT_COMMON_SYSTEM_PROMPT.md` | Draft shared base system prompt for cognitive agents (final lives in `delegate_prompts.py`). |
| `TRIMMED_MEAN_USAGE.md` | Byzantine-resistant trimmed-mean aggregation usage guide (aggregator-era). |
| `BACKLOG_TRAINING_DATA.md` | Training-data pipeline analysis. **Stale (pre-substrate)** — references the VL-JEPA path. |
| `architecture/OVERVIEW.md` · `CONTRACTS.md` · `NODES.md` | Architecture reference set. Partly **stale (pre-substrate)** — verify against `CLAUDE.md` before trusting. |
| `guides/QUICKSTART.md` · `guides/TRAINING_LOOP.md` | **Stale (pre-substrate)** — the "Absolute Zero" role-split loop and `demo.py` quickstart are gone; kept for history. |
| `CHEATSHEET.md` | Common operations quick reference — partly **stale (pre-substrate)**. |
| `architecture-audit.md` | 2026-03-25 audit of orchestrator vs child-agent divergences (historical snapshot). |

## Moved from root (2026-07-06)

- [BACKLOG.md](BACKLOG.md) — THE project board: user decisions, ratified-unbuilt designs, follow-ups, experiments, critical path. Update in place.
- [PLAN.md](PLAN.md) — the original architectural blueprint (historical; the maintained map is the root README + CLAUDE.md).
- [VALIDATION_FINDINGS.md](VALIDATION_FINDINGS.md) — VL-JEPA validation record (historical — do not edit).

