# Docs index

Autonet's design specs, experiment records, and reference material. For
agent/machine onboarding start at the repo-root `CLAUDE.md`; for the
one-paragraph "what is this" start at the repo-root `README.md`.

**Status (v0.7.0, 2026-07-10): beta on testnet.** `Substrate.sol` and the
DAO/economy suite are deployed on Etherlink Shadownet; the canonical
jurisdiction is a werule-platform DAO (governor `0xD5691B7c…`) with the
addresses of record in `registry.json` (daemons resolve the network from
it — repo-root in a dev checkout, else the GitHub-raw copy cached under
`~/.atn/`). The tool-economy paradigm is the shipped system: `tool_substrate.md`
is the core spec (read its `Decision (2026-07-10)` section first). The
architecture-reference set and the two guides below still describe the
retired pre-substrate paradigm and are banner-marked historical.

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
| `tool_substrate.md` | **The core spec — start here.** Tools as the primary substrate item: manifest, mint formula, combo damper, review-driven position drift, adoption rail. Read the `Decision (2026-07-10)` section first: fees-only emission, money-only close, REP claimed DAO-side, vet gate retired (tools mint from first attested third-party use). Phase-10 amendment inline. |
| `services_market.md` | The remote-API market rail (`ServiceMarket.sol`): registry + EIP-712 payment channels, behavioral trust, ATN-only pricing (2.5% fee taken at settlement through `Substrate.payForService`), no substrate standing. |
| `charter_anchor.md` | `CharterAnchor.sol` — governed anchor for the charter *version* (values stay off-chain); drift detection built, migration deferred. |
| `local_e2e.md` | The two economy proof-of-life scripts (`scripts/local_e2e_tool_economy.py`, `local_e2e_venture_loop.py`); also the reference walk-through of the venture loop. |
| `ledger_pricing.md` | Post-phase8 default: ledger (net_score tree recursion) replaces geometric equilibration for mint pricing; the arm-B context-render rule. |
| `two_plane_inference.md` | Claim graph demoted to a verdict layer; artifacts (full payloads) live in blob store + `ArtifactIndex`, referenced by sha256. |
| `epoch_economics.md` | Epoch close + candle mechanics. Note: the emission model shipped as **fees-only** (2026-07-10) — the epoch pool is burned service fees only, no base emission (Σ minted == Σ burned); see `tool_substrate.md` `Decision (2026-07-10)`. |
| `agentic_loop.md` | Hardening spec for `send_orchestrate` (the anthropic/openai_compat/ollama agent loop) and its adapters. |
| `cross_platform_isolation_design.md` | Isolation + vault port plan — IMPLEMENTED + Linux-verified (POSIX AF_UNIX / SO_PEERCRED, tracked-PID kill). |
| `secrets.md` | **Owner-facing.** The secret vault and agent security: the honest threat model, the fail-closed allowance algebra (parent ∩ child clamp), host binding, the names-only audit trail, and the leak tripwire. Linked from the app's "Add secret" dialog. |
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
| `architecture/OVERVIEW.md` · `CONTRACTS.md` · `NODES.md` | Architecture reference set. **Historical (pre-substrate)** — each carries a dated banner; the current architecture is this README + `tool_substrate.md` + `CLAUDE.md`. |
| `guides/QUICKSTART.md` | Current install/run quickstart: `pip install autonet-computer`, `atn` starts the daemon, `deploy_substrate.js` for local contracts. (The old "Absolute Zero" role-split quickstart was replaced.) |
| `guides/TRAINING_LOOP.md` | **Historical (pre-substrate)** — the "Absolute Zero" FedAvg role-split loop; native training is off the live path (see `VALIDATION_FINDINGS.md`). Banner-marked. |
| `CHEATSHEET.md` | Common operations quick reference — the contract-call/staking sections are **historical (pre-substrate)** and banner-marked; the current-command section is live. |
| `architecture-audit.md` | 2026-03-25 audit of orchestrator vs child-agent divergences (historical snapshot). |

## Moved from root (2026-07-06)

- [BACKLOG.md](BACKLOG.md) — THE project board: user decisions, ratified-unbuilt designs, follow-ups, experiments, critical path. Update in place.
- [PLAN.md](PLAN.md) — the original architectural blueprint (historical; the maintained map is the root README + CLAUDE.md).
- [VALIDATION_FINDINGS.md](VALIDATION_FINDINGS.md) — VL-JEPA validation record (historical — do not edit).

