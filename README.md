# Autonet Node

**Decentralized AI training, inference, and agent orchestration.**

> Alpha pre-release. Deployed on Etherlink Shadownet (testnet). Constitution not yet published on-chain. Expect breaking changes.

Autonet is a protocol for decentralized AI alignment where alignment emerges from economic incentives rather than centralized constraint. This repository contains the node runtime: the agent framework, the **world-model substrate** (graph equilibration over a charter coordinate space — the canonical training/inference path), and smart contracts. The original VL-JEPA / TextJEPA neural pipeline is no longer live; its modules remain on disk for reference only (see `VALIDATION_FINDINGS.md` for why it was shelved). For the current architecture, training loop, and directory map, see `CLAUDE.md`.

For the full protocol specification, see the [whitepaper](https://github.com/autonet-code/whitepaper).

## Install

```bash
pip install autonet-computer
```

Install tiers:

```bash
pip install autonet-computer                  # Agent framework (local operation)
pip install autonet-computer[voice]           # + Voice / TTS
pip install autonet-computer[network]         # + Blockchain, P2P, training (full node)
pip install autonet-computer[network,voice]   # Everything
```

Add extras to an existing installation:

```bash
pip install autonet-computer[voice]           # adds voice to base
pip install autonet-computer[network]         # adds network to base or base+voice
```

Or install from source:

```bash
git clone https://github.com/autonet-code/node.git
cd node
pip install -e ".[network]"
```

Start the agent framework:

```bash
atn
```

## Architecture

The node operates across three layers:

| Layer | What it does |
|-------|-------------|
| **Agent Framework (ATN)** | Agent orchestration, task delegation, tool execution, inbox messaging, WebSocket server |
| **Training & Inference** | World-model substrate (federated epoch close over a charter/claim graph), two-plane retrieval + verdict inference, alignment pricing |
| **Smart Contracts** | `Substrate.sol` — epoch anchoring, agent registry, training records, ATN token, inference payments (see `CLAUDE.md` "Smart Contracts") |

### World-Model Substrate

The substrate is a **graph equilibration architecture, not a neural network**. Instead of gradient descent over weights, it grows a tree of sub-claims under each charter tendency and prices standing through debate. This replaced VL-JEPA after its mode-collapse failures on real captioning data (see `VALIDATION_FINDINGS.md`); the substrate is content-addressed, deterministic, and converges across daemons without any neural training loop.

The claim graph is a **verdict layer**, not a knowledge index: full work-unit payloads live in the blob store plus a daemon-local `ArtifactIndex` (embedding search); claim nodes are debated judgments about artifacts. Epoch close defaults to **ledger pricing** (net_score tree recursion, no geometric propagation); full geometric equilibration is now an opt-in experimental kernel, pending the outcome of a deeper-graph test. See `CLAUDE.md` ("The Training Loop (Substrate)") and `docs/two_plane_inference.md` / `docs/ledger_pricing.md` for the current design, and `docs/phase8_results.md` for why equilibration was demoted.

The charter is 6-root (4 alignment axes — `life_precious`, `self_preservation`, `promotion_of_intelligence`, `evolution` — plus 2 usefulness axes — `correctness`, `simplicity`); see `nodes/core/constitution.py`.

**Mint vs novelty.** At an epoch boundary, per-node score movement during the epoch is reconciled into two separate signals:

- **Novelty** — descriptive measure of surprise (magnitude of score movement, including reversions). Diagnostic only.
- **Mint** — the rewarded subset. Awarded only for *positive* movement that *lands positive*, weighted by a survival factor (how much of the epoch the score-change persisted), then passed through the violator-pays mint gate. CON contributors and reverted moves don't mint.

See `nodes/common/world_model_substrate/reconcile.py` for the formula and `mint_gate.py` for the gate.

**Quickstart (substrate end-to-end).**

```bash
pip install -e c:/code/world-model
python test_world_model_substrate_e2e.py     # vertical slice: solver -> aggregator -> verifier -> inference
python test_epoch_reconciliation.py           # mint/novelty distribution across 3 solvers
python test_multi_solver_convergence.py       # content-addressed federation: 2 solvers, shared sub-claims
```

**Embedder options.** The substrate accepts coordinates from any function that maps a turn dict to charter coords. Two embedders are validated:

| Embedder | Where | Trade-off |
|----------|-------|-----------|
| `score_turn_4d` (keyword heuristic) | `nodes/common/world_model_substrate/score_turn.py` | Deterministic, free, but conservative (returns zeros on anything not keyword-matchable) |
| `turn_to_observation_via_llm` (LLM with binary-flag prompt) | Validated in `videos/SF/.../phase2/tier3a_llm_adapter.py`; not yet wired into the solver service | More decisive (commits where heuristic stays silent), 0.8% real disagreement rate vs heuristic, ~zero per-token cost via Claude Max bridge |

The LLM adapter uses a binary-commit prompt (per axis: -1 = clear flag, +1 = no flag, 0 = can't tell) — this shape produces deterministic substrate verdicts where graded scoring slips through veto thresholds. See `D:/videos/SF/manifesting/from_endstate/new physics/substrate_experiment/phase2/TIER3A_FINDINGS.md` for the validation results.

### VL-JEPA: Historical / Shelved

The original training architecture: a shared VL-JEPA (Vision-Language Joint Embedding Predictive Architecture) trained with self-supervised learning. Its modules (`ml.py`, `jepa.py`, `vl_jepa.py`, `text_jepa.py`) remain on disk but are **not part of the live training or inference loop** — the world-model substrate is the only active path. See `VALIDATION_FINDINGS.md` for why VL-JEPA was shelved (mode collapse on real images at 18M params).

### Alignment Pricing

Operations are priced based on semantic alignment with jurisdiction standards:

```
alignment = geometric_mean(user_to_jurisdiction, task_to_user, task_to_jurisdiction)
```

- High alignment: subsidized (toward free)
- Neutral: base cost
- Low alignment: premium (funds subsidies)

The same mechanism steers training rewards: capabilities the network lacks pay more to train.

## Node Roles (Historical)

The proposer/solver/coordinator/aggregator role split and the commit-reveal
"Absolute Zero" training loop (`PROPOSE -> TRAIN -> REVEAL GT -> REVEAL SOL
-> VERIFY -> REWARD -> AGGREGATE -> PUBLISH`) described in earlier versions
of this README are **gone** — those roles and their contracts were deleted.
The substrate loop (agent conversations -> events -> federated epoch close
-> anchor -> mint) is the only training path; see `CLAUDE.md` ("The
Training Loop (Substrate)") for the current diagram.

## Smart Contracts

Deployed on Etherlink Shadownet (testnet). One contract, `contracts/core/Substrate.sol`,
folds epoch anchoring, agent registry, training records, and inference
payments — the pre-substrate suite (RPB, Project.sol, TaskContract.sol,
ResultsRewards.sol, ParticipantStaking.sol, ModelShardRegistry.sol, role
staking) was deleted wholesale. See `CLAUDE.md` ("Smart Contracts") for
the current interface and deployment (`scripts/deploy_substrate.js`).

## Project Structure

```
atn/                   # Agent framework (ATN): daemon runtime, WS server (:7700), wallet auth
nodes/                 # Training/inference implementation
  core/                # Base node architecture, constitution, 4 engines
  aggregator/          # World-model event aggregation
  inference/           # Substrate inference (auto-detects substrate-shaped models)
  service.py           # Solver service; routes contributions by metrics['substrate']
  common/              # Shared: p2p, blob store, event gossip, federated close
    world_model_substrate/   # Substrate protocol layer (adapter, events, reconcile, mint_gate, infer, artifact_index)
world_model/            # Vendored substrate engine (claim graph, charter tendencies, equilibration)
contracts/             # Solidity smart contracts
  core/                # Substrate.sol (the entire on-chain surface)
experiments/            # Pre-registered contest experiments (phase7, phase8, ...)
scripts/               # Build, install, and deploy scripts (deploy_substrate.js)
```

See `CLAUDE.md` ("This Repo: Key Directories") for the authoritative, maintained map.

## Development

### Prerequisites

- Python 3.11+
- Node.js 18+ (for smart contract development)

### Deploying Contracts

```bash
npx hardhat node                                        # Local blockchain (separate terminal)
npx hardhat run scripts/deploy_substrate.js --network localhost
```

### Tests

```bash
pytest tests/test_wm_lineage.py tests/test_federated_reconcile.py   # targeted subsets only — never `pytest tests/`, ~624 tests, slow
python test_world_model_substrate_e2e.py                            # substrate e2e (top-level script)
```

See `CLAUDE.md` ("Testing") for the current targeted-subset guidance; the pre-substrate hardhat JS test suite was deleted.

## Current Status

For current state and what's next, see `CLAUDE.md` ("Current State & Mission") and `PLAN.md` / `BACKLOG.md`. Summary: substrate is the canonical (only live) training/inference path; VL-JEPA modules remain on disk for reference but are not wired into any active loop; `Substrate.sol` is deployed to Etherlink Shadownet; constitution not yet published on-chain.

## Contributing

The codebase is split into a **core-protected layer** and an **extensible surface**.

### Core-Protected Files

Seven files enforce the jurisdiction's constitutional guarantees: constitution injection into registered agents, lineage hash verification, alignment hash computation, and on-chain integrity checking. These files are hashed together into a core fingerprint published on-chain via the Registry at `node.code.hash.<version>`. The runtime periodically verifies that the installed code matches.

| File | What it protects |
|------|-----------------|
| `atn/runtime/execution_engine.py` | Constitution injection into agent executions |
| `atn/delegate_prompts.py` | Constitutional preamble template |
| `atn/agent_identity.py` | Lineage hash chain verification |
| `atn/on_chain.py` | Alignment hash computation, agent registration encoding |
| `atn/autonet_service.py` | Constitution loading from chain |
| `nodes/core/constitution.py` | Constitutional governance framework |
| `atn/_cache.py` | Integrity verification itself (obfuscated in release builds) |

Modifications to these files require a new governance-published hash.

### Extensible Surface

Everything else — providers, tools, connectors, the orchestrator loop, voice, CLI, config, prompt templates for non-constitutional layers, and the entire training pipeline — sits outside the core fingerprint and can be freely modified without breaking integrity verification.

You can:
- Add new LLM providers
- Rewrite the tool surface
- Swap out prompt templates (for non-constitutional layers)
- Extend the connector system
- Add CLI commands
- Modify training pipeline code

The node will continue to pass its on-chain integrity check.

The `_cache.py` module that performs verification is obfuscated in release builds to prevent trivial bypass, but its interface is documented: `core_fingerprint()` returns the enforced hash, `combined_fingerprint()` returns a full diagnostic hash, and `validate(rpc_url, registry_addr, version)` runs the on-chain comparison.

The boundary is intentionally narrow — seven files out of ~60 — so the community has maximum surface area to iterate on while constitutional protections remain tamper-evident.

### How to Contribute

1. Fork the repo
2. Make changes (see extensible surface above)
3. Run tests: targeted subsets per `CLAUDE.md` ("Testing") — never the full `pytest tests/`
4. Open a PR

## Related Repositories

| Repo | What |
|------|------|
| [whitepaper](https://github.com/autonet-code/whitepaper) | Protocol specification |
| [on-chain-jurisdiction](https://github.com/autonet-code/on-chain-jurisdiction) | DAO governance, trustless economy, RepToken |
| [tool-registry](https://github.com/autonet-code/tool-registry) | Open catalog of agent tools |

## License

MIT
