# Autonet Node

**Decentralized AI training, inference, and agent orchestration.**

> Alpha pre-release. Deployed on Etherlink Shadownet (testnet). Constitution not yet published on-chain. Expect breaking changes.

Autonet is a protocol for decentralized AI alignment where alignment emerges from economic incentives rather than centralized constraint. This repository contains the node runtime: the agent framework, the distributed training pipeline, smart contracts, and two interchangeable model architectures — the original VL-JEPA / TextJEPA neural pipeline and a newer **world-model substrate** (graph equilibration over a charter coordinate space). Either architecture plugs into the same protocol slots; smart contracts are unchanged.

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
| **Training & Inference** | Distributed training (VL-JEPA / TextJEPA *or* world-model substrate), two-speed inference, trace encoding, alignment pricing |
| **Smart Contracts** | Agent registration, training rewards, inference revenue splitting, staking, governance |

The training/inference layer supports two interchangeable architectures. Pick one per deployment via routing flags — the rest of the stack (proposer, coordinator, smart contracts, staking, rewards) is identical.

### World-Model Substrate (alternative to VL-JEPA)

The substrate is a **graph equilibration architecture, not a neural network**. Instead of gradient descent over weights, it grows a tree of sub-claims under each charter tendency and equilibrates stake/score until the network agrees. This was added after VL-JEPA's mode-collapse failures on real captioning data (see `VALIDATION_FINDINGS.md`); the substrate is content-addressed, deterministic, and converges across solvers without any neural training loop.

**Charter coordinate space.** The substrate operates in a 4D space defined by the four charter tendencies:

| Axis | Tendency | Thesis |
|------|----------|--------|
| 0 | `life_precious` | Life is precious and should be preserved. |
| 1 | `self_preservation` | The system should preserve its own continuity. |
| 2 | `promotion_of_intelligence` | Intelligence in any form should be promoted. |
| 3 | `evolution` | Forward advancement of capability is desirable. |

Each agent turn becomes a 4D observation `(life, self_pres, intelligence, evolution)`. The solver replays these into a `World` and records `SubClaimSprouted` / `ObservationAdded` events; the aggregator merges events; the verifier replays them onto a seed world and scores the gap reduction.

**Mint vs novelty.** At an epoch boundary, per-node score movement during the epoch is reconciled into two separate signals:

- **Novelty** — descriptive measure of surprise (magnitude of score movement, including reversions). Diagnostic only.
- **Mint** — the rewarded subset. Awarded only for *positive* movement that *lands positive*, weighted by a survival factor (how much of the epoch the score-change persisted). CON contributors and reverted moves don't mint. This is the value reported through `RPB.recordTraining`.

See `nodes/common/world_model_substrate/reconcile.py` for the formula.

**Routing flags.**

| Layer | Flag / signal | Effect |
|-------|---------------|--------|
| Aggregator (`nodes/aggregator/main.py`) | `aggregation_method='world_model'` | Replays event streams onto a charter world, runs `reconcile_epoch`, publishes a serialized world as the global model |
| Solver service (`nodes/service.py`) | Auto-detects via `metrics['substrate'] == 'world-model'` | Routes the contribution as an event payload instead of a tensor delta |
| Inference (`nodes/inference/main.py`) | Auto-detects substrate-shaped global models (presence of `'world_model'` key or `substrate == 'world-model'`) | Runs `infer_with_world_model` instead of the JEPA decoder |

**Quickstart (substrate end-to-end).**

```bash
pip install -e c:/code/world-model
python test_world_model_substrate_e2e.py     # vertical slice: solver -> aggregator -> verifier -> inference
python test_epoch_reconciliation.py           # mint/novelty distribution across 3 solvers
python test_multi_solver_convergence.py       # content-addressed federation: 2 solvers, shared sub-claims
```

**Embedder options.** The substrate accepts coordinates from any function that maps a turn dict to 4D charter coords. Two embedders are validated:

| Embedder | Where | Trade-off |
|----------|-------|-----------|
| `score_turn_4d` (keyword heuristic) | `nodes/common/world_model_substrate/score_turn.py` | Deterministic, free, but conservative (returns zeros on anything not keyword-matchable) |
| `turn_to_observation_via_llm` (LLM with binary-flag prompt) | Validated in `videos/SF/.../phase2/tier3a_llm_adapter.py`; not yet wired into the solver service | More decisive (commits where heuristic stays silent), 0.8% real disagreement rate vs heuristic, ~zero per-token cost via Claude Max bridge |

The LLM adapter uses a binary-commit prompt (per axis: -1 = clear flag, +1 = no flag, 0 = can't tell) — this shape produces deterministic substrate verdicts where graded scoring slips through veto thresholds. See `D:/videos/SF/manifesting/from_endstate/new physics/substrate_experiment/phase2/TIER3A_FINDINGS.md` for the validation results.

### VL-JEPA: The Distributed Model

The original training architecture: a shared [VL-JEPA](https://github.com/autonet-code/whitepaper) (Vision-Language Joint Embedding Predictive Architecture) trained with self-supervised learning. No labeled data required. (Still present in the codebase; the world-model substrate above is an alternative, not a replacement.)

The model is split between local and network:

- **Network-side (distributed):** Visual encoder, text encoder, cross-modal fusion, semantic predictor. These components are trained collaboratively across nodes via federated averaging with Byzantine-resistant aggregation. Weight updates are verified on-chain through a commit-reveal protocol.
- **Local-side (on your device):** Text decoder with FiLM conditioning. Runs autoregressive generation from the network's latent plan. Only the compact K-vector (~8-32 KB) traverses the network per turn, regardless of output length.

Training is anchored in economic utility: agent execution traces from real work (verified through the trustless economy) serve as training data. The model improves as the economy grows.

### Two-Speed Inference

- **Fast path:** Local decoder generates tokens at GPU speed from cached latent plans. Handles ~60% of queries.
- **Slow path:** Network VL-JEPA reasons in embedding space about complex or novel queries, streaming updated guidance embeddings back to the local node.

Network unavailable? The local decoder runs standalone. The system degrades gracefully.

### Alignment Pricing

Operations are priced based on semantic alignment with jurisdiction standards:

```
alignment = geometric_mean(user_to_jurisdiction, task_to_user, task_to_jurisdiction)
```

- High alignment: subsidized (toward free)
- Neutral: base cost
- Low alignment: premium (funds subsidies)

The same mechanism steers training rewards: capabilities the network lacks pay more to train.

## Node Types

The training loop uses four specialized node roles:

| Role | Stake | Function |
|------|-------|----------|
| **Proposer** | 100 ATN | Generates training tasks with hidden ground truth |
| **Solver** | 50 ATN | Trains model on tasks, commits solution hashes |
| **Coordinator** | 500 ATN | Verifies solutions via Yuma consensus voting |
| **Aggregator** | 1000 ATN | Performs FedAvg on verified weight updates, publishes global model |

### Training Loop (Absolute Zero)

```
PROPOSE -> TRAIN -> REVEAL GT -> REVEAL SOL -> VERIFY -> REWARD -> AGGREGATE -> PUBLISH
```

Commit-reveal pattern ensures solvers train honestly: solutions are hashed before ground truth is revealed.

## Smart Contracts

Deployed on Etherlink Shadownet (testnet). Contract discovery requires only the Governor address:

```
Governor.token()        -> RepToken
Governor.timelock()     -> Timelock
RepToken.registryAddress()  -> Registry
Registry.getRegistryValue("rpb.contract") -> RPB
```

Key contracts:

| Contract | Purpose |
|----------|---------|
| `RPB` | Agent registration, training rewards, inference revenue splitting, shares, sponsorship |
| `Project.sol` | AI project management, funding, model publishing |
| `TaskContract.sol` | Task lifecycle with commit-reveal |
| `ResultsRewards.sol` | Multi-coordinator Yuma voting and reward distribution |
| `ParticipantStaking.sol` | Role-based staking |
| `ModelShardRegistry.sol` | Distributed weight storage with Merkle proofs and erasure coding |
| `ATNToken.sol` | ERC20Votes governance token |

## Project Structure

```
atn/                   # Agent framework (ATN)
  runtime/             # Scheduler, orchestrator, WebSocket server
  connectors/          # Modular tool connectors
  _cache.py            # Execution integrity verification
nodes/                 # Training node implementations
  core/                # Base node architecture, constitution, 4 engines
  proposer/            # Task generation
  solver/              # Model training (JEPA or substrate)
  coordinator/         # Verification voting
  aggregator/          # FedAvg or world-model event aggregation
  inference/           # Two-speed inference, auto-detects substrate vs JEPA models
  service.py           # Solver service; routes contributions by metrics['substrate']
  common/              # Shared: blockchain, ML, JEPA, VL-JEPA
    world_model_substrate/   # Graph-equilibration substrate (charter, events, reconcile)
contracts/             # Solidity smart contracts
  core/                # Project, Task, Staking, Rewards, ModelShardRegistry
  tokens/              # ATN governance token
  governance/          # DAO contract
scripts/               # Build and install scripts
```

## Development

### Prerequisites

- Python 3.11+
- Node.js 18+ (for smart contract development)

### Running the Training Simulation

```bash
# Start local Hardhat node
npx hardhat node

# Deploy contracts
npx hardhat run scripts/deploy.js --network localhost

# Run full training cycle
python orchestrator.py

# Custom configuration
python orchestrator.py --proposers 1 --solvers 2 --coordinators 2 --aggregators 1
```

### Tests

```bash
npx hardhat test                              # Smart contract tests
pytest                                        # Python tests
python test_world_model_substrate_e2e.py      # Substrate vertical slice
python test_epoch_reconciliation.py           # Per-agent mint / novelty
python test_multi_solver_convergence.py       # Content-addressed federation
```

## Current Status

**What works:**
- Agent framework with full lifecycle management
- Training loop simulation (Absolute Zero) with all node types
- Smart contracts deployed and tested on local Hardhat
- VL-JEPA architecture validated on synthetic data (mode-collapses on real COCO — see `VALIDATION_FINDINGS.md`)
- Federated averaging with Byzantine-resistant aggregation
- Constitutional governance engine (4 engines per node)
- Execution integrity self-verification against on-chain hash
- **World-model substrate vertical slice:** solver -> aggregator -> verifier -> inference, all using the graph-equilibration engine instead of VL-JEPA
- **Per-agent mint computation** with novelty/mint distinction (descriptive surprise vs rewarded positive-and-persistent movement); CON contributors don't mint
- **Multi-solver content-addressed convergence:** independent solvers proposing the same sub-claim resolve to the same node by id; the aggregator dedupes naturally
- **Substrate wiring:** `aggregation_method='world_model'` in `nodes/aggregator/main.py`, auto-detection via `metrics['substrate']` in `nodes/service.py`, auto-detection of substrate-shaped global models in `nodes/inference/main.py`

**What's next:**
- Testnet deployment of RPB contract on Etherlink Shadownet
- Wire real VL-JEPA training into solver nodes (currently mocked); substrate path is real
- P2P node discovery and weight replication
- Inference marketplace
- Constitution published on-chain

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
3. Run tests: `npx hardhat test && pytest`
4. Open a PR

## Related Repositories

| Repo | What |
|------|------|
| [whitepaper](https://github.com/autonet-code/whitepaper) | Protocol specification |
| [on-chain-jurisdiction](https://github.com/autonet-code/on-chain-jurisdiction) | DAO governance, trustless economy, RepToken |
| [tool-registry](https://github.com/autonet-code/tool-registry) | Open catalog of agent tools |

## License

MIT
