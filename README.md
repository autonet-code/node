# Autonet Node

**Decentralized AI training, inference, and agent orchestration.**

> Alpha pre-release. Deployed on Etherlink Shadownet (testnet). Constitution not yet published on-chain. Expect breaking changes.

Autonet is a protocol for decentralized AI alignment where alignment emerges from economic incentives rather than centralized constraint. This repository contains the node runtime: the agent framework, the distributed training pipeline, smart contracts, and the VL-JEPA model architecture.

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
| **Training & Inference** | VL-JEPA distributed training, two-speed inference, trace encoding, alignment pricing |
| **Smart Contracts** | Agent registration, training rewards, inference revenue splitting, staking, governance |

### VL-JEPA: The Distributed Model

The network trains a shared [VL-JEPA](https://github.com/autonet-code/whitepaper) (Vision-Language Joint Embedding Predictive Architecture) model using self-supervised learning. No labeled data required.

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
  solver/              # Model training
  coordinator/         # Verification voting
  aggregator/          # FedAvg weight aggregation
  common/              # Shared: blockchain, ML, JEPA, VL-JEPA
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
npx hardhat test              # Smart contract tests
pytest                        # Python tests
```

## Current Status

**What works:**
- Agent framework with full lifecycle management
- Training loop simulation (Absolute Zero) with all node types
- Smart contracts deployed and tested on local Hardhat
- VL-JEPA architecture validated on synthetic data
- Federated averaging with Byzantine-resistant aggregation
- Constitutional governance engine (4 engines per node)
- Execution integrity self-verification against on-chain hash

**What's next:**
- Testnet deployment of RPB contract on Etherlink Shadownet
- Wire real VL-JEPA training into solver nodes (currently mocked)
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
