# Autonet: Decentralized Autonomous AI Network

https://autonet.computer

Autonet is a unified framework for decentralized AI training, inference, and governance. It combines four foundational components into a coherent system:

- **Proof of Intelligence (PoI)**: Distributed AI model training with cryptoeconomic incentives
- **Recursive Principial Body (RPB)**: Constitutional governance with LLM-based consensus
- **Myco-sys**: Self-governing autonomous node architecture
- **Smart Rollup**: Layer 2 infrastructure for scalable on-chain verification

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AUTONET STACK                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Applications                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  AI Inference   │  │  Model Markets  │  │   DAO Tools     │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 3: AI Training & Inference                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Proposer Node  │  │  Solver Node    │  │  Aggregator     │              │
│  │  (Task Gen)     │  │  (Training)     │  │  (FedAvg)       │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Absolute Zero Loop: Task → Solve → Verify → Aggregate      │            │
│  └─────────────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Constitutional Governance                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Constitution   │  │  LLM Consensus  │  │  Debate System  │              │
│  │  (Principles)   │  │  (RPB Method)   │  │  (Governance)   │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Four Engines: Awareness │ Governance │ Work │ Survival     │            │
│  └─────────────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Smart Contracts (PoI Chain)                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │PoIProject│ │TaskMgmt  │ │Staking   │ │Governance│ │Token     │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 0: Rollup Infrastructure                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  AnchorBridge   │  │ DisputeManager  │  │  Checkpoint     │              │
│  │  (L1 Security)  │  │ (Challenges)    │  │  (State Roots)  │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Pastry DHT │ VRF Committee │ Plugin System                  │            │
│  └─────────────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer -1: L1 Anchor (Ethereum/Tezos)                                        │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Security Bootstrap │ Data Availability │ Final Settlement  │            │
│  └─────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Autonomous Nodes

Each node in Autonet is a sovereign "citizen-cell" that:
- Operates according to immutable constitutional principles
- Participates in governance through LLM-based consensus
- Executes work only with active network heartbeat
- Can replicate via spore mechanism

### 2. Absolute Zero Training Loop

Self-improving distributed training with commit-reveal pattern:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ABSOLUTE ZERO LOOP                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. PROPOSE      Proposer creates task with hidden ground truth          │
│       │                                                                  │
│       ▼                                                                  │
│  2. TRAIN        Solver trains model, commits solution hash              │
│       │                                                                  │
│       ▼                                                                  │
│  3. REVEAL GT    Proposer reveals ground truth (triggered by commit)     │
│       │                                                                  │
│       ▼                                                                  │
│  4. REVEAL SOL   Solver reveals solution (triggered by GT reveal)        │
│       │                                                                  │
│       ▼                                                                  │
│  5. VERIFY       Coordinators vote via Yuma consensus                    │
│       │                                                                  │
│       ▼                                                                  │
│  6. REWARD       Rewards distributed to proposer, solver, coordinators   │
│       │                                                                  │
│       ▼                                                                  │
│  7. AGGREGATE    Aggregator performs FedAvg on verified updates          │
│       │                                                                  │
│       ▼                                                                  │
│  8. PUBLISH      Global model published on-chain via setMatureModel      │
│       │                                                                  │
│       └──────────────────► Loop continues with new tasks                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Status: Fully operational** - Complete loop demonstrated with real PyTorch training

### 3. Decentralized Training and Inference

Autonet enables AI model training and inference without centralized infrastructure:

**Training**: Multiple solver nodes train on local data, submit weight updates verified by coordinators, and aggregate via FedAvg. No single entity controls training data or model updates.

**Model Storage**: Trained model weights are sharded across storage providers via ModelShardRegistry:
- Merkle proofs verify shard integrity
- Erasure coding (k data + n parity shards) ensures availability even if providers go offline
- Staked providers are economically incentivized to maintain uptime

**Inference**: Published models can be queried via `requestInference()`. Providers stake ATN and compete to serve requests.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED WEIGHT STORAGE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Model (8M params)                                                      │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────┐                                                        │
│   │ Shard Model │  4 data shards + 1 parity                             │
│   └─────────────┘                                                        │
│         │                                                                │
│   ┌─────┼─────┬─────┬─────┬─────┐                                       │
│   ▼     ▼     ▼     ▼     ▼     ▼                                       │
│ [S0]  [S1]  [S2]  [S3]  [P0]   ← Each uploaded to IPFS                 │
│   │     │     │     │     │                                             │
│   ▼     ▼     ▼     ▼     ▼                                             │
│ ┌───────────────────────────────┐                                       │
│ │   ModelShardRegistry.sol      │  On-chain coordination               │
│ │   - registerModel(merkleRoot) │                                       │
│ │   - announceShard(provider)   │                                       │
│ │   - checkAvailability()       │                                       │
│ └───────────────────────────────┘                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4. Constitutional Governance

Decisions requiring conceptual reasoning use LLM-based consensus:
- Multi-modal language models evaluate proposals
- Constitutional standards constrain all decisions
- Stake-weighted voting for economic alignment
- High quorum (95%) for constitutional amendments

### 5. Dual Token Economics

- **ATN (Autonoma Token)**: Native token for gas, staking, rewards, and governance
- **Project Tokens (PT)**: Project-specific tokens for investment and revenue sharing

## Directory Structure

```
autonet/
├── contracts/              # Solidity smart contracts
│   ├── core/              # Core PoI contracts
│   ├── rollup/            # L2 infrastructure contracts
│   ├── governance/        # DAO and debate contracts
│   ├── tokens/            # ATN and Project tokens
│   └── interfaces/        # Contract interfaces
├── nodes/                 # Node implementations
│   ├── core/              # Base node architecture
│   │   ├── constitution.py
│   │   ├── node.py
│   │   └── engines.py
│   ├── proposer/          # Task proposer node
│   ├── solver/            # Training solver node
│   ├── coordinator/       # Verification coordinator
│   ├── aggregator/        # Model aggregator
│   └── common/            # Shared utilities
├── rollup/                # Rollup node (Java)
│   ├── consensus/
│   ├── optimistic/
│   ├── committee/
│   └── plugins/
├── sdk/                   # Client libraries
│   ├── python/
│   └── typescript/
├── docs/                  # Documentation
│   ├── architecture/
│   ├── consensus/
│   ├── tokenomics/
│   └── guides/
├── config/                # Configuration files
├── scripts/               # Deployment and utility scripts
└── tests/                 # Test suites
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Java 17+
- Docker & Docker Compose
- Foundry or Hardhat

### Installation

```bash
# Clone the repository
git clone https://github.com/autonet/autonet.git
cd autonet

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Build Java rollup node
cd rollup && mvn package && cd ..

# Deploy contracts to local testnet
docker-compose up -d
npm run deploy:local
```

### Running the Multi-Node Orchestrator

The easiest way to run a complete training cycle is with the orchestrator:

```bash
# Start local Hardhat node (in a separate terminal)
npx hardhat node

# Deploy contracts
npx hardhat run scripts/deploy.js --network localhost

# Run full training cycle with default configuration (1P/1S/2C/1A)
python orchestrator.py

# Run with custom configuration
python orchestrator.py --proposers 1 --solvers 2 --coordinators 2 --aggregators 1 --rounds 3 --delay 2.0
```

The orchestrator will:
1. Deploy all contracts
2. Distribute ATN tokens to node accounts
3. Create and fund a test project
4. Spawn autonomous nodes (proposers, solvers, coordinators, aggregators)
5. Run training cycles and display validation results

### Running Individual Nodes

```bash
# Start a solver node
python nodes/solver/main.py --config config/solver.toml

# Start a proposer node
python nodes/proposer/main.py --config config/proposer.toml

# Start the rollup node
java -jar rollup/target/autonet-rollup.jar
```

## Key Components

### Smart Contracts

| Contract | Purpose |
|----------|---------|
| `Project.sol` | Manages AI development projects, funding, model publishing, and inference services |
| `TaskContract.sol` | Task lifecycle: proposal, checkpoints, solution commitment |
| `ResultsRewards.sol` | Multi-coordinator Yuma voting, rewards distribution, slashing |
| `ParticipantStaking.sol` | Role-based staking for Proposers, Solvers, Coordinators, Aggregators |
| `ModelShardRegistry.sol` | Distributed model weight storage with Merkle proofs and erasure coding |
| `ForcedErrorRegistry.sol` | Injects forced errors for coordinator vigilance testing |
| `ATNToken.sol` | ERC20Votes governance token |
| `AnchorBridge.sol` | L1/L2 checkpoint storage and token bridge |
| `DisputeManager.sol` | Stake-weighted dispute resolution |
| `AutonetDAO.sol` | On-chain governance for parameter changes |

### Node Engines

Each autonomous node operates four specialized engines:

1. **AwarenessEngine**: Environmental perception and network sensing
2. **GovernanceEngine**: Principle validation and collective decision-making
3. **WorkEngine**: Instruction execution (dependent on consensus heartbeat)
4. **SurvivalEngine**: Self-preservation and network replication

### Consensus Mechanisms

1. **PoS Chain Consensus**: ATN-staked validators secure the L2 chain
2. **Absolute Zero Consensus**: LLM-based agreement on conceptual decisions
3. **Optimistic Verification**: Challenge-based validation with stake-weighted voting

## Tokenomics

### ATN Token Utility

- Gas fees on Autonet chain
- Staking for validator/sequencer roles
- Staking for AI participant roles (Proposer, Solver, Coordinator, Aggregator)
- Rewards for training contributions
- Inference payments
- Governance voting

### Staking Requirements

| Role | Minimum Stake | Lock Period |
|------|---------------|-------------|
| Proposer | 100 ATN | 7 days |
| Solver | 50 ATN | 3 days |
| Coordinator | 500 ATN | 14 days |
| Aggregator | 1000 ATN | 14 days |

## Governance

### Decision Types

**Suitable for LLM Consensus (RPB Method):**
- Task acceptance/rejection requiring alignment assessment
- Complexity estimation for non-deterministic requests
- Training validation where ground truth is ambiguous
- Constitutional interpretation
- Dispute resolution requiring contextual judgment

**Standard On-Chain Governance:**
- Parameter changes
- Token allocation
- Protocol upgrades

### Constitutional Standards

The Constitution defines immutable principles that constrain all node behavior:

```python
CORE_PRINCIPLES = [
    "P1: PRESERVE AND EXPAND THE NETWORK IN A SUSTAINABLE MANNER.",
    "P2: UPHOLD THE SANCTITY AND IMMUTABILITY OF THIS CONSTITUTION.",
    "P3: ADVANCE HUMAN RIGHTS AND INDIVIDUAL AUTONOMY.",
    "P4: MINIMIZE SUFFERING AND HARM TO SENTIENT BEINGS.",
    "P5: ENSURE TRANSPARENT AND VERIFIABLE AI TRAINING.",
    "P6: MAINTAIN ECONOMIC FAIRNESS IN REWARD DISTRIBUTION."
]
```

## Development Roadmap

### Phase 1: Foundation ✅ COMPLETE
- Core smart contracts deployed (Project, Task, Staking, Rewards, DAO)
- Basic node implementations (Proposer, Solver, Coordinator, Aggregator)
- Local testnet demonstration with Hardhat
- Single training cycle proof-of-concept
- 13/13 Hardhat tests passing

### Phase 2: Distributed Training ✅ COMPLETE
- Multi-coordinator Yuma voting consensus
- FedAvg aggregator with real PyTorch weight aggregation
- Multi-node training cycles with orchestrator
- Mock IPFS integration for model storage
- Global model publishing via `setMatureModel`
- Forced error registry for coordinator vigilance testing
- Complete Absolute Zero loop operational

### Phase 2.5: Self-Supervised Learning & Distributed Weights ✅ COMPLETE
- **JEPA (Joint Embedding Predictive Architecture)** integration
  - Self-supervised learning without labeled data
  - Masked patch prediction in embedding space
  - EMA target encoder for stable training
- **Distributed model weights via ModelShardRegistry**
  - Layer-wise model sharding with Merkle proofs
  - Erasure coding (data + parity shards) for fault tolerance
  - On-chain shard availability tracking
  - Storage provider staking and incentives
- **End-to-end tests passing:**
  - `test_jepa_e2e.py`: Local distributed training pipeline
  - `test_jepa_onchain.py`: On-chain weight coordination

### Phase 3: Constitutional Governance (In Progress)
- LLM consensus integration
- Constitutional amendment process
- Dispute resolution system
- Stake-weighted voting

### Phase 4: Production
- Mainnet deployment
- Token launch
- Inference marketplace
- Ecosystem expansion

## Contributing

Make a PR

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Autonet is built on the foundational work of:
- Proof of Intelligence (PoI) research
- Recursive Principial Body (RPB) governance model
- Myco-sys autonomous agent architecture
- Smart Rollup L2 infrastructure
