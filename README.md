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

Self-improving distributed training:
1. **Proposers** generate training tasks with ground-truth solutions
2. **Solvers** perform local training on data shards
3. **Coordinators** verify task completion quality
4. **Aggregators** combine verified updates into improved global models

### 3. Constitutional Governance

Decisions requiring conceptual reasoning use LLM-based consensus:
- Multi-modal language models evaluate proposals
- Constitutional standards constrain all decisions
- Stake-weighted voting for economic alignment
- High quorum (95%) for constitutional amendments

### 4. Dual Token Economics

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

### Running a Node

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
| `PoIProject.sol` | Manages AI development projects, funding, and inference services |
| `TaskContract.sol` | Task lifecycle: proposal, claiming, solution commitment |
| `ResultsRewards.sol` | Verification, rewards distribution, slashing |
| `ParticipantStaking.sol` | Role-based staking for Proposers, Solvers, Coordinators |
| `ATNToken.sol` | ERC20Votes governance token |
| `AnchorBridge.sol` | L1/L2 checkpoint storage and token bridge |
| `DisputeManager.sol` | Stake-weighted dispute resolution |
| `Constitution.sol` | On-chain constitutional standards |

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

### Phase 1: Foundation
- Core smart contracts deployed
- Basic node implementations
- Local testnet demonstration
- Single training cycle proof-of-concept

### Phase 2: Distributed Training
- Full Coordinator verification logic
- Aggregator implementation
- Multi-node training cycles
- IPFS integration for model storage

### Phase 3: Constitutional Governance
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
