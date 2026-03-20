# Autonet Architecture Overview

## System Layers

### Layer -1: L1 Anchor (Ethereum/Tezos)
- Security bootstrapping
- Data availability
- Final settlement for disputes

### Layer 0: Rollup Infrastructure
- **AnchorBridge**: Stores checkpoint roots, manages token bridge
- **DisputeManager**: Stake-weighted voting on challenges
- Validators submit periodic state roots

### Layer 1: Application Contracts
- **Project**: AI project lifecycle, funding, inference
- **TaskContract**: Training task management
- **ParticipantStaking**: Role-based staking
- **ResultsRewards**: Verification and rewards
- **AutonetDAO**: Governance

### Layer 2: Node Software
- Constitutional framework (immutable principles)
- Four engines: Awareness, Governance, Work, Survival
- Specialized nodes: Proposer, Solver, Coordinator, Aggregator

### Layer 3: Storage (Node-Native Blob Store)
- Model weights
- Task specifications
- Solutions and verification reports
- Referenced on-chain via content hashes

## Data Flow

```
                    ┌─────────────┐
                    │   Project   │
                    │  (funding)  │
                    └──────┬──────┘
                           │ creates
                           ▼
┌──────────┐     ┌─────────────────┐     ┌────────────┐
│ Proposer │────▶│  TaskContract   │◀────│   Solver   │
│  Node    │     │  (task specs)   │     │   Node     │
└──────────┘     └────────┬────────┘     └────────────┘
      │                   │                     │
      │ ground truth      │ task active         │ solution
      ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────┐
│                  ResultsRewards                      │
│         (reveals, verification, rewards)            │
└──────────────────────┬──────────────────────────────┘
                       │ verified updates
                       ▼
              ┌─────────────────┐
              │   Aggregator    │
              │ (FedAvg → new   │
              │  global model)  │
              └─────────────────┘
```

## Token Economics

### ATN (Autonoma Token)
- Gas on Autonet chain
- Staking for roles
- Rewards for participation
- Governance voting

### Project Tokens (PT)
- Project-specific shares
- Inference discounts
- Revenue sharing

## Consensus Mechanisms

### Chain Consensus (PoS)
- ATN-staked validators
- Checkpoint submission to L1

### Training Consensus
- Commit-reveal for solutions
- Coordinator verification
- Challenge period for disputes

### Governance Consensus
- Stake-weighted voting
- 20% quorum, 66% supermajority
- Constitutional amendments: 95% quorum
