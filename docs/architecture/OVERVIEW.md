# Autonet Architecture Overview

> **HISTORICAL — pre-substrate (banner added 2026-07-10).** This document
> describes the retired FedAvg / role-split paradigm (Project, TaskContract,
> ParticipantStaking, ResultsRewards, AnchorBridge, Proposer/Solver/
> Coordinator/Aggregator nodes). **None of that is the live system.** Those
> contracts were deleted in Phase 5.6a. The current architecture is:
> - **Substrate paradigm, tool economy.** The on-chain surface is four
>   contracts under `contracts/core/`: `Substrate.sol` (epoch anchoring,
>   agent registry, money-only training records, ATN token, service
>   payments), `ServiceMarket.sol`, `VentureVault.sol`, `CharterAnchor.sol`.
> - **Fees-only emission, money-only close** (v0.7.0, 2026-07-10): the epoch
>   pool = burned service fees only; REP is claimed DAO-side from ATN
>   earnings 1:1 (no reputation surface in `Substrate.sol`).
> - **Read instead:** the repo-root `README.md` (living paper), the docs
>   index `docs/README.md`, `docs/tool_substrate.md` (core spec), and
>   `CLAUDE.md`. The body below is kept only for history.

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
