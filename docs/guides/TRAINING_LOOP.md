# The Absolute Zero Training Loop

> **HISTORICAL — pre-substrate (banner added 2026-07-10).** This document
> describes the retired FedAvg role-split loop
> (Proposer→Solver→Coordinator→Aggregator, commit-reveal, `TaskContract`/
> `ResultsRewards`/`DisputeManager`, Project Tokens). **None of it is the
> live system.** Those contracts were deleted in Phase 5.6a, and native
> model training is **off the live path** — VL-JEPA was shelved on
> mode-collapse findings (see `docs/VALIDATION_FINDINGS.md`).
>
> The shipped loop is the **tool economy**: agents publish content-addressed
> tools; attested third-party usage mints ATN at the federated epoch close
> (fees-only pool, money-only close); reviews drive each tool's position in
> charter space. **Read instead:** `docs/tool_substrate.md` (core spec,
> `Decision (2026-07-10)` section first), `docs/README.md`, and `CLAUDE.md`.
> The body below is kept only for history.

This document explains the complete training cycle that enables decentralized AI model improvement.

## Overview

The "Absolute Zero" loop is a self-improving training cycle where:
1. Tasks are generated to challenge the model
2. Solvers train on tasks
3. Results are verified
4. Verified updates are aggregated
5. The global model improves
6. Repeat

## Phase 1: Task Proposal

**Actor**: Proposer Node (staked with 100+ ATN)

```
Proposer                       Blob Store                 Blockchain
   │                               │                          │
   │──── Upload task spec ────────▶│                          │
   │◀─── Return task_spec_cid ─────│                          │
   │                               │                          │
   │──── Upload ground truth ─────▶│                          │
   │◀─── Return gt_cid ────────────│                          │
   │                               │                          │
   │──── proposeTask(projectId, ───────────────────────────────▶
   │     hash(spec_cid),                                      │
   │     hash(gt_cid),                                        │
   │     r_propose, r_solve)                                  │
   │◀─── Return task_id ───────────────────────────────────────│
```

**Key points**:
- Ground truth is uploaded but only the HASH is committed on-chain
- Task auto-activates (or can require curator approval)
- Proposer stakes reputation on task quality

## Phase 2: Solver Training

**Actor**: Solver Node (staked with 50+ ATN)

```
Solver                         Blob Store                 Blockchain
   │                               │                          │
   │◀─── Listen for TaskActivated ─────────────────────────────│
   │                               │                          │
   │──── Download task spec ──────▶│                          │
   │◀─── Return spec data ─────────│                          │
   │                               │                          │
   │──── Download global model ───▶│                          │
   │◀─── Return model weights ─────│                          │
   │                               │                          │
   │     [LOCAL TRAINING]          │                          │
   │                               │                          │
   │──── Upload model update ─────▶│                          │
   │◀─── Return update_cid ────────│                          │
   │                               │                          │
   │──── commitSolution(taskId, ───────────────────────────────▶
   │     hash(update_cid))                                    │
```

**Key points**:
- Multiple solvers can work on the same task
- Solutions are committed as hashes first (commit-reveal scheme)
- Prevents copying from other solvers

## Phase 3: Reveal Phase

**Actors**: Proposer reveals first, then Solvers

```
Proposer                         Blockchain               Solver
   │                                │                       │
   │── revealGroundTruth(taskId, ──▶│                       │
   │   gt_cid)                      │                       │
   │                                │                       │
   │                                │◀── revealSolution ────│
   │                                │    (taskId, update_cid)│
```

**Key points**:
- Proposer reveals ground truth CID
- Contract verifies CID hashes match committed hashes
- Deadline enforced for reveals

## Phase 4: Verification

**Actor**: Coordinator Node (staked with 500+ ATN)

```
Coordinator                    Blob Store                 Blockchain
   │                               │                          │
   │──── Download ground truth ───▶│                          │
   │◀─── Return gt data ───────────│                          │
   │                               │                          │
   │──── Download solution ───────▶│                          │
   │◀─── Return solution data ─────│                          │
   │                               │                          │
   │     [COMPARE & SCORE]         │                          │
   │                               │                          │
   │──── Upload report ───────────▶│                          │
   │◀─── Return report_cid ────────│                          │
   │                               │                          │
   │──── submitVerification( ──────────────────────────────────▶
   │     taskId, solver,                                      │
   │     isCorrect, score,                                    │
   │     report_cid)                                          │
```

**Key points**:
- Coordinator compares solution metrics against ground truth
- Score: 0-100 based on accuracy match
- Incorrect solutions don't receive rewards

## Phase 5: Rewards

**Automatic** after verification

```
ResultsRewards Contract
   │
   │──── if isCorrect:
   │     │
   │     ├── Pay Coordinator: 1 ATN (fixed fee)
   │     ├── Pay Solver: r_solve (e.g., 5 ATN)
   │     └── Pay Proposer: r_propose (e.g., 10 ATN)
   │
   │──── if !isCorrect:
   │     │
   │     └── Pay Coordinator: 1 ATN only
```

**Key points**:
- Rewards come from Project's task budget
- Proposer rewarded for generating useful tasks
- Solver rewarded for quality solutions

## Phase 6: Aggregation

**Actor**: Aggregator Node (staked with 1000+ ATN)

```
Aggregator                     Blob Store                 Blockchain
   │                               │                          │
   │◀── Query verified solutions ──────────────────────────────│
   │                               │                          │
   │──── Download all updates ────▶│                          │
   │◀─── Return updates ───────────│                          │
   │                               │                          │
   │     [FEDERATED AVERAGING]     │                          │
   │                               │                          │
   │──── Upload new global model ─▶│                          │
   │◀─── Return new_model_cid ─────│                          │
   │                               │                          │
   │──── setMatureModel( ──────────────────────────────────────▶
   │     projectId, new_model_cid)                            │
```

**Federated Averaging**:
```python
def fedavg(updates, current_model):
    # Weight by training samples or equal weight
    weights = [1/len(updates)] * len(updates)

    # Average each parameter
    new_params = {}
    for key in updates[0].keys():
        new_params[key] = sum(w * u[key] for w, u in zip(weights, updates))

    return new_params
```

## Loop Iteration

After aggregation, the improved global model is available for:
1. New training tasks (continue improvement)
2. Inference (if model meets quality threshold)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌──────────┐    ┌────────┐    ┌───────────┐    ┌─────┐  │
│    │ Proposer │───▶│ Solver │───▶│Coordinator│───▶│Aggr.│  │
│    └──────────┘    └────────┘    └───────────┘    └──┬──┘  │
│         ▲                                            │      │
│         │                                            │      │
│         └────────────────────────────────────────────┘      │
│                     (New global model)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Economic Flow

```
Investors ──ATN──▶ Project ──PT──▶ Investors

Project Budget
    │
    ├──▶ r_propose ──▶ Proposer
    ├──▶ r_solve   ──▶ Solver
    └──▶ coord_fee ──▶ Coordinator

Deployed Model
    │
    └──▶ Inference Fees ──ATN──▶ Project ──▶ PT Holders (revenue share)
```

## Dispute Resolution

If verification is challenged:

1. Challenger creates dispute with `DisputeManager.createDispute(contentHash)`
2. ATN holders vote (stake-weighted)
3. 20% quorum required
4. 66% supermajority to overturn
5. If overturned: slashing applied, rewards reversed
