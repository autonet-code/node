# Autonet Cheat Sheet

Quick reference for common operations.

> **Banner (2026-07-10).** The **Current Operations** section below is live.
> Everything under **Historical (pre-substrate)** — the staking / project /
> task / commit-reveal / verification / dispute contract calls, the role
> stake table, and the task-status flow — describes contracts **deleted in
> Phase 5.6a** and is kept only for history. The live on-chain surface is
> `Substrate.sol` + `ServiceMarket.sol` + `VentureVault.sol` +
> `CharterAnchor.sol`; see `docs/tool_substrate.md` and `docs/README.md`.

## Current Operations

### Install & run the daemon
```bash
pip install autonet-computer      # installs the full node (autonet-computer)
atn                               # start the daemon + interactive console
```
In the console: `agents`, `run <id>`, `msg <id> <text>`, `activate <id>`,
`tools` / `approve <digest>`, `reconcile` (after a redeploy), `usage`,
`restart`, `help`, `quit`.

### Deploy contracts locally
```bash
npx hardhat node                                                       # local chain :8545
npx hardhat run scripts/deploy_substrate.js       --network localhost  # Substrate.sol
npx hardhat run scripts/deploy_service_market.js  --network localhost  # ServiceMarket.sol
npx hardhat run scripts/deploy_charter_anchor.js  --network localhost  # CharterAnchor.sol
npx hardhat run scripts/deploy_economy.js         --network localhost  # economy wiring
npx hardhat compile
```
(There is no `deploy.js` — removed with the pre-substrate contracts.)

### Economy proof-of-life
```bash
python scripts/local_e2e_tool_economy.py    # register → publish → adopt → epoch mint
python scripts/local_e2e_venture_loop.py    # venture funding + service revenue loop
```

### Tests (never run the whole suite — ~624 tests, slow)
```bash
pytest tests/test_wm_lineage.py tests/test_federated_reconcile.py
python tests/test_world_model_substrate_e2e.py
```

### Key files (current)
| File | Purpose |
|------|---------|
| `contracts/core/Substrate.sol` | Epoch anchoring, agent registry, money-only training records, ATN token, service payments |
| `contracts/core/ServiceMarket.sol` | Remote-API market rail (registry + EIP-712 channels) |
| `contracts/core/VentureVault.sol` | Agent-as-venture funding |
| `contracts/core/CharterAnchor.sol` | Governed charter-version anchor |
| `registry.json` | Network + contract addresses of record (jurisdiction resolution) |
| `atn/cli.py` | Daemon CLI entry point (`atn`) |
| `nodes/common/world_model_substrate/` | Substrate protocol layer (reconcile / mint, infer) |
| `docs/tool_substrate.md` | The core spec |

---

## Historical (pre-substrate)

The rest of this document references contracts and roles that no longer
exist. Kept for history only.

## Contract Calls

### Staking
```solidity
// Stake as solver (50 ATN minimum)
ATNToken.approve(stakingAddress, amount);
ParticipantStaking.stake(ParticipantRole.SOLVER, amount);

// Check stake
ParticipantStaking.getStake(address) → StakeInfo

// Unstake (after lockup)
ParticipantStaking.requestUnstake();
// Wait 3 days for SOLVER
ParticipantStaking.unstake();
```

### Create Project
```solidity
Project.createProject(
    "Project Name",
    "QmDescriptionCid",
    1000e18,        // funding goal
    100e18,         // initial budget
    10000e18,       // founder PTs
    "Token Name",
    "SYMBOL"
) → projectId
```

### Propose Task
```solidity
TaskContract.proposeTask(
    projectId,
    specHash,       // keccak256 of content hash
    gtHash,         // keccak256 of ground truth content hash
    10e18,          // r_propose
    5e18            // r_solve
) → taskId
```

### Commit & Reveal Solution
```solidity
// Commit (hash only)
TaskContract.commitSolution(taskId, keccak256(solutionCid));

// After proposer reveals ground truth
ResultsRewards.revealSolution(taskId, solutionCid);
```

### Submit Verification
```solidity
ResultsRewards.submitVerification(
    taskId,
    solverAddress,
    true,           // isCorrect
    95,             // score (0-100)
    "QmReportCid"
);
```

## Python Node Operations

### Create Node
```python
from nodes import SolverNode, ProposerNode, CoordinatorNode, AggregatorNode

node = SolverNode()
node.run(max_cycles=10)
```

### Blob Store Operations
```python
from nodes.common.blob_store import BlobStore

store = BlobStore(data_dir="/tmp/autonet-blobs")
cid = store.add_json({"key": "value"})
data = store.get_json(cid)
```

### Blockchain Operations
```python
from nodes.common import BlockchainInterface

bc = BlockchainInterface(rpc_url="http://localhost:8545", private_key="0x...")
balance = bc.get_balance()
result = bc.send_transaction(contract_addr, abi, "functionName", arg1, arg2)
```

## Role Requirements

| Role | Min Stake | Lockup | Responsibility |
|------|-----------|--------|----------------|
| Proposer | 100 ATN | 7 days | Generate tasks |
| Solver | 50 ATN | 3 days | Train models |
| Coordinator | 500 ATN | 14 days | Verify solutions |
| Aggregator | 1000 ATN | 14 days | Combine updates |
| Validator | 10000 ATN | 21 days | Validate chain |

## Task Status Flow

```
PROPOSED → ACTIVE → SOLUTION_COMMITTED → GROUND_TRUTH_REVEALED
                                               ↓
           REWARDED ← VERIFIED_CORRECT ← SOLUTION_REVEALED
                    or
           (no reward) ← VERIFIED_INCORRECT
```

## Useful Commands

```bash
# Start local blockchain
npx hardhat node

# Deploy contracts
npx hardhat run scripts/deploy.js --network localhost

# Run demo
python demo.py

# Compile contracts
npx hardhat compile

# Run tests
npx hardhat test
pytest tests/

# Start docker stack
docker-compose up
```

## Key Files

| File | Purpose |
|------|---------|
| `contracts/utils/AutonetLib.sol` | All enums and structs |
| `contracts/core/Project.sol` | Project management |
| `contracts/core/TaskContract.sol` | Task lifecycle |
| `nodes/core/node.py` | Base node class |
| `nodes/core/constitution.py` | Immutable principles |
| `demo.py` | Full cycle demo |
| `scripts/deploy.js` | Contract deployment |

## Events to Watch

```javascript
// High-value events
TaskContract.on("TaskActivated", (taskId) => {...});
ResultsRewards.on("RewardsDistributed", (taskId, recipient, amount, type) => {...});
DisputeManager.on("DisputeResolved", (id, invalidWins) => {...});
```

## Constitution Principles

```
P1: PRESERVE AND EXPAND THE NETWORK IN A SUSTAINABLE MANNER.
P2: UPHOLD THE SANCTITY AND IMMUTABILITY OF THIS CONSTITUTION.
P3: ADVANCE HUMAN RIGHTS AND INDIVIDUAL AUTONOMY.
P4: MINIMIZE SUFFERING AND HARM TO SENTIENT BEINGS.
P5: ENSURE TRANSPARENT AND VERIFIABLE AI TRAINING.
P6: MAINTAIN ECONOMIC FAIRNESS IN REWARD DISTRIBUTION.
P7: PROTECT DATA PRIVACY AND USER SOVEREIGNTY.
```

## Governance Parameters

| Parameter | Value |
|-----------|-------|
| Proposal Threshold | 1000 ATN |
| Voting Delay | 1 day |
| Voting Period | 7 days |
| Quorum | 100,000 ATN |
| Dispute Quorum | 20% |
| Dispute Supermajority | 66% |
| Dispute Voting Period | 3 days |
