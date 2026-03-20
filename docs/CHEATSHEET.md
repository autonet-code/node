# Autonet Cheat Sheet

Quick reference for common operations.

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
