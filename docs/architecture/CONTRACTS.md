# Smart Contract Reference

> **HISTORICAL, pre-substrate (banner added 2026-07-10).** Every contract
> documented below (ATNToken, ParticipantStaking, TaskContract,
> ResultsRewards, Project, DisputeManager, AnchorBridge, AutonetDAO) was
> **deleted wholesale in Phase 5.6a**. None of these exist in the repo today.
> The live on-chain surface is four contracts under `contracts/core/`:
> - **`Substrate.sol`**: constitutional core, covering epoch anchoring
>   (`submitAnchor`), agent registry (`registerAgent`/`updatePeerId`/
>   `updateEndpoint`), money-only training records
>   (`recordTrainingForEpoch(amount, epochIdHash, proof)`, 2-field merkle
>   leaf), ATN token (transferable, only mint path, no admin keys), and
>   service payments (`payForService`). No reputation surface: REP is
>   claimed DAO-side (RepToken, 1:1 on ATN earnings).
> - **`ServiceMarket.sol`**: remote-API market (registry + EIP-712 payment
>   channels), ATN-only, settling through `Substrate.payForService`.
> - **`VentureVault.sol`**: agent-as-venture funding.
> - **`CharterAnchor.sol`**: governed anchor for the charter version.
> - **Read instead:** `docs/README.md`, `docs/tool_substrate.md`,
>   `docs/services_market.md`, `docs/charter_anchor.md`, and the contract
>   NatSpec. The body below is kept only for history.

## Contract Dependency Graph

```
ATNToken ◄─────────────────────────────────────────────┐
    │                                                   │
    ▼                                                   │
ParticipantStaking ◄───────────────────────┐           │
    │                                       │           │
    ▼                                       │           │
TaskContract ───────────────────────────────┼───────────┤
    │                                       │           │
    ▼                                       │           │
ResultsRewards ─────────────────────────────┤           │
    │                                       │           │
    ▼                                       │           │
Project ────────────────────────────────────┼───────────┤
                                            │           │
AutonetDAO ◄────────────────────────────────┴───────────┘
```

## AutonetLib.sol

Shared library with all enums and structs.

### Enums

```solidity
enum ParticipantRole {
    NONE,
    PROPOSER,      // 100 ATN stake, 7 day lockup
    SOLVER,        // 50 ATN stake, 3 day lockup
    COORDINATOR,   // 500 ATN stake, 14 day lockup
    AGGREGATOR,    // 1000 ATN stake, 14 day lockup
    VALIDATOR,     // 10000 ATN stake, 21 day lockup
    CURATOR
}

enum TaskStatus {
    PROPOSED,              // Just created
    ACTIVE,                // Ready for solvers
    CLAIMED,               // Solver working
    SOLUTION_COMMITTED,    // Hash submitted
    SOLUTION_REVEALED,     // CID revealed
    GROUND_TRUTH_REVEALED, // Proposer revealed
    VERIFICATION_PENDING,  // Awaiting coordinator
    VERIFIED_CORRECT,      // Passed
    VERIFIED_INCORRECT,    // Failed
    REWARDED,              // Payments done
    ARCHIVED,              // Closed
    DISPUTED               // Under review
}

enum ProjectStatus {
    FUNDING,         // Accepting ATN investments
    ACTIVE_TRAINING, // Training in progress
    MATURING,        // Final validation
    DEPLOYED,        // Inference live
    PAUSED,
    FAILED
}

enum DisputeStatus {
    CREATED,
    VOTING,
    RESOLVED_VALID,
    RESOLVED_INVALID
}
```

### Structs

```solidity
struct TaskProposal {
    bytes32 specHash;
    bytes32 groundTruthSolHash;
    address proposer;
    uint256 proposedLearnabilityReward;
    uint256 proposedSolverReward;
    uint256 creationBlock;
    TaskStatus status;
    uint256 projectId;
}

struct SolverSubmission {
    bytes32 solutionHash;
    address solver;
    uint256 submissionBlock;
    uint256 score;
}

struct VerificationReport {
    address coordinator;
    bool isCorrect;
    uint256 score;
    bytes32 supportingDataHash;
    uint256 verificationBlock;
}

struct StakeInfo {
    uint256 amount;
    ParticipantRole role;
    uint256 lockupUntil;
    bool active;
}

struct Dispute {
    bytes32 contentHash;
    uint256 snapshot;
    uint256 totalVotes;
    uint256 invalidVotes;
    DisputeStatus status;
    uint256 createdAt;
}

struct DiscountTier {
    uint256 ptThreshold;
    uint256 discountPermille;  // per 1000
}
```

## Contract Details

### ATNToken.sol

ERC20Votes token with DAO-controlled minting.

```solidity
// Key functions
mint(address to, uint256 amount)           // DAO only
setDaoAddress(address newDao)              // Current DAO only
burn(uint256 amount)                       // Any holder
delegate(address delegatee)                // For voting
```

### ParticipantStaking.sol

Manages stake deposits, lockups, slashing.

```solidity
// State
mapping(address => StakeInfo) stakes
mapping(ParticipantRole => uint256) minStakeAmount
mapping(ParticipantRole => uint256) stakeLockupDuration
mapping(address => bool) authorizedSlashers

// Key functions
stake(ParticipantRole role, uint256 amount)
addToStake(uint256 amount)
requestUnstake()                           // Starts lockup timer
unstake()                                  // After lockup
slash(address participant, uint256 amount, string reason)  // Authorized only
isActiveParticipant(address, ParticipantRole) → bool
```

### Project.sol

AI project lifecycle management.

```solidity
// Key functions
createProject(name, descriptionCid, fundingGoal, initialBudget,
              founderPTs, ptName, ptSymbol) → projectId
fundProject(projectId, atnAmount, expectedPTs)
allocateTaskBudget(projectId, amount)
setMatureModel(projectId, weightsCid, price)
setDiscountTiers(projectId, tiers[])
getEffectivePrice(projectId, user) → price
requestInference(projectId, inputCid) → requestId
disburseFromBudget(projectId, recipient, amount) → bool
withdrawRevenue(projectId)
```

### TaskContract.sol

Training task lifecycle.

```solidity
// Key functions
proposeTask(projectId, specHash, groundTruthHash,
            learnabilityReward, solverReward) → taskId
activateTask(taskId)                       // Governance only
commitSolution(taskId, solutionHash)       // Staked solver only
updateTaskStatus(taskId, newStatus)        // ResultsRewards or gov
getTaskProposal(taskId) → TaskProposal
getSubmission(taskId, solver) → SolverSubmission
getCommittedSolvers(taskId) → address[]
```

### ResultsRewards.sol

Verification and reward distribution.

```solidity
// Key functions
revealGroundTruth(taskId, cid)             // Proposer only
revealSolution(taskId, cid)                // Solver only
submitVerification(taskId, solver, isCorrect, score, reportCid)
// Auto-triggers _processRewards() which:
//   - Pays coordinator fee (1 ATN)
//   - Pays solver reward if correct
//   - Pays proposer reward if solver correct
```

### DisputeManager.sol

Stake-weighted dispute resolution.

```solidity
// Config
quorumNumerator = 2000      // 20%
supermajorityNum = 6600     // 66%
votingPeriod = 3 days

// Key functions
createDispute(contentHash) → disputeId
vote(disputeId, supportInvalid)
finalizeDispute(disputeId)
getDispute(disputeId) → Dispute
```

### AnchorBridge.sol

L1/L2 bridge for checkpoints and tokens.

```solidity
// Key functions
addValidator(address)
removeValidator(address)
submitCheckpoint(epoch, root)              // Validator only
deposit(amount)                            // Lock tokens for L2
withdraw(amount, withdrawalId, proof[])    // With Merkle proof
```

### AutonetDAO.sol

Governance with proposal/vote/execute cycle.

```solidity
// Key functions
propose(description, calldataHash) → proposalId
castVote(proposalId, support)
execute(proposalId, target, data)
cancel(proposalId)
getProposalState(proposalId) → ProposalState
```

## Events to Monitor

```solidity
// Project
ProjectCreated(projectId, founder, name, projectToken)
ProjectFunded(projectId, funder, atnAmount, ptsIssued)
MatureModelUpdated(projectId, weightsCid)
InferenceRequested(projectId, user, requestId, inputCid, fee)

// Tasks
TaskProposed(taskId, projectId, proposer, specHash, groundTruthHash)
TaskActivated(taskId)
SolutionCommitted(taskId, solver, solutionHash)

// Verification
GroundTruthRevealed(taskId, proposer, cid)
SolutionRevealed(taskId, solver, cid)
VerificationSubmitted(taskId, coordinator, correct, score)
RewardsDistributed(taskId, recipient, amount, rewardType)

// Staking
Staked(participant, role, amount)
Unstaked(participant, role, amount)
Slashed(participant, role, amount, reason)

// Disputes
DisputeCreated(id, contentHash, challenger)
Voted(id, voter, supportInvalid, weight)
DisputeResolved(id, invalidWins)
```
