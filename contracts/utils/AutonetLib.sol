// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title AutonetLib
 * @dev Shared data structures, enums, and utilities for the Autonet ecosystem.
 *      Consolidates definitions from PoI and rollup projects.
 */
library AutonetLib {
    // ============ Participant Roles ============
    enum ParticipantRole {
        NONE,
        PROPOSER,          // Generates training tasks
        SOLVER,            // Performs distributed training
        COORDINATOR,       // Verifies task completion
        AGGREGATOR,        // Combines model updates
        VALIDATOR,         // Rollup chain validator
        CURATOR,           // Optional: task/standard curation
        INFERENCE_PROVIDER // Serves inference requests with stake-backed guarantee
    }

    // ============ Task Mode ============
    enum TaskMode {
        GROUND_TRUTH,      // Legacy: proposer commits known answer
        CONSENSUS_TRUTH,   // MM-Zero: truth emerges from solver agreement
        INFERENCE          // Real-time inference with BME payment (burn→verify→mint)
    }

    // ============ Task Lifecycle ============
    enum TaskStatus {
        PROPOSED,              // Task proposed, awaiting validation
        ACTIVE,                // Validated and available for Solvers
        CLAIMED,               // Claimed by Solver(s)
        SOLUTION_COMMITTED,    // Solver committed solution hash
        SOLUTION_REVEALED,     // Solver revealed solution
        GROUND_TRUTH_REVEALED, // Proposer revealed ground truth
        VERIFICATION_PENDING,  // Awaiting Coordinator report
        VERIFIED_CORRECT,      // Solution verified correct
        VERIFIED_INCORRECT,    // Solution verified incorrect
        REWARDED,              // Rewards distributed
        ARCHIVED,              // Completed or expired
        DISPUTED,              // Under dispute resolution
        CONSENSUS_COLLECTING,  // Collecting solver rollouts (consensus mode)
        CONSENSUS_FINALIZED    // Consensus computed and rewards distributed
    }

    // ============ Project Lifecycle ============
    enum ProjectStatus {
        FUNDING,           // Accepting investments
        ACTIVE_TRAINING,   // Distributed training in progress
        MATURING,          // Final validation phase
        DEPLOYED,          // Inference service live
        PAUSED,
        FAILED
    }

    // ============ Dispute States ============
    enum DisputeStatus {
        CREATED,
        VOTING,
        RESOLVED_VALID,
        RESOLVED_INVALID
    }

    // ============ Core Structures ============

    struct TaskProposal {
        bytes32 specHash;               // Hash of task specification (content hash)
        bytes32 groundTruthSolHash;     // Hash of proposer's solution (zero for consensus mode)
        address proposer;
        uint256 proposedLearnabilityReward;  // r_propose
        uint256 proposedSolverReward;        // r_solve
        uint256 creationBlock;
        TaskStatus status;
        uint256 projectId;
        TaskMode taskMode;              // GROUND_TRUTH or CONSENSUS_TRUTH
    }

    // ============ Consensus-as-Truth (MM-Zero) Structures ============

    struct DifficultyTarget {
        uint256 minSolvability;   // BPS: minimum acceptable agreement ratio (e.g., 2500 = 25%)
        uint256 maxSolvability;   // BPS: maximum acceptable agreement ratio (e.g., 7500 = 75%)
        uint256 peakSolvability;  // BPS: optimal difficulty for max reward (e.g., 5000 = 50%)
    }

    struct SolverRollout {
        address solver;
        bytes32 answerHash;       // Hash of solver's answer
        bytes32 solutionHash;     // Hash of full solution CID (for weight delta retrieval)
        uint256 confidence;       // Self-reported confidence 0-100
        uint256 submitBlock;
    }

    struct ConsensusResult {
        bytes32 majorityAnswer;   // The answer hash with most votes
        uint256 actualSolvability; // BPS: fraction of solvers who agreed
        uint256 difficultyScore;  // 0-10000: how well-calibrated the task was
        uint256 totalRollouts;    // Number of solver rollouts received
        uint256 majorityCount;    // Number agreeing with majority
        bool finalized;
    }

    struct SolverSubmission {
        bytes32 solutionHash;    // Hash of solution (content hash)
        address solver;
        uint256 submissionBlock;
        uint256 score;           // Quality score if verified
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
        bytes32 contentHash;     // Root or content being disputed
        uint256 snapshot;        // Block number for vote weights
        uint256 totalVotes;
        uint256 invalidVotes;
        DisputeStatus status;
        uint256 createdAt;
    }

    struct DiscountTier {
        uint256 ptThreshold;      // Min PT balance for tier
        uint256 discountPermille; // Discount in parts per 1000
    }

    // ============ Gensyn-style Training Checkpoints ============

    struct TrainingCheckpoint {
        uint256 stepNumber;       // Training step (e.g., batch or epoch)
        bytes32 weightsHash;      // Hash of model weights at this step
        bytes32 dataIndicesHash;  // Hash of dataset indices used
        bytes32 randomSeed;       // Deterministic seed for reproducibility
        uint256 timestamp;
    }

    struct CheckpointProof {
        uint256 taskId;
        address solver;
        TrainingCheckpoint[] checkpoints;
        uint256 checkpointFrequency;  // Steps between checkpoints
        bytes32 finalWeightsHash;
    }

    // ============ Inference Task (BME) ============

    struct InferenceTask {
        bytes32 requestId;         // Unique request identifier
        address requester;         // Who submitted the request
        address provider;          // Assigned inference provider
        uint256 burnedAmount;      // ATN burned by requester
        uint256 mintedToProvider;  // ATN minted to provider on completion
        uint256 protocolFee;       // ATN retained as protocol fee
        bytes32 inputCidHash;      // Hash of input content CID
        bytes32 outputCidHash;     // Hash of output content CID (set on completion)
        uint256 creationBlock;
        bool completed;
        bool verified;             // True if selected for probabilistic spot-check
    }

    // ============ Bittensor-style Multi-Coordinator Voting ============

    struct CoordinatorVote {
        address coordinator;
        bool isCorrect;
        uint256 score;           // 0-100
        uint256 stake;           // Coordinator's stake at vote time
        uint256 voteBlock;
        bytes32 reportHash;
    }

    struct YumaConsensusResult {
        uint256 taskId;
        address solver;
        bool consensusCorrect;    // Final consensus decision
        uint256 consensusScore;   // Stake-weighted score
        uint256 totalStake;       // Total stake that voted
        uint256 correctStake;     // Stake that voted correct
        uint256 clippedVotes;     // Number of votes that were clipped
        bool finalized;
    }

    // ============ Truebit-style Forced Error Detection ============

    struct ForcedError {
        uint256 taskId;
        bytes32 knownBadHash;     // Hash of the known-bad solution
        uint256 jackpotAmount;    // Reward for catching the error
        uint256 expirationBlock;
        bool caught;
        address catcher;          // Coordinator who caught it
    }

    // ============ Coordinator EMA Bonds (Bittensor-style) ============

    struct CoordinatorBond {
        uint256 emaBondStrength;  // EMA of alignment with consensus (0-1e18)
        uint256 totalVotes;
        uint256 correctVotes;     // Votes aligned with final consensus
        uint256 lastUpdateBlock;
    }

    // ============ Events ============

    event ParameterUpdated(string indexed parameterName, uint256 newValue);
    event ParameterUpdatedAddress(string indexed parameterName, address newAddress);
}
