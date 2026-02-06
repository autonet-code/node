// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../utils/AutonetLib.sol";
import "./TaskContract.sol";
import "./ParticipantStaking.sol";
import "./Project.sol";

/**
 * @title ResultsRewards
 * @dev Handles solution/ground-truth reveals, multi-coordinator Yuma voting,
 *      EMA bond tracking, and reward distribution.
 *
 * Implements Bittensor-style Yuma Consensus where:
 * - Multiple coordinators vote on solution correctness
 * - Votes are weighted by stake
 * - Outlier votes are clipped to consensus
 * - EMA bonds reward consistent alignment with consensus
 */
contract ResultsRewards {
    TaskContract public immutable taskContract;
    ParticipantStaking public immutable staking;
    Project public projectContract;
    address public governance;

    // ============ Configuration ============
    uint256 public constant VOTING_PERIOD_BLOCKS = 100;  // ~20 minutes
    uint256 public constant MIN_COORDINATORS = 2;        // Minimum votes for consensus
    uint256 public constant CLIP_THRESHOLD_BPS = 2000;   // 20% deviation triggers clipping
    uint256 public constant EMA_DECAY = 900;             // 0.9 in basis points (900/1000)
    uint256 public constant BOND_MULTIPLIER_MAX = 1500;  // 1.5x max bonus for strong bonds

    // ============ Storage ============

    // TaskID => Proposer => Ground Truth CID
    mapping(uint256 => mapping(address => string)) public revealedGroundTruths;
    // TaskID => Solver => Solution CID
    mapping(uint256 => mapping(address => string)) public revealedSolutions;

    // TaskID => Solver => Coordinator votes array
    mapping(uint256 => mapping(address => AutonetLib.CoordinatorVote[])) public votes;
    // TaskID => Solver => Yuma consensus result
    mapping(uint256 => mapping(address => AutonetLib.YumaConsensusResult)) public consensusResults;
    // TaskID => Solver => voting start block
    mapping(uint256 => mapping(address => uint256)) public votingStartBlock;

    // Coordinator => EMA bond data
    mapping(address => AutonetLib.CoordinatorBond) public coordinatorBonds;

    // TaskID => rewards processed
    mapping(uint256 => bool) public rewardsProcessed;

    // Legacy compatibility: TaskID => Coordinator => Report (for single-coordinator mode)
    mapping(uint256 => mapping(address => AutonetLib.VerificationReport)) public verificationReports;

    // ============ Events ============

    event GroundTruthRevealed(uint256 indexed taskId, address indexed proposer, string cid);
    event SolutionRevealed(uint256 indexed taskId, address indexed solver, string cid);
    event CoordinatorVoted(uint256 indexed taskId, address indexed solver, address indexed coordinator, bool isCorrect, uint256 score);
    event VoteClipped(uint256 indexed taskId, address indexed coordinator, uint256 originalScore, uint256 clippedScore);
    event YumaConsensusReached(uint256 indexed taskId, address indexed solver, bool consensusCorrect, uint256 consensusScore);
    event BondUpdated(address indexed coordinator, uint256 newBondStrength, bool alignedWithConsensus);
    event VerificationSubmitted(uint256 indexed taskId, address indexed coordinator, bool correct, uint256 score);
    event RewardsDistributed(uint256 indexed taskId, address indexed recipient, uint256 amount, string rewardType);

    // ============ Modifiers ============

    modifier onlyStakedCoordinator() {
        require(staking.isActiveParticipant(msg.sender, AutonetLib.ParticipantRole.COORDINATOR), "Not active Coordinator");
        _;
    }

    modifier onlyGovernance() {
        require(msg.sender == governance, "Not governance");
        _;
    }

    // ============ Constructor ============

    constructor(address _taskContract, address _staking, address _governance) {
        taskContract = TaskContract(_taskContract);
        staking = ParticipantStaking(_staking);
        governance = _governance;
    }

    function setProjectContract(address _project) external onlyGovernance {
        projectContract = Project(_project);
    }

    // ============ Reveal Functions ============

    function revealGroundTruth(uint256 _taskId, string calldata _cid) external {
        AutonetLib.TaskProposal memory proposal = taskContract.getTaskProposal(_taskId);
        require(msg.sender == proposal.proposer, "Not proposer");
        require(keccak256(abi.encodePacked(_cid)) == proposal.groundTruthSolHash, "CID mismatch");

        revealedGroundTruths[_taskId][msg.sender] = _cid;
        taskContract.updateTaskStatus(_taskId, AutonetLib.TaskStatus.GROUND_TRUTH_REVEALED);
        emit GroundTruthRevealed(_taskId, msg.sender, _cid);
    }

    function revealSolution(uint256 _taskId, string calldata _cid) external {
        AutonetLib.SolverSubmission memory sub = taskContract.getSubmission(_taskId, msg.sender);
        require(sub.solver == msg.sender, "No submission");
        require(keccak256(abi.encodePacked(_cid)) == sub.solutionHash, "CID mismatch");

        revealedSolutions[_taskId][msg.sender] = _cid;
        taskContract.updateTaskStatus(_taskId, AutonetLib.TaskStatus.SOLUTION_REVEALED);

        // Start voting period
        votingStartBlock[_taskId][msg.sender] = block.number;

        emit SolutionRevealed(_taskId, msg.sender, _cid);
    }

    // ============ Multi-Coordinator Voting (Yuma Consensus) ============

    /**
     * @dev Submit a verification vote. Multiple coordinators can vote.
     * @param _taskId The task being verified
     * @param _solver The solver whose solution is being verified
     * @param _isCorrect Whether the coordinator believes the solution is correct
     * @param _score Quality score 0-100
     * @param _reportCid IPFS CID of the detailed report
     */
    function submitVote(
        uint256 _taskId,
        address _solver,
        bool _isCorrect,
        uint256 _score,
        string calldata _reportCid
    ) external onlyStakedCoordinator {
        AutonetLib.TaskProposal memory proposal = taskContract.getTaskProposal(_taskId);
        require(
            proposal.status == AutonetLib.TaskStatus.SOLUTION_REVEALED ||
            proposal.status == AutonetLib.TaskStatus.GROUND_TRUTH_REVEALED,
            "Not ready for verification"
        );
        require(bytes(revealedSolutions[_taskId][_solver]).length > 0, "Solution not revealed");

        // Check voting period
        uint256 startBlock = votingStartBlock[_taskId][_solver];
        require(startBlock > 0, "Voting not started");
        require(block.number <= startBlock + VOTING_PERIOD_BLOCKS, "Voting period ended");

        // Check coordinator hasn't already voted
        AutonetLib.CoordinatorVote[] storage taskVotes = votes[_taskId][_solver];
        for (uint256 i = 0; i < taskVotes.length; i++) {
            require(taskVotes[i].coordinator != msg.sender, "Already voted");
        }

        // Get coordinator's stake
        AutonetLib.StakeInfo memory stakeInfo = staking.getStake(msg.sender);
        require(stakeInfo.active, "Stake not active");

        // Record vote
        taskVotes.push(AutonetLib.CoordinatorVote({
            coordinator: msg.sender,
            isCorrect: _isCorrect,
            score: _score,
            stake: stakeInfo.amount,
            voteBlock: block.number,
            reportHash: keccak256(abi.encodePacked(_reportCid))
        }));

        emit CoordinatorVoted(_taskId, _solver, msg.sender, _isCorrect, _score);
    }

    /**
     * @dev Finalize voting and compute Yuma consensus.
     * Can be called after voting period ends or when enough coordinators have voted.
     */
    function finalizeVoting(uint256 _taskId, address _solver) external {
        AutonetLib.YumaConsensusResult storage result = consensusResults[_taskId][_solver];
        require(!result.finalized, "Already finalized");

        uint256 startBlock = votingStartBlock[_taskId][_solver];
        AutonetLib.CoordinatorVote[] storage taskVotes = votes[_taskId][_solver];

        // Either voting period ended or minimum coordinators voted
        require(
            block.number > startBlock + VOTING_PERIOD_BLOCKS ||
            taskVotes.length >= MIN_COORDINATORS,
            "Cannot finalize yet"
        );
        require(taskVotes.length >= MIN_COORDINATORS, "Insufficient coordinator votes");

        // Compute Yuma consensus
        (bool consensusCorrect, uint256 consensusScore, uint256 totalStake, uint256 correctStake, uint256 clippedCount) =
            _computeYumaConsensus(_taskId, taskVotes);

        // Store result
        result.taskId = _taskId;
        result.solver = _solver;
        result.consensusCorrect = consensusCorrect;
        result.consensusScore = consensusScore;
        result.totalStake = totalStake;
        result.correctStake = correctStake;
        result.clippedVotes = clippedCount;
        result.finalized = true;

        // Update coordinator bonds
        _updateCoordinatorBonds(taskVotes, consensusCorrect);

        // Update task status
        taskContract.updateTaskStatus(
            _taskId,
            consensusCorrect ? AutonetLib.TaskStatus.VERIFIED_CORRECT : AutonetLib.TaskStatus.VERIFIED_INCORRECT
        );

        emit YumaConsensusReached(_taskId, _solver, consensusCorrect, consensusScore);

        // Process rewards
        _processRewardsYuma(_taskId, _solver, taskVotes, consensusCorrect, consensusScore);
    }

    /**
     * @dev Compute Yuma consensus with stake-weighted voting and clipping.
     */
    function _computeYumaConsensus(uint256 _taskId, AutonetLib.CoordinatorVote[] storage taskVotes)
        internal
        returns (bool consensusCorrect, uint256 consensusScore, uint256 totalStake, uint256 correctStake, uint256 clippedCount)
    {
        uint256 numVotes = taskVotes.length;

        // Calculate total stake and stake-weighted score
        uint256 weightedScoreSum = 0;
        for (uint256 i = 0; i < numVotes; i++) {
            totalStake += taskVotes[i].stake;
            if (taskVotes[i].isCorrect) {
                correctStake += taskVotes[i].stake;
            }
            weightedScoreSum += taskVotes[i].score * taskVotes[i].stake;
        }

        // Consensus correct if majority stake agrees
        consensusCorrect = correctStake > totalStake / 2;

        // Calculate stake-weighted average score (before clipping)
        uint256 avgScore = totalStake > 0 ? weightedScoreSum / totalStake : 0;

        // Apply clipping: scores that deviate too much from average are clipped
        uint256 clippedWeightedSum = 0;
        uint256 clippedTotalStake = 0;

        for (uint256 i = 0; i < numVotes; i++) {
            uint256 score = taskVotes[i].score;
            uint256 stake = taskVotes[i].stake;

            // Calculate deviation from average
            uint256 deviation = score > avgScore ? score - avgScore : avgScore - score;
            uint256 deviationBps = avgScore > 0 ? (deviation * 10000) / avgScore : 0;

            if (deviationBps > CLIP_THRESHOLD_BPS) {
                // Clip to threshold
                uint256 clippedScore;
                if (score > avgScore) {
                    clippedScore = avgScore + (avgScore * CLIP_THRESHOLD_BPS) / 10000;
                } else {
                    clippedScore = avgScore > (avgScore * CLIP_THRESHOLD_BPS) / 10000 ?
                        avgScore - (avgScore * CLIP_THRESHOLD_BPS) / 10000 : 0;
                }
                emit VoteClipped(_taskId, taskVotes[i].coordinator, score, clippedScore);
                clippedWeightedSum += clippedScore * stake;
                clippedCount++;
            } else {
                clippedWeightedSum += score * stake;
            }
            clippedTotalStake += stake;
        }

        consensusScore = clippedTotalStake > 0 ? clippedWeightedSum / clippedTotalStake : 0;
    }

    /**
     * @dev Update coordinator EMA bonds based on alignment with consensus.
     */
    function _updateCoordinatorBonds(AutonetLib.CoordinatorVote[] storage taskVotes, bool consensusCorrect) internal {
        for (uint256 i = 0; i < taskVotes.length; i++) {
            address coordinator = taskVotes[i].coordinator;
            bool alignedWithConsensus = taskVotes[i].isCorrect == consensusCorrect;

            AutonetLib.CoordinatorBond storage bond = coordinatorBonds[coordinator];

            // Update EMA: new = decay * old + (1-decay) * current
            // Using basis points: 900/1000 = 0.9 decay
            uint256 currentValue = alignedWithConsensus ? 1e18 : 0;
            bond.emaBondStrength = (EMA_DECAY * bond.emaBondStrength + (1000 - EMA_DECAY) * currentValue) / 1000;

            bond.totalVotes++;
            if (alignedWithConsensus) {
                bond.correctVotes++;
            }
            bond.lastUpdateBlock = block.number;

            emit BondUpdated(coordinator, bond.emaBondStrength, alignedWithConsensus);
        }
    }

    /**
     * @dev Process rewards with bond multipliers for coordinators.
     */
    function _processRewardsYuma(
        uint256 _taskId,
        address _solver,
        AutonetLib.CoordinatorVote[] storage taskVotes,
        bool _isCorrect,
        uint256 _consensusScore
    ) internal {
        require(!rewardsProcessed[_taskId], "Already processed");

        AutonetLib.TaskProposal memory proposal = taskContract.getTaskProposal(_taskId);
        uint256 projectId = proposal.projectId;

        // Distribute coordinator fees with bond multiplier
        uint256 baseCoordFee = 1 * 1e18; // 1 ATN base fee
        for (uint256 i = 0; i < taskVotes.length; i++) {
            address coordinator = taskVotes[i].coordinator;
            bool alignedWithConsensus = taskVotes[i].isCorrect == _isCorrect;

            // Only pay coordinators who aligned with consensus
            if (alignedWithConsensus) {
                // Apply bond multiplier (up to 1.5x for strong bonds)
                uint256 bondStrength = coordinatorBonds[coordinator].emaBondStrength;
                uint256 multiplier = 1000 + (bondStrength * (BOND_MULTIPLIER_MAX - 1000)) / 1e18;
                uint256 adjustedFee = (baseCoordFee * multiplier) / 1000;

                try projectContract.disburseFromBudget(projectId, coordinator, adjustedFee) returns (bool success) {
                    if (success) {
                        emit RewardsDistributed(_taskId, coordinator, adjustedFee, "CoordinatorFee");
                    }
                } catch {
                    // Continue if disbursement fails (e.g., insufficient budget)
                }
            } else {
                // Slash coordinators who voted against consensus
                uint256 slashAmount = taskVotes[i].stake / 10; // 10% of stake
                staking.slash(coordinator, slashAmount, "Voted against consensus");
            }
        }

        if (_isCorrect) {
            // Solver reward scaled by consensus score
            uint256 scaledSolverReward = (proposal.proposedSolverReward * _consensusScore) / 100;
            try projectContract.disburseFromBudget(projectId, _solver, scaledSolverReward) returns (bool success) {
                if (success) {
                    emit RewardsDistributed(_taskId, _solver, scaledSolverReward, "SolverReward");
                }
            } catch {
                // Continue if disbursement fails
            }

            // Proposer reward
            try projectContract.disburseFromBudget(projectId, proposal.proposer, proposal.proposedLearnabilityReward) returns (bool success) {
                if (success) {
                    emit RewardsDistributed(_taskId, proposal.proposer, proposal.proposedLearnabilityReward, "ProposerReward");
                }
            } catch {
                // Continue if disbursement fails
            }
        }

        rewardsProcessed[_taskId] = true;
        taskContract.updateTaskStatus(_taskId, AutonetLib.TaskStatus.REWARDED);
    }

    // ============ Legacy Single-Coordinator Mode (for backwards compatibility) ============

    /**
     * @dev Submit verification in single-coordinator mode (legacy).
     * This auto-finalizes without waiting for other coordinators.
     */
    function submitVerification(
        uint256 _taskId,
        address _solver,
        bool _isCorrect,
        uint256 _score,
        string calldata _reportCid
    ) external onlyStakedCoordinator {
        AutonetLib.TaskProposal memory proposal = taskContract.getTaskProposal(_taskId);
        require(
            proposal.status == AutonetLib.TaskStatus.SOLUTION_REVEALED ||
            proposal.status == AutonetLib.TaskStatus.GROUND_TRUTH_REVEALED,
            "Not ready for verification"
        );
        require(bytes(revealedSolutions[_taskId][_solver]).length > 0, "Solution not revealed");

        verificationReports[_taskId][msg.sender] = AutonetLib.VerificationReport({
            coordinator: msg.sender,
            isCorrect: _isCorrect,
            score: _score,
            supportingDataHash: keccak256(abi.encodePacked(_reportCid)),
            verificationBlock: block.number
        });

        taskContract.updateTaskStatus(
            _taskId,
            _isCorrect ? AutonetLib.TaskStatus.VERIFIED_CORRECT : AutonetLib.TaskStatus.VERIFIED_INCORRECT
        );

        emit VerificationSubmitted(_taskId, msg.sender, _isCorrect, _score);

        // Auto-process rewards (simplified single-coordinator mode)
        _processRewards(_taskId, _solver, msg.sender, _isCorrect);
    }

    function _processRewards(
        uint256 _taskId,
        address _solver,
        address _coordinator,
        bool _isCorrect
    ) internal {
        require(!rewardsProcessed[_taskId], "Already processed");

        AutonetLib.TaskProposal memory proposal = taskContract.getTaskProposal(_taskId);
        uint256 projectId = proposal.projectId;

        // Coordinator fee (1 ATN)
        uint256 coordFee = 1 * 1e18;
        if (projectContract.disburseFromBudget(projectId, _coordinator, coordFee)) {
            emit RewardsDistributed(_taskId, _coordinator, coordFee, "CoordinatorFee");
        }

        if (_isCorrect) {
            // Solver reward
            if (projectContract.disburseFromBudget(projectId, _solver, proposal.proposedSolverReward)) {
                emit RewardsDistributed(_taskId, _solver, proposal.proposedSolverReward, "SolverReward");
            }
            // Proposer reward
            if (projectContract.disburseFromBudget(projectId, proposal.proposer, proposal.proposedLearnabilityReward)) {
                emit RewardsDistributed(_taskId, proposal.proposer, proposal.proposedLearnabilityReward, "ProposerReward");
            }
        }

        rewardsProcessed[_taskId] = true;
        taskContract.updateTaskStatus(_taskId, AutonetLib.TaskStatus.REWARDED);
    }

    // ============ View Functions ============

    function getVotes(uint256 _taskId, address _solver) external view returns (AutonetLib.CoordinatorVote[] memory) {
        return votes[_taskId][_solver];
    }

    function getConsensusResult(uint256 _taskId, address _solver) external view returns (AutonetLib.YumaConsensusResult memory) {
        return consensusResults[_taskId][_solver];
    }

    function getCoordinatorBond(address _coordinator) external view returns (AutonetLib.CoordinatorBond memory) {
        return coordinatorBonds[_coordinator];
    }

    function getVoteCount(uint256 _taskId, address _solver) external view returns (uint256) {
        return votes[_taskId][_solver].length;
    }

    function isVotingOpen(uint256 _taskId, address _solver) external view returns (bool) {
        uint256 startBlock = votingStartBlock[_taskId][_solver];
        if (startBlock == 0) return false;
        return block.number <= startBlock + VOTING_PERIOD_BLOCKS;
    }
}
