// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../utils/AutonetLib.sol";
import "./ParticipantStaking.sol";
import "./RPB.sol";

/**
 * @title TaskContract
 * @dev Manages the Absolute Zero training task lifecycle:
 *      proposal, claiming, solution commitment, and status updates.
 */
contract TaskContract {
    ParticipantStaking public immutable staking;
    RPB public immutable rpb;
    address public governance;
    address public resultsRewardsContract;

    uint256 public nextTaskId = 1;

    struct Task {
        uint256 id;
        address rpbAddress;
        AutonetLib.TaskProposal proposal;
        address[] committedSolvers;
    }

    mapping(uint256 => Task) private _tasks;
    mapping(uint256 => mapping(address => AutonetLib.SolverSubmission)) public submissions;

    // Gensyn-style checkpoint storage: TaskID => Solver => Checkpoints
    mapping(uint256 => mapping(address => AutonetLib.TrainingCheckpoint[])) public checkpoints;
    // TaskID => Solver => checkpoint frequency (steps between checkpoints)
    mapping(uint256 => mapping(address => uint256)) public checkpointFrequency;

    event TaskProposed(
        uint256 indexed taskId,
        address indexed rpbAddress,
        address indexed proposer,
        bytes32 specHash,
        bytes32 groundTruthHash
    );
    event TaskActivated(uint256 indexed taskId);
    event SolutionCommitted(uint256 indexed taskId, address indexed solver, bytes32 solutionHash);
    event TaskStatusUpdated(uint256 indexed taskId, AutonetLib.TaskStatus newStatus);
    event CheckpointSubmitted(uint256 indexed taskId, address indexed solver, uint256 stepNumber, bytes32 weightsHash);

    modifier taskExists(uint256 _taskId) {
        require(_tasks[_taskId].id != 0, "Task does not exist");
        _;
    }

    modifier onlyStakedProposer() {
        require(staking.isActiveParticipant(msg.sender, AutonetLib.ParticipantRole.PROPOSER), "Not active Proposer");
        _;
    }

    modifier onlyStakedSolver() {
        require(staking.isActiveParticipant(msg.sender, AutonetLib.ParticipantRole.SOLVER), "Not active Solver");
        _;
    }

    modifier onlyGovernance() {
        require(msg.sender == governance, "Not governance");
        _;
    }

    constructor(address _staking, address _governance, address _rpb) {
        staking = ParticipantStaking(_staking);
        governance = _governance;
        rpb = RPB(_rpb);
    }

    function setResultsRewardsContract(address _contract) external onlyGovernance {
        resultsRewardsContract = _contract;
    }

    function proposeTask(
        address _rpbAddress,
        bytes32 _specHash,
        bytes32 _groundTruthHash,
        uint256 _learnabilityReward,
        uint256 _solverReward
    ) external onlyStakedProposer returns (uint256 taskId) {
        taskId = nextTaskId++;

        Task storage t = _tasks[taskId];
        t.id = taskId;
        t.rpbAddress = _rpbAddress;
        t.proposal = AutonetLib.TaskProposal({
            specHash: _specHash,
            groundTruthSolHash: _groundTruthHash,
            proposer: msg.sender,
            proposedLearnabilityReward: _learnabilityReward,
            proposedSolverReward: _solverReward,
            creationBlock: block.number,
            status: AutonetLib.TaskStatus.PROPOSED,
            rpbAddress: _rpbAddress,
            taskMode: AutonetLib.TaskMode.GROUND_TRUTH
        });

        emit TaskProposed(taskId, _rpbAddress, msg.sender, _specHash, _groundTruthHash);

        // Auto-activate for MVP
        _activateTask(taskId);
    }

    function _activateTask(uint256 _taskId) internal {
        _tasks[_taskId].proposal.status = AutonetLib.TaskStatus.ACTIVE;
        emit TaskActivated(_taskId);
    }

    function activateTask(uint256 _taskId) external onlyGovernance taskExists(_taskId) {
        require(_tasks[_taskId].proposal.status == AutonetLib.TaskStatus.PROPOSED, "Not in proposed state");
        _activateTask(_taskId);
    }

    function commitSolution(uint256 _taskId, bytes32 _solutionHash, bytes32 _charterCid)
        external taskExists(_taskId) onlyStakedSolver
    {
        Task storage t = _tasks[_taskId];
        require(
            t.proposal.status == AutonetLib.TaskStatus.ACTIVE ||
            t.proposal.status == AutonetLib.TaskStatus.SOLUTION_COMMITTED,
            "Task not accepting solutions"
        );

        submissions[_taskId][msg.sender] = AutonetLib.SolverSubmission({
            solutionHash: _solutionHash,
            charterCid: _charterCid,
            solver: msg.sender,
            submissionBlock: block.number,
            score: 0
        });

        // Track unique solvers
        bool found = false;
        for (uint i = 0; i < t.committedSolvers.length; i++) {
            if (t.committedSolvers[i] == msg.sender) {
                found = true;
                break;
            }
        }
        if (!found) {
            t.committedSolvers.push(msg.sender);
        }

        t.proposal.status = AutonetLib.TaskStatus.SOLUTION_COMMITTED;
        emit SolutionCommitted(_taskId, msg.sender, _solutionHash);
    }

    // ============ Consensus-as-Truth (MM-Zero) Functions ============

    // TaskID => DifficultyTarget
    mapping(uint256 => AutonetLib.DifficultyTarget) public difficultyTargets;
    // TaskID => SolverRollout[]
    mapping(uint256 => AutonetLib.SolverRollout[]) private _rollouts;
    // TaskID => Solver => has submitted rollout
    mapping(uint256 => mapping(address => bool)) public hasSubmittedRollout;
    // TaskID => ConsensusResult
    mapping(uint256 => AutonetLib.ConsensusResult) public consensusResults;

    uint256 public constant MIN_ROLLOUTS = 2;
    uint256 public constant ROLLOUT_PERIOD_BLOCKS = 50; // ~10 minutes

    event ConsensusTaskProposed(
        uint256 indexed taskId,
        address indexed rpbAddress,
        address indexed proposer,
        bytes32 specHash,
        uint256 peakSolvability
    );
    event RolloutSubmitted(uint256 indexed taskId, address indexed solver, bytes32 answerHash, uint256 confidence);
    event ConsensusFinalized(uint256 indexed taskId, bytes32 majorityAnswer, uint256 actualSolvability, uint256 difficultyScore);

    /**
     * @dev Propose a consensus-mode task. No ground truth needed.
     * The Proposer specifies a difficulty target instead.
     */
    function proposeConsensusTask(
        address _rpbAddress,
        bytes32 _specHash,
        AutonetLib.DifficultyTarget calldata _difficultyTarget,
        uint256 _learnabilityReward,
        uint256 _solverReward
    ) external onlyStakedProposer returns (uint256 taskId) {
        require(_difficultyTarget.minSolvability < _difficultyTarget.maxSolvability, "Invalid solvability range");
        require(
            _difficultyTarget.peakSolvability >= _difficultyTarget.minSolvability &&
            _difficultyTarget.peakSolvability <= _difficultyTarget.maxSolvability,
            "Peak outside range"
        );
        require(_difficultyTarget.maxSolvability <= 10000, "Max solvability exceeds 100%");

        taskId = nextTaskId++;

        Task storage t = _tasks[taskId];
        t.id = taskId;
        t.rpbAddress = _rpbAddress;
        t.proposal = AutonetLib.TaskProposal({
            specHash: _specHash,
            groundTruthSolHash: bytes32(0), // No ground truth in consensus mode
            proposer: msg.sender,
            proposedLearnabilityReward: _learnabilityReward,
            proposedSolverReward: _solverReward,
            creationBlock: block.number,
            status: AutonetLib.TaskStatus.CONSENSUS_COLLECTING,
            rpbAddress: _rpbAddress,
            taskMode: AutonetLib.TaskMode.CONSENSUS_TRUTH
        });

        difficultyTargets[taskId] = _difficultyTarget;

        emit ConsensusTaskProposed(taskId, _rpbAddress, msg.sender, _specHash, _difficultyTarget.peakSolvability);
    }

    /**
     * @dev Submit a rollout for a consensus-mode task.
     * Multiple solvers can submit independent rollouts.
     */
    function submitRollout(
        uint256 _taskId,
        bytes32 _answerHash,
        bytes32 _solutionHash,
        uint256 _confidence
    ) external taskExists(_taskId) onlyStakedSolver {
        Task storage t = _tasks[_taskId];
        require(t.proposal.taskMode == AutonetLib.TaskMode.CONSENSUS_TRUTH, "Not consensus task");
        require(t.proposal.status == AutonetLib.TaskStatus.CONSENSUS_COLLECTING, "Not collecting rollouts");
        require(!hasSubmittedRollout[_taskId][msg.sender], "Already submitted rollout");
        require(_confidence <= 100, "Confidence must be 0-100");
        require(block.number <= t.proposal.creationBlock + ROLLOUT_PERIOD_BLOCKS, "Rollout period ended");

        _rollouts[_taskId].push(AutonetLib.SolverRollout({
            solver: msg.sender,
            answerHash: _answerHash,
            solutionHash: _solutionHash,
            confidence: _confidence,
            submitBlock: block.number
        }));

        hasSubmittedRollout[_taskId][msg.sender] = true;

        emit RolloutSubmitted(_taskId, msg.sender, _answerHash, _confidence);
    }

    /**
     * @dev Get all rollouts for a task.
     */
    function getRollouts(uint256 _taskId) external view returns (AutonetLib.SolverRollout[] memory) {
        return _rollouts[_taskId];
    }

    /**
     * @dev Get rollout count for a task.
     */
    function getRolloutCount(uint256 _taskId) external view returns (uint256) {
        return _rollouts[_taskId].length;
    }

    /**
     * @dev Get difficulty target for a task.
     */
    function getDifficultyTarget(uint256 _taskId) external view returns (AutonetLib.DifficultyTarget memory) {
        return difficultyTargets[_taskId];
    }

    /**
     * @dev Check if rollout collection is complete (enough rollouts or period ended).
     */
    function isRolloutCollectionComplete(uint256 _taskId) external view returns (bool) {
        Task storage t = _tasks[_taskId];
        if (t.proposal.taskMode != AutonetLib.TaskMode.CONSENSUS_TRUTH) return false;
        if (t.proposal.status != AutonetLib.TaskStatus.CONSENSUS_COLLECTING) return false;

        return _rollouts[_taskId].length >= MIN_ROLLOUTS ||
               block.number > t.proposal.creationBlock + ROLLOUT_PERIOD_BLOCKS;
    }

    // ============ Gensyn-style Checkpoint Functions ============

    /**
     * @dev Submit a training checkpoint for partial verification.
     * Checkpoints enable Gensyn-style dispute resolution where only
     * the first disagreeing step needs to be re-computed.
     *
     * @param _taskId The task being trained
     * @param _stepNumber Training step number (batch or epoch)
     * @param _weightsHash Hash of model weights at this step
     * @param _dataIndicesHash Hash of dataset indices used in this step
     * @param _randomSeed Deterministic seed for reproducibility (RepOps)
     */
    function submitCheckpoint(
        uint256 _taskId,
        uint256 _stepNumber,
        bytes32 _weightsHash,
        bytes32 _dataIndicesHash,
        bytes32 _randomSeed
    ) external taskExists(_taskId) onlyStakedSolver {
        Task storage t = _tasks[_taskId];
        require(
            t.proposal.status == AutonetLib.TaskStatus.ACTIVE ||
            t.proposal.status == AutonetLib.TaskStatus.CLAIMED,
            "Task not in training state"
        );

        // Ensure checkpoints are submitted in order
        AutonetLib.TrainingCheckpoint[] storage solverCheckpoints = checkpoints[_taskId][msg.sender];
        if (solverCheckpoints.length > 0) {
            require(
                _stepNumber > solverCheckpoints[solverCheckpoints.length - 1].stepNumber,
                "Checkpoint step must be increasing"
            );
        }

        solverCheckpoints.push(AutonetLib.TrainingCheckpoint({
            stepNumber: _stepNumber,
            weightsHash: _weightsHash,
            dataIndicesHash: _dataIndicesHash,
            randomSeed: _randomSeed,
            timestamp: block.timestamp
        }));

        emit CheckpointSubmitted(_taskId, msg.sender, _stepNumber, _weightsHash);
    }

    /**
     * @dev Set checkpoint frequency for a task.
     * This defines how often the solver will checkpoint during training.
     */
    function setCheckpointFrequency(uint256 _taskId, uint256 _frequency)
        external taskExists(_taskId) onlyStakedSolver
    {
        require(_frequency > 0, "Frequency must be positive");
        checkpointFrequency[_taskId][msg.sender] = _frequency;
    }

    /**
     * @dev Get all checkpoints for a solver's training run.
     */
    function getCheckpoints(uint256 _taskId, address _solver)
        external view returns (AutonetLib.TrainingCheckpoint[] memory)
    {
        return checkpoints[_taskId][_solver];
    }

    /**
     * @dev Get a specific checkpoint by index.
     */
    function getCheckpoint(uint256 _taskId, address _solver, uint256 _index)
        external view returns (AutonetLib.TrainingCheckpoint memory)
    {
        require(_index < checkpoints[_taskId][_solver].length, "Index out of bounds");
        return checkpoints[_taskId][_solver][_index];
    }

    /**
     * @dev Get checkpoint count for a solver.
     */
    function getCheckpointCount(uint256 _taskId, address _solver) external view returns (uint256) {
        return checkpoints[_taskId][_solver].length;
    }

    /**
     * @dev Find checkpoint at or before a given step number (for dispute pinpointing).
     * Returns the index of the checkpoint, or reverts if none found.
     */
    function findCheckpointAtStep(uint256 _taskId, address _solver, uint256 _targetStep)
        external view returns (uint256 checkpointIndex)
    {
        AutonetLib.TrainingCheckpoint[] storage solverCheckpoints = checkpoints[_taskId][_solver];
        require(solverCheckpoints.length > 0, "No checkpoints");

        // Binary search for checkpoint at or before target step
        uint256 low = 0;
        uint256 high = solverCheckpoints.length;

        while (low < high) {
            uint256 mid = (low + high) / 2;
            if (solverCheckpoints[mid].stepNumber <= _targetStep) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        require(low > 0, "No checkpoint at or before step");
        return low - 1;
    }

    // ============ Inference Tasks (BME) ============

    /// @notice InferenceTaskSubmitted: emitted when a user burns ATN for an inference task
    event InferenceTaskSubmitted(
        bytes32 indexed inferenceId,
        address indexed requester,
        uint256 burnedAmount,
        bytes32 inputCidHash,
        address rpbAddress
    );

    event InferenceTaskCompleted(
        bytes32 indexed inferenceId,
        address indexed provider,
        bytes32 outputCidHash
    );

    // inferenceId => InferenceTask metadata
    mapping(bytes32 => AutonetLib.InferenceTask) public inferenceTasks;
    uint256 public nextInferenceNonce;

    /**
     * @dev Submit an inference task with BME burn-on-submission.
     *
     *      The caller pays by burning ATN immediately. This creates an on-chain
     *      record that off-chain inference nodes can listen for. The actual
     *      provider payment (mint) is handled by InferenceProviderBridge.
     *
     *      On-chain record enables:
     *      - Auditability of inference demand
     *      - ForcedError-style verification for inference outputs
     *      - Reward attribution for the inference registry
     *
     * @param _rpbAddress  RPB jurisdiction whose model should serve this request
     * @param _inputCidHash Hash of the input content identifier
     * @param _burnAmount   ATN to burn (user's max payment)
     * @return inferenceId  Unique identifier for this inference task
     */
    function submitInferenceTask(
        address _rpbAddress,
        bytes32 _inputCidHash,
        uint256 _burnAmount
    ) external returns (bytes32 inferenceId) {
        require(_burnAmount > 0, "Zero burn amount");

        inferenceId = keccak256(abi.encodePacked(
            block.chainid,
            address(this),
            msg.sender,
            nextInferenceNonce++,
            block.timestamp
        ));

        // BME burn: tokens destroyed on submission, provider gets minted on completion
        require(rpb.transferFrom(msg.sender, address(this), _burnAmount), "Transfer failed");
        rpb.burn(_burnAmount);

        inferenceTasks[inferenceId] = AutonetLib.InferenceTask({
            requestId: inferenceId,
            requester: msg.sender,
            provider: address(0),
            burnedAmount: _burnAmount,
            mintedToProvider: 0,
            protocolFee: 0,
            inputCidHash: _inputCidHash,
            outputCidHash: bytes32(0),
            creationBlock: block.number,
            completed: false,
            verified: false
        });

        emit InferenceTaskSubmitted(inferenceId, msg.sender, _burnAmount, _inputCidHash, _rpbAddress);
    }

    /**
     * @dev Record completion of an inference task (called by authorized bridge or governance).
     *      Marks task as complete with output CID hash for auditability.
     */
    function recordInferenceCompletion(
        bytes32 _inferenceId,
        address _provider,
        bytes32 _outputCidHash
    ) external {
        require(msg.sender == resultsRewardsContract || msg.sender == governance, "Not authorized");
        AutonetLib.InferenceTask storage t = inferenceTasks[_inferenceId];
        require(t.creationBlock > 0, "Inference task not found");
        require(!t.completed, "Already completed");

        t.provider = _provider;
        t.outputCidHash = _outputCidHash;
        t.completed = true;

        emit InferenceTaskCompleted(_inferenceId, _provider, _outputCidHash);
    }

    /**
     * @dev Get inference task details.
     */
    function getInferenceTask(bytes32 _inferenceId)
        external view returns (AutonetLib.InferenceTask memory)
    {
        return inferenceTasks[_inferenceId];
    }

    // ============ Status Updates ============

    function updateTaskStatus(uint256 _taskId, AutonetLib.TaskStatus _newStatus)
        external taskExists(_taskId)
    {
        require(msg.sender == resultsRewardsContract || msg.sender == governance, "Not authorized");
        _tasks[_taskId].proposal.status = _newStatus;
        emit TaskStatusUpdated(_taskId, _newStatus);
    }

    // View functions
    function getTask(uint256 _taskId) external view returns (
        uint256 id,
        address rpbAddress,
        address proposer,
        AutonetLib.TaskStatus status,
        uint256 solverReward,
        uint256 learnabilityReward
    ) {
        Task storage t = _tasks[_taskId];
        return (
            t.id,
            t.rpbAddress,
            t.proposal.proposer,
            t.proposal.status,
            t.proposal.proposedSolverReward,
            t.proposal.proposedLearnabilityReward
        );
    }

    function getTaskProposal(uint256 _taskId) external view returns (AutonetLib.TaskProposal memory) {
        return _tasks[_taskId].proposal;
    }

    function getSubmission(uint256 _taskId, address _solver) external view returns (AutonetLib.SolverSubmission memory) {
        return submissions[_taskId][_solver];
    }

    function getCommittedSolvers(uint256 _taskId) external view returns (address[] memory) {
        return _tasks[_taskId].committedSolvers;
    }
}
