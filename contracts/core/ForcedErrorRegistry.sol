// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../utils/AutonetLib.sol";
import "./ParticipantStaking.sol";
import "./RPB.sol";

/**
 * @title ForcedErrorRegistry
 * @dev Implements Truebit-style forced error detection to keep verifiers honest.
 *
 * How it works:
 * 1. Governance periodically injects known-bad "forced error" tasks
 * 2. These look like normal tasks but have deliberately incorrect solutions
 * 3. Coordinators who correctly identify forced errors receive jackpot rewards
 * 4. Coordinators who approve forced errors are slashed
 *
 * This solves the "verifier's dilemma" - without forced errors, verifiers
 * might rubber-stamp everything since real errors are rare.
 */
contract ForcedErrorRegistry {
    RPB public immutable rpb;
    ParticipantStaking public immutable staking;
    address public governance;

    // ============ Configuration ============
    uint256 public jackpotAmount = 50 * 1e18;           // 50 ATN jackpot
    uint256 public slashAmountForMissed = 25 * 1e18;    // 25 ATN slash for approving forced error
    uint256 public forcedErrorProbability = 50;          // 5% probability (50/1000)
    uint256 public expirationBlocks = 1000;              // ~3 hours to catch

    // ============ Storage ============

    // TaskID => ForcedError data
    mapping(uint256 => AutonetLib.ForcedError) public forcedErrors;

    // Track which tasks are forced errors (for quick lookup)
    mapping(uint256 => bool) public isForcedError;

    // Total jackpot pool
    uint256 public jackpotPool;

    // Nonce for pseudo-random forced error selection
    uint256 private _nonce;

    // ============ Events ============

    event ForcedErrorInjected(uint256 indexed taskId, bytes32 knownBadHash, uint256 jackpotAmount);
    event ForcedErrorCaught(uint256 indexed taskId, address indexed catcher, uint256 jackpotAwarded);
    event ForcedErrorMissed(uint256 indexed taskId, address indexed coordinator, uint256 slashAmount);
    event ForcedErrorExpired(uint256 indexed taskId);
    event JackpotFunded(address indexed funder, uint256 amount);
    event ConfigUpdated(string parameter, uint256 newValue);

    // ============ Modifiers ============

    modifier onlyGovernance() {
        require(msg.sender == governance, "Not governance");
        _;
    }

    // ============ Constructor ============

    constructor(address _rpb, address _staking, address _governance) {
        rpb = RPB(_rpb);
        staking = ParticipantStaking(_staking);
        governance = _governance;
    }

    // ============ Jackpot Funding ============

    /**
     * @dev Fund the jackpot pool. Anyone can contribute.
     */
    function fundJackpotPool(uint256 _amount) external {
        require(rpb.transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        jackpotPool += _amount;
        emit JackpotFunded(msg.sender, _amount);
    }

    // ============ Forced Error Management ============

    /**
     * @dev Inject a forced error for a task.
     * Called by governance when creating "trap" tasks.
     *
     * @param _taskId The task to mark as a forced error
     * @param _knownBadHash Hash of the deliberately incorrect solution
     */
    function injectForcedError(uint256 _taskId, bytes32 _knownBadHash) external onlyGovernance {
        require(!isForcedError[_taskId], "Already a forced error");
        require(jackpotPool >= jackpotAmount, "Insufficient jackpot pool");

        // Reserve jackpot from pool so concurrent injections can't over-commit
        jackpotPool -= jackpotAmount;

        forcedErrors[_taskId] = AutonetLib.ForcedError({
            taskId: _taskId,
            knownBadHash: _knownBadHash,
            jackpotAmount: jackpotAmount,
            expirationBlock: block.number + expirationBlocks,
            caught: false,
            catcher: address(0)
        });

        isForcedError[_taskId] = true;

        emit ForcedErrorInjected(_taskId, _knownBadHash, jackpotAmount);
    }

    /**
     * @dev Check if a task should randomly become a forced error.
     * Uses pseudo-random selection based on block data.
     * Called during task creation to probabilistically inject errors.
     */
    function shouldInjectForcedError() external returns (bool) {
        _nonce++;
        uint256 random = uint256(keccak256(abi.encodePacked(
            block.timestamp,
            block.prevrandao,
            msg.sender,
            _nonce
        ))) % 1000;

        return random < forcedErrorProbability && jackpotPool >= jackpotAmount;
    }

    /**
     * @dev Report catching a forced error.
     * Called by a coordinator who correctly identified a bad solution.
     *
     * @param _taskId The task ID
     * @param _solutionHash Hash of the solution they're flagging as bad
     */
    function reportForcedError(uint256 _taskId, bytes32 _solutionHash) external {
        require(isForcedError[_taskId], "Not a forced error task");
        require(staking.isActiveParticipant(msg.sender, AutonetLib.ParticipantRole.COORDINATOR), "Not active Coordinator");

        AutonetLib.ForcedError storage fe = forcedErrors[_taskId];
        require(!fe.caught, "Already caught");
        require(block.number <= fe.expirationBlock, "Forced error expired");
        require(_solutionHash == fe.knownBadHash, "Wrong solution hash");

        fe.caught = true;
        fe.catcher = msg.sender;

        // Award jackpot (already reserved from pool during injection)
        uint256 award = fe.jackpotAmount;
        require(rpb.transfer(msg.sender, award), "Jackpot transfer failed");

        emit ForcedErrorCaught(_taskId, msg.sender, award);
    }

    /**
     * @dev Process a coordinator who approved a forced error (missed it).
     * Called after verification is finalized if a coordinator approved a known-bad solution.
     *
     * @param _taskId The task ID
     * @param _coordinator The coordinator who approved the bad solution
     */
    function processCoordinatorMiss(uint256 _taskId, address _coordinator) external onlyGovernance {
        require(isForcedError[_taskId], "Not a forced error task");

        AutonetLib.ForcedError storage fe = forcedErrors[_taskId];
        require(fe.caught || block.number > fe.expirationBlock, "Error period still active");

        // Slash the coordinator who missed it
        staking.slash(_coordinator, slashAmountForMissed, "Missed forced error");

        emit ForcedErrorMissed(_taskId, _coordinator, slashAmountForMissed);
    }

    /**
     * @dev Mark an uncaught forced error as expired.
     * If no coordinator catches the error, it expires with no jackpot awarded.
     */
    function expireForcedError(uint256 _taskId) external {
        require(isForcedError[_taskId], "Not a forced error task");

        AutonetLib.ForcedError storage fe = forcedErrors[_taskId];
        require(!fe.caught, "Already caught");
        require(block.number > fe.expirationBlock, "Not yet expired");

        // Return reserved jackpot to pool (it wasn't awarded)
        jackpotPool += fe.jackpotAmount;

        emit ForcedErrorExpired(_taskId);
    }

    // ============ Configuration ============

    function setJackpotAmount(uint256 _amount) external onlyGovernance {
        jackpotAmount = _amount;
        emit ConfigUpdated("jackpotAmount", _amount);
    }

    function setSlashAmount(uint256 _amount) external onlyGovernance {
        slashAmountForMissed = _amount;
        emit ConfigUpdated("slashAmountForMissed", _amount);
    }

    function setForcedErrorProbability(uint256 _probability) external onlyGovernance {
        require(_probability <= 1000, "Probability too high");
        forcedErrorProbability = _probability;
        emit ConfigUpdated("forcedErrorProbability", _probability);
    }

    function setExpirationBlocks(uint256 _blocks) external onlyGovernance {
        expirationBlocks = _blocks;
        emit ConfigUpdated("expirationBlocks", _blocks);
    }

    function setGovernance(address _newGovernance) external onlyGovernance {
        governance = _newGovernance;
    }

    // ============ View Functions ============

    function getForcedError(uint256 _taskId) external view returns (AutonetLib.ForcedError memory) {
        return forcedErrors[_taskId];
    }

    function isTaskForcedError(uint256 _taskId) external view returns (bool) {
        return isForcedError[_taskId];
    }

    function isForcedErrorActive(uint256 _taskId) external view returns (bool) {
        if (!isForcedError[_taskId]) return false;
        AutonetLib.ForcedError memory fe = forcedErrors[_taskId];
        return !fe.caught && block.number <= fe.expirationBlock;
    }

    function getJackpotPool() external view returns (uint256) {
        return jackpotPool;
    }
}
