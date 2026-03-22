// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title CapabilityScorecard
 * @notice Tracks per-module capability scores for the Autonet training network.
 *
 * Modules the network lacks should pay more to train. This contract maintains
 * a scorecard of module capabilities and derives reward multipliers from
 * capability gaps. The same pricing function described in the emergent
 * alignment paper for inference is applied here to training incentives.
 *
 * Score range: 0 (no capability) to 10000 (fully capable, basis points).
 * Multiplier: inversely proportional to score. Low score = high multiplier.
 *
 * Coordinators or governance-approved oracles update scores after evaluating
 * training results. Proposers read the scorecard to create tasks targeting
 * underserved modules (higher reward multiplier = more ATN for training that module).
 */
contract CapabilityScorecard is ReentrancyGuard {
    uint256 public constant MAX_SCORE = 10000; // Basis points
    uint256 public constant MULTIPLIER_PRECISION = 10000;

    // Base multiplier when score equals target (1x = 10000)
    uint256 public constant BASE_MULTIPLIER = 10000;
    // Maximum multiplier for zero-capability modules (3x = 30000)
    uint256 public maxMultiplier = 30000;
    // Minimum multiplier for saturated modules (0.5x = 5000)
    uint256 public minMultiplier = 5000;

    address public admin;
    address public autonetAddress; // Autonet.sol for integration

    struct ModuleScore {
        bytes32 moduleId;       // e.g., keccak256("visual_encoder"), keccak256("text_decoder")
        uint256 score;          // 0-10000 bps
        uint256 targetScore;    // Desired capability level (default 8000 = 80%)
        uint256 lastUpdatedAt;
        address lastUpdatedBy;
        uint256 evaluationCount; // How many evaluations contributed
    }

    // moduleId → ModuleScore
    mapping(bytes32 => ModuleScore) public modules;
    bytes32[] public moduleIds;

    // Who can update scores (coordinators, oracles)
    mapping(address => bool) public evaluators;

    // --- EVENTS ---
    event ModuleRegistered(bytes32 indexed moduleId, uint256 targetScore);
    event ScoreUpdated(bytes32 indexed moduleId, uint256 oldScore, uint256 newScore, address indexed evaluator);
    event EvaluatorSet(address indexed evaluator, bool enabled);
    event MultiplierBoundsUpdated(uint256 minMultiplier, uint256 maxMultiplier);

    // --- ERRORS ---
    error NotAuthorized();
    error InvalidScore();
    error ModuleNotFound();
    error ModuleAlreadyExists();
    error ZeroAddress();

    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAuthorized();
        _;
    }

    modifier onlyEvaluator() {
        if (!evaluators[msg.sender] && msg.sender != admin) revert NotAuthorized();
        _;
    }

    constructor(address _admin) {
        if (_admin == address(0)) revert ZeroAddress();
        admin = _admin;
        evaluators[_admin] = true;
    }

    // =========================================================================
    // MODULE MANAGEMENT
    // =========================================================================

    /**
     * @notice Register a new training module with a target score.
     * @param moduleId Identifier (e.g., keccak256("visual_encoder"))
     * @param targetScore Desired capability level in basis points (e.g., 8000 = 80%)
     */
    function registerModule(bytes32 moduleId, uint256 targetScore) external onlyAdmin {
        if (modules[moduleId].lastUpdatedAt != 0) revert ModuleAlreadyExists();
        if (targetScore == 0 || targetScore > MAX_SCORE) revert InvalidScore();

        modules[moduleId] = ModuleScore({
            moduleId: moduleId,
            score: 0,
            targetScore: targetScore,
            lastUpdatedAt: block.timestamp,
            lastUpdatedBy: msg.sender,
            evaluationCount: 0
        });
        moduleIds.push(moduleId);

        emit ModuleRegistered(moduleId, targetScore);
    }

    /**
     * @notice Update a module's capability score.
     * @dev Uses exponential moving average: new_score = (old * (n-1) + reported) / n
     *      This prevents a single evaluator from making large jumps.
     * @param moduleId The module to update
     * @param reportedScore The evaluator's assessed score (0-10000 bps)
     */
    function updateScore(bytes32 moduleId, uint256 reportedScore) external onlyEvaluator {
        ModuleScore storage mod = modules[moduleId];
        if (mod.lastUpdatedAt == 0) revert ModuleNotFound();
        if (reportedScore > MAX_SCORE) revert InvalidScore();

        uint256 oldScore = mod.score;
        uint256 count = mod.evaluationCount;

        if (count == 0) {
            mod.score = reportedScore;
        } else {
            // EMA: new = (old * count + reported) / (count + 1)
            // Capped at 20 evaluations to allow faster movement over time
            uint256 effectiveCount = count > 20 ? 20 : count;
            mod.score = (oldScore * effectiveCount + reportedScore) / (effectiveCount + 1);
        }

        mod.evaluationCount = count + 1;
        mod.lastUpdatedAt = block.timestamp;
        mod.lastUpdatedBy = msg.sender;

        emit ScoreUpdated(moduleId, oldScore, mod.score, msg.sender);
    }

    // =========================================================================
    // REWARD MULTIPLIER COMPUTATION
    // =========================================================================

    /**
     * @notice Compute the reward multiplier for training a given module.
     * @dev Linear interpolation between maxMultiplier (score=0) and
     *      minMultiplier (score >= targetScore).
     *
     *      multiplier = maxMultiplier - ((score / targetScore) * (maxMultiplier - minMultiplier))
     *
     *      If score >= targetScore, multiplier = minMultiplier (module is saturated).
     *      If score = 0, multiplier = maxMultiplier (module is missing).
     *
     * @param moduleId The module to query
     * @return multiplier In basis points (10000 = 1x, 30000 = 3x, 5000 = 0.5x)
     */
    function getRewardMultiplier(bytes32 moduleId) public view returns (uint256) {
        ModuleScore storage mod = modules[moduleId];
        if (mod.lastUpdatedAt == 0) {
            // Unknown module — base multiplier
            return BASE_MULTIPLIER;
        }

        if (mod.score >= mod.targetScore) {
            return minMultiplier; // Saturated
        }

        // Linear interpolation: from maxMultiplier (score=0) to minMultiplier (score=target)
        uint256 range = maxMultiplier - minMultiplier;
        uint256 progress = (mod.score * MULTIPLIER_PRECISION) / mod.targetScore;
        uint256 reduction = (range * progress) / MULTIPLIER_PRECISION;

        return maxMultiplier - reduction;
    }

    /**
     * @notice Get all module IDs and their current multipliers.
     * @dev Used by proposers to find the highest-value training targets.
     */
    function getScorecard() external view returns (
        bytes32[] memory ids,
        uint256[] memory scores,
        uint256[] memory targets,
        uint256[] memory multipliers
    ) {
        uint256 len = moduleIds.length;
        ids = new bytes32[](len);
        scores = new uint256[](len);
        targets = new uint256[](len);
        multipliers = new uint256[](len);

        for (uint256 i = 0; i < len; i++) {
            bytes32 id = moduleIds[i];
            ModuleScore storage mod = modules[id];
            ids[i] = id;
            scores[i] = mod.score;
            targets[i] = mod.targetScore;
            multipliers[i] = getRewardMultiplier(id);
        }
    }

    /**
     * @notice Get the number of registered modules.
     */
    function getModuleCount() external view returns (uint256) {
        return moduleIds.length;
    }

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    function setEvaluator(address evaluator, bool enabled) external onlyAdmin {
        if (evaluator == address(0)) revert ZeroAddress();
        evaluators[evaluator] = enabled;
        emit EvaluatorSet(evaluator, enabled);
    }

    function setMultiplierBounds(uint256 _min, uint256 _max) external onlyAdmin {
        if (_min == 0 || _max <= _min) revert InvalidScore();
        minMultiplier = _min;
        maxMultiplier = _max;
        emit MultiplierBoundsUpdated(_min, _max);
    }

    function setAutonetAddress(address _autonet) external onlyAdmin {
        if (_autonet == address(0)) revert ZeroAddress();
        autonetAddress = _autonet;
    }

    function transferAdmin(address newAdmin) external onlyAdmin {
        if (newAdmin == address(0)) revert ZeroAddress();
        admin = newAdmin;
    }
}
