// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title GuildRegistry
 * @notice Manages training guilds — jurisdictions that specialize in training
 *         specific model modules.
 *
 * A guild owns training responsibility for a set of modules (bytes32 IDs from
 * CapabilityScorecard). Guild members stake ATN to join, the guild elects an
 * aggregator, and the aggregator submits guild-level updates to the network.
 *
 * Guild reputation is tracked via EMA of improvement scores reported by the
 * network aggregator after evaluating each guild's contribution.
 *
 * Stories: 8.1 (creation/membership) + 8.4 (reputation/competition)
 */
contract GuildRegistry is ReentrancyGuard {
    // =========================================================================
    // Constants
    // =========================================================================

    uint256 public constant MAX_SCORE = 10000; // Basis points
    uint256 public constant MULTIPLIER_PRECISION = 10000;
    uint256 public constant BASE_REPUTATION = 5000; // 50% starting reputation

    // =========================================================================
    // State
    // =========================================================================

    IERC20 public immutable atnToken;
    address public admin;

    uint256 public minGuildStake;   // Minimum ATN for guild creator
    uint256 public minMemberStake;  // Minimum ATN for guild members
    uint256 public nextGuildId;

    struct Guild {
        uint256 guildId;
        string name;
        address creator;
        address aggregator;        // Elected guild-level aggregator
        bytes32[] moduleIds;        // Modules this guild trains
        uint256 memberCount;
        uint256 totalStaked;
        uint256 createdAt;
        bool active;
    }

    struct GuildReputation {
        uint256 score;              // 0-10000 bps (EMA-smoothed)
        uint256 evaluationCount;
        uint256 lastUpdatedAt;
    }

    struct MemberInfo {
        uint256 guildId;
        uint256 stakedAmount;
        uint256 joinedAt;
        bool active;
    }

    // guildId → Guild
    mapping(uint256 => Guild) public guilds;

    // guildId → moduleId → GuildReputation
    mapping(uint256 => mapping(bytes32 => GuildReputation)) public guildReputation;

    // moduleId → guildId (a module can only be assigned to one guild at a time)
    mapping(bytes32 => uint256) public moduleToGuild;

    // address → guildId → MemberInfo
    mapping(address => mapping(uint256 => MemberInfo)) public members;

    // guildId → list of member addresses (for enumeration)
    mapping(uint256 => address[]) internal _guildMembers;

    // Who can report guild metrics (network aggregators)
    mapping(address => bool) public reporters;

    // =========================================================================
    // Events
    // =========================================================================

    event GuildCreated(
        uint256 indexed guildId,
        string name,
        address indexed creator,
        bytes32[] moduleIds
    );
    event MemberJoined(uint256 indexed guildId, address indexed member, uint256 stakeAmount);
    event MemberLeft(uint256 indexed guildId, address indexed member, uint256 stakeReturned);
    event AggregatorSet(uint256 indexed guildId, address indexed aggregator);
    event GuildMetricsRecorded(
        uint256 indexed guildId,
        bytes32 indexed moduleId,
        uint256 improvementScore,
        uint256 newReputation
    );
    event ReporterSet(address indexed reporter, bool enabled);
    event ModuleAssigned(bytes32 indexed moduleId, uint256 indexed guildId);
    event ModuleReleased(bytes32 indexed moduleId, uint256 indexed guildId);

    // =========================================================================
    // Errors
    // =========================================================================

    error NotAuthorized();
    error GuildNotFound();
    error GuildNotActive();
    error ModuleAlreadyAssigned(bytes32 moduleId, uint256 existingGuildId);
    error InsufficientStake(uint256 required, uint256 provided);
    error AlreadyMember();
    error NotMember();
    error NotGuildCreator();
    error InvalidInput();
    error TransferFailed();
    error NoModules();

    // =========================================================================
    // Modifiers
    // =========================================================================

    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAuthorized();
        _;
    }

    modifier onlyReporter() {
        if (!reporters[msg.sender] && msg.sender != admin) revert NotAuthorized();
        _;
    }

    modifier guildExists(uint256 guildId) {
        if (guilds[guildId].createdAt == 0) revert GuildNotFound();
        _;
    }

    modifier guildActive(uint256 guildId) {
        if (!guilds[guildId].active) revert GuildNotActive();
        _;
    }

    // =========================================================================
    // Constructor
    // =========================================================================

    constructor(address _atnToken, address _admin, uint256 _minGuildStake, uint256 _minMemberStake) {
        atnToken = IERC20(_atnToken);
        admin = _admin;
        minGuildStake = _minGuildStake;
        minMemberStake = _minMemberStake;
        nextGuildId = 1; // Start at 1 so 0 means "no guild"
    }

    // =========================================================================
    // Story 8.1: Guild Creation & Membership
    // =========================================================================

    /**
     * @notice Create a new guild responsible for training specific modules.
     * @param name Human-readable guild name
     * @param moduleIds Module IDs this guild will train (from CapabilityScorecard)
     */
    function createGuild(
        string calldata name,
        bytes32[] calldata moduleIds
    ) external nonReentrant returns (uint256 guildId) {
        if (bytes(name).length == 0) revert InvalidInput();
        if (moduleIds.length == 0) revert NoModules();

        // Check that no module is already assigned to another guild
        for (uint256 i = 0; i < moduleIds.length; i++) {
            uint256 existing = moduleToGuild[moduleIds[i]];
            if (existing != 0) {
                revert ModuleAlreadyAssigned(moduleIds[i], existing);
            }
        }

        // Transfer creator's stake
        if (minGuildStake > 0) {
            bool ok = atnToken.transferFrom(msg.sender, address(this), minGuildStake);
            if (!ok) revert TransferFailed();
        }

        guildId = nextGuildId++;

        // Store guild
        Guild storage g = guilds[guildId];
        g.guildId = guildId;
        g.name = name;
        g.creator = msg.sender;
        g.aggregator = msg.sender; // Creator is default aggregator
        g.memberCount = 1; // Creator counts as member
        g.totalStaked = minGuildStake;
        g.createdAt = block.timestamp;
        g.active = true;

        // Copy module IDs
        for (uint256 i = 0; i < moduleIds.length; i++) {
            g.moduleIds.push(moduleIds[i]);
            moduleToGuild[moduleIds[i]] = guildId;
            emit ModuleAssigned(moduleIds[i], guildId);
        }

        // Creator is automatically a member
        members[msg.sender][guildId] = MemberInfo({
            guildId: guildId,
            stakedAmount: minGuildStake,
            joinedAt: block.timestamp,
            active: true
        });
        _guildMembers[guildId].push(msg.sender);

        // Initialize reputation for each module at base level
        for (uint256 i = 0; i < moduleIds.length; i++) {
            guildReputation[guildId][moduleIds[i]] = GuildReputation({
                score: BASE_REPUTATION,
                evaluationCount: 0,
                lastUpdatedAt: block.timestamp
            });
        }

        emit GuildCreated(guildId, name, msg.sender, moduleIds);
    }

    /**
     * @notice Join an existing guild by staking ATN.
     * @param guildId The guild to join
     */
    function joinGuild(uint256 guildId) external nonReentrant guildExists(guildId) guildActive(guildId) {
        if (members[msg.sender][guildId].active) revert AlreadyMember();

        uint256 stakeRequired = minMemberStake;
        if (stakeRequired > 0) {
            bool ok = atnToken.transferFrom(msg.sender, address(this), stakeRequired);
            if (!ok) revert TransferFailed();
        }

        members[msg.sender][guildId] = MemberInfo({
            guildId: guildId,
            stakedAmount: stakeRequired,
            joinedAt: block.timestamp,
            active: true
        });
        _guildMembers[guildId].push(msg.sender);

        guilds[guildId].memberCount++;
        guilds[guildId].totalStaked += stakeRequired;

        emit MemberJoined(guildId, msg.sender, stakeRequired);
    }

    /**
     * @notice Leave a guild and reclaim staked ATN.
     * @param guildId The guild to leave
     */
    function leaveGuild(uint256 guildId) external nonReentrant guildExists(guildId) {
        MemberInfo storage info = members[msg.sender][guildId];
        if (!info.active) revert NotMember();

        // Cannot leave if you're the aggregator (must transfer first)
        if (guilds[guildId].aggregator == msg.sender && guilds[guildId].memberCount > 1) {
            revert NotAuthorized(); // Must set a new aggregator first
        }

        uint256 stakeToReturn = info.stakedAmount;
        info.active = false;
        info.stakedAmount = 0;

        guilds[guildId].memberCount--;
        guilds[guildId].totalStaked -= stakeToReturn;

        // Return stake
        if (stakeToReturn > 0) {
            bool ok = atnToken.transfer(msg.sender, stakeToReturn);
            if (!ok) revert TransferFailed();
        }

        emit MemberLeft(guildId, msg.sender, stakeToReturn);
    }

    /**
     * @notice Set the guild's aggregator. Only callable by guild creator.
     * @param guildId The guild
     * @param aggregator Address of the new aggregator (must be a member)
     */
    function setGuildAggregator(
        uint256 guildId,
        address aggregator
    ) external guildExists(guildId) guildActive(guildId) {
        if (msg.sender != guilds[guildId].creator) revert NotGuildCreator();
        if (!members[aggregator][guildId].active) revert NotMember();

        guilds[guildId].aggregator = aggregator;
        emit AggregatorSet(guildId, aggregator);
    }

    // =========================================================================
    // Story 8.4: Guild Reputation & Competition
    // =========================================================================

    /**
     * @notice Record improvement metrics for a guild's module training.
     * @dev Called by network aggregator after evaluating a guild's contribution.
     *      Uses EMA (same pattern as CapabilityScorecard) to smooth scores.
     * @param guildId The guild being evaluated
     * @param moduleId The module that was trained
     * @param improvementScore How much the guild improved this module (0-10000 bps)
     */
    function recordGuildMetrics(
        uint256 guildId,
        bytes32 moduleId,
        uint256 improvementScore
    ) external onlyReporter guildExists(guildId) {
        if (improvementScore > MAX_SCORE) revert InvalidInput();

        GuildReputation storage rep = guildReputation[guildId][moduleId];

        uint256 oldScore = rep.score;
        uint256 count = rep.evaluationCount;

        if (count == 0) {
            // First evaluation: set directly
            rep.score = improvementScore;
        } else {
            // EMA: new = (old * min(count,20) + reported) / (min(count,20) + 1)
            uint256 effectiveCount = count > 20 ? 20 : count;
            rep.score = (oldScore * effectiveCount + improvementScore) / (effectiveCount + 1);
        }

        rep.evaluationCount = count + 1;
        rep.lastUpdatedAt = block.timestamp;

        emit GuildMetricsRecorded(guildId, moduleId, improvementScore, rep.score);
    }

    /**
     * @notice Get guild ranking for a specific module, sorted by reputation.
     * @dev Returns guildIds ranked from highest to lowest reputation score.
     *      In V1, only one guild per module. This supports future multi-guild
     *      competition for the same module.
     * @param moduleId The module to rank guilds for
     * @return rankedGuildIds Guild IDs sorted by reputation (best first)
     * @return scores Corresponding reputation scores
     */
    function getGuildRanking(bytes32 moduleId) external view returns (
        uint256[] memory rankedGuildIds,
        uint256[] memory scores
    ) {
        // In V1, one guild per module. Build array of all guilds training this module.
        // For future: iterate all guilds and filter those with this moduleId.
        uint256 assignedGuild = moduleToGuild[moduleId];
        if (assignedGuild == 0) {
            return (new uint256[](0), new uint256[](0));
        }

        rankedGuildIds = new uint256[](1);
        scores = new uint256[](1);
        rankedGuildIds[0] = assignedGuild;
        scores[0] = guildReputation[assignedGuild][moduleId].score;
    }

    /**
     * @notice Compute a reward multiplier for a guild based on its reputation.
     * @dev High-reputation guilds get a bonus multiplier (up to 1.5x = 15000).
     *      Low-reputation guilds get a penalty (down to 0.5x = 5000).
     *      Base = 10000 (1x) at reputation score 5000 (50%).
     * @param guildId The guild
     * @param moduleId The module
     * @return multiplier In basis points (10000 = 1x)
     */
    function getGuildRewardMultiplier(
        uint256 guildId,
        bytes32 moduleId
    ) external view returns (uint256) {
        GuildReputation storage rep = guildReputation[guildId][moduleId];
        if (rep.lastUpdatedAt == 0) {
            return MULTIPLIER_PRECISION; // Base 1x for unknown
        }

        // Linear: 0 reputation → 5000 (0.5x), 5000 → 10000 (1x), 10000 → 15000 (1.5x)
        // multiplier = 5000 + (score * 10000 / 10000) = 5000 + score
        return 5000 + rep.score;
    }

    // =========================================================================
    // View Functions
    // =========================================================================

    /**
     * @notice Get full guild details.
     */
    function getGuild(uint256 guildId) external view guildExists(guildId) returns (
        string memory name,
        bytes32[] memory moduleIds,
        uint256 memberCount,
        address aggregator,
        address creator,
        uint256 totalStaked,
        bool active
    ) {
        Guild storage g = guilds[guildId];
        return (g.name, g.moduleIds, g.memberCount, g.aggregator, g.creator, g.totalStaked, g.active);
    }

    /**
     * @notice Get the guild ID assigned to a module.
     * @return guildId (0 if unassigned)
     */
    function getModuleGuild(bytes32 moduleId) external view returns (uint256) {
        return moduleToGuild[moduleId];
    }

    /**
     * @notice Check if an address is an active member of a guild.
     */
    function isMember(address addr, uint256 guildId) external view returns (bool) {
        return members[addr][guildId].active;
    }

    /**
     * @notice Check if an address is the aggregator for a guild.
     */
    function isGuildAggregator(address addr, uint256 guildId) external view returns (bool) {
        return guilds[guildId].aggregator == addr;
    }

    /**
     * @notice Get the member list for a guild.
     */
    function getGuildMembers(uint256 guildId) external view returns (address[] memory) {
        return _guildMembers[guildId];
    }

    /**
     * @notice Get the total number of guilds created.
     */
    function getGuildCount() external view returns (uint256) {
        return nextGuildId - 1;
    }

    /**
     * @notice Get guild reputation for a specific module.
     */
    function getReputation(uint256 guildId, bytes32 moduleId) external view returns (
        uint256 score,
        uint256 evaluationCount,
        uint256 lastUpdatedAt
    ) {
        GuildReputation storage rep = guildReputation[guildId][moduleId];
        return (rep.score, rep.evaluationCount, rep.lastUpdatedAt);
    }

    // =========================================================================
    // Admin
    // =========================================================================

    function setReporter(address reporter, bool enabled) external onlyAdmin {
        reporters[reporter] = enabled;
        emit ReporterSet(reporter, enabled);
    }

    function setMinGuildStake(uint256 amount) external onlyAdmin {
        minGuildStake = amount;
    }

    function setMinMemberStake(uint256 amount) external onlyAdmin {
        minMemberStake = amount;
    }

    function transferAdmin(address newAdmin) external onlyAdmin {
        if (newAdmin == address(0)) revert InvalidInput();
        admin = newAdmin;
    }
}
