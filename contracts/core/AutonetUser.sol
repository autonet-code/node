// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IAutonetUser} from "../interfaces/IAutonetUser.sol";

/**
 * @title AutonetUser
 * @notice User identity contract for the Autonet system.
 *
 * Deployed per-user via Autonet.createUserContract().
 * Stores on-chain identity and public preferences.
 *
 * What's on-chain (this contract):
 * - Wallet ownership proof
 * - Standards agreement hash
 * - Public preferences
 * - Optional alignment score
 * - Usage statistics
 *
 * What's off-chain (Chevin/Firestore):
 * - Goals & aspirations
 * - Detailed standards interpretation
 * - Conversation history
 * - Service credentials
 */
contract AutonetUser is IAutonetUser {
    // --- IDENTITY ---
    address public immutable override owner;
    address public immutable override autonet;
    uint256 public immutable override createdAt;

    // --- STANDARDS ---
    bytes32 public override jurisdictionStandardsHash;
    uint256 public override agreedToStandardsAt;

    // --- PREFERENCES ---
    mapping(string => string) private preferences;
    string[] private preferenceKeys;
    mapping(string => bool) private preferenceKeyExists;

    // --- ALIGNMENT ---
    uint256 public override alignmentScore;
    uint256 public override alignmentUpdatedAt;

    // --- STATS ---
    uint256 public override totalAttestations;
    uint256 public override totalUsageUnits;

    // --- ERRORS ---
    error NotOwner();
    error NotAutonet();
    error InvalidScore();
    error KeyTooLong();
    error ValueTooLong();

    // --- MODIFIERS ---
    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    modifier onlyAutonet() {
        if (msg.sender != autonet) revert NotAutonet();
        _;
    }

    /**
     * @notice Create a new user contract
     * @param _owner Wallet that controls this user
     * @param _autonet Autonet contract this user belongs to
     * @param _standardsHash Hash of jurisdiction standards at signup
     */
    constructor(
        address _owner,
        address _autonet,
        bytes32 _standardsHash
    ) {
        owner = _owner;
        autonet = _autonet;
        createdAt = block.timestamp;
        jurisdictionStandardsHash = _standardsHash;
        agreedToStandardsAt = block.timestamp;
        alignmentScore = 0;
        alignmentUpdatedAt = 0;
    }

    // =========================================================================
    // PREFERENCES
    // =========================================================================

    /**
     * @notice Get a preference value
     * @param key Preference key
     * @return value Preference value (empty string if not set)
     */
    function getPreference(string calldata key) external view override returns (string memory) {
        return preferences[key];
    }

    /**
     * @notice Set a preference value
     * @param key Preference key (max 64 chars)
     * @param value Preference value (max 256 chars)
     */
    function setPreference(string calldata key, string calldata value) external override onlyOwner {
        if (bytes(key).length > 64) revert KeyTooLong();
        if (bytes(value).length > 256) revert ValueTooLong();

        // Track new keys
        if (!preferenceKeyExists[key]) {
            preferenceKeys.push(key);
            preferenceKeyExists[key] = true;
        }

        preferences[key] = value;
        emit PreferenceSet(key, value);
    }

    /**
     * @notice Get all preference keys
     * @return keys Array of all preference keys set
     */
    function getPreferenceKeys() external view override returns (string[] memory) {
        return preferenceKeys;
    }

    // =========================================================================
    // ALIGNMENT
    // =========================================================================

    /**
     * @notice Update alignment score
     * @param score New score (0-10000 basis points)
     *
     * This is a self-reported score. Future versions may require:
     * - ZK proofs of alignment
     * - Attestations from services
     * - Oracle verification
     */
    function updateAlignment(uint256 score) external override onlyOwner {
        if (score > 10000) revert InvalidScore();

        uint256 oldScore = alignmentScore;
        alignmentScore = score;
        alignmentUpdatedAt = block.timestamp;

        emit AlignmentUpdated(oldScore, score);
    }

    // =========================================================================
    // STATS (Called by Autonet contract)
    // =========================================================================

    /**
     * @notice Increment usage stats
     * @param units Number of usage units to add
     *
     * Called by Autonet.attestUsage() to track user activity.
     */
    function incrementStats(uint256 units) external override onlyAutonet {
        totalAttestations++;
        totalUsageUnits += units;

        emit StatsIncremented(totalAttestations, totalUsageUnits);
    }
}
