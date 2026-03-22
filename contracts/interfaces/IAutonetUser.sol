// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IAutonetUser
 * @notice Interface for user identity contracts in the Autonet system.
 *
 * Each user can deploy a UserContract to establish on-chain identity:
 * - Proves wallet ownership
 * - Records agreement to jurisdiction standards
 * - Exposes public preferences for dApps
 * - Tracks optional alignment score (privacy-preserving)
 *
 * This separates on-chain verifiable data from off-chain private data
 * stored in Chevin/Firestore.
 */
interface IAutonetUser {
    // --- IDENTITY ---

    /// @notice Get the wallet that owns this user contract
    function owner() external view returns (address);

    /// @notice Get the Autonet contract this user belongs to
    function autonet() external view returns (address);

    /// @notice Get when this user contract was created
    function createdAt() external view returns (uint256);

    // --- STANDARDS AGREEMENT ---

    /// @notice Get the hash of jurisdiction standards at signup
    /// @dev Hash of the registry snapshot when user joined
    function jurisdictionStandardsHash() external view returns (bytes32);

    /// @notice Get when the user agreed to standards
    function agreedToStandardsAt() external view returns (uint256);

    // --- PUBLIC PREFERENCES ---
    // dApps can read these to customize experience

    /// @notice Get a user's public preference
    /// @param key The preference key (e.g., "theme", "notifications")
    function getPreference(string calldata key) external view returns (string memory);

    /// @notice Set a public preference (only owner can call)
    /// @param key The preference key
    /// @param value The preference value
    function setPreference(string calldata key, string calldata value) external;

    /// @notice Get all preference keys set by user
    function getPreferenceKeys() external view returns (string[] memory);

    // --- ALIGNMENT (Optional, Privacy-Preserving) ---

    /// @notice Get user's declared alignment score (0-10000 basis points)
    /// @dev 10000 = 100% alignment with declared standards
    function alignmentScore() external view returns (uint256);

    /// @notice Get when alignment was last updated
    function alignmentUpdatedAt() external view returns (uint256);

    /// @notice Update alignment score (only owner can call)
    /// @param score New score in basis points (0-10000)
    function updateAlignment(uint256 score) external;

    // --- USAGE STATS (Aggregated from Autonet) ---

    /// @notice Get total attestations made by this user
    function totalAttestations() external view returns (uint256);

    /// @notice Get total usage units attested
    function totalUsageUnits() external view returns (uint256);

    /// @notice Increment stats (only callable by Autonet contract)
    function incrementStats(uint256 units) external;

    // --- EVENTS ---

    event PreferenceSet(string key, string value);
    event AlignmentUpdated(uint256 oldScore, uint256 newScore);
    event StatsIncremented(uint256 newTotalAttestations, uint256 newTotalUnits);
}
