// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IJurisdictionData Interface
 * @dev Defines read-only functions required by other contracts to query epoch data from the Jurisdiction contract.
 */
interface IJurisdictionData {
    /**
     * @dev Returns the start timestamp of a specific passive income epoch.
     * @param epochId The ID of the epoch to query.
     * @return The unix timestamp (as uint48) when the epoch started.
     */
    function getPassiveIncomeEpochStart(uint256 epochId) external view returns (uint48);

    /**
     * @dev Returns the start timestamp of a specific delegate reward epoch.
     * @param epochId The ID of the epoch to query.
     * @return The unix timestamp (as uint48) when the epoch started.
     */
    function getDelegateRewardEpochStart(uint256 epochId) external view returns (uint48);
}
// IJurisdictionData.sol