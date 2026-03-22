// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title AutonetMockRegistry
 * @notice Mock Registry contract for testing Autonet
 */
contract AutonetMockRegistry {
    mapping(string => string) private _registry;

    function editRegistry(string memory key, string memory value) external {
        _registry[key] = value;
    }

    function getRegistryValue(string memory key) external view returns (string memory) {
        return _registry[key];
    }

    // Allow receiving ETH (matches real Registry's payable fallback)
    receive() external payable {}
    fallback() external payable {}
}
