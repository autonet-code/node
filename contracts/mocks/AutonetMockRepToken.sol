// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/**
 * @title AutonetMockRepToken
 * @notice Mock RepToken for testing Autonet governance token conversion
 * @dev Implements ERC20 for transferFrom which Autonet needs
 */
contract AutonetMockRepToken is ERC20 {
    address public admin;

    constructor(string memory name, string memory symbol) ERC20(name, symbol) {
        admin = msg.sender;
    }

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }

    function burn(address from, uint256 amount) external {
        _burn(from, amount);
    }

    // Match RepToken interface
    function getVotes(address account) external view returns (uint256) {
        return balanceOf(account);
    }
}
