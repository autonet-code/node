// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Clones} from "@openzeppelin/contracts/proxy/Clones.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Autonet} from "./Autonet.sol";

/**
 * @title AutonetFactory
 * @notice Factory for deploying Autonet instances attached to Jurisdictions.
 *
 * Uses the clone pattern to minimize deployment cost. Each user deploys
 * their own Autonet instance linked to a specific Jurisdiction.
 *
 * Deployment flow:
 * 1. Admin deploys AutonetFactory with Autonet implementation
 * 2. User calls createAutonet() with their Jurisdiction's addresses
 * 3. Factory clones implementation and initializes
 * 4. User becomes admin of their Autonet instance
 */
contract AutonetFactory {
    // --- STATE ---
    address public immutable implementation;
    address[] public deployedAutonets;
    mapping(address => address[]) public autonetsByDeployer;

    // Track which Economy contracts already have an Autonet
    // (optional enforcement - can be disabled)
    mapping(address => address) public autonetForEconomy;
    bool public enforceOnePerEconomy;

    // --- EVENTS ---
    event AutonetCreated(
        address indexed autonet,
        address indexed deployer,
        address indexed economy,
        string name,
        string symbol
    );
    event EnforcementChanged(bool enforced);

    // --- ERRORS ---
    error AlreadyDeployedForEconomy(address economy, address existing);
    error ZeroAddress();
    error InitializationFailed();
    error DeployerNotJurisdictionMember();

    /**
     * @notice Deploy factory with Autonet implementation
     * @param _implementation Address of deployed Autonet to clone
     * @param _enforceOnePerEconomy If true, only one Autonet per Economy allowed
     */
    constructor(address _implementation, bool _enforceOnePerEconomy) {
        if (_implementation == address(0)) revert ZeroAddress();
        implementation = _implementation;
        enforceOnePerEconomy = _enforceOnePerEconomy;
    }

    /**
     * @notice Create a new Autonet instance for a Jurisdiction
     * @param economyAddress The Economy contract of the Jurisdiction
     * @param registryAddress The Registry contract
     * @param repTokenAddress The governance RepToken
     * @param timelockAddress The Timelock for governance
     * @param name Token name (e.g., "Autonet Credits - Literate")
     * @param symbol Token symbol (e.g., "ATN-LTR")
     * @return autonet Address of the new Autonet instance
     */
    function createAutonet(
        address economyAddress,
        address payable registryAddress,
        address repTokenAddress,
        address timelockAddress,
        string calldata name,
        string calldata symbol
    ) external returns (address autonet) {
        // Validate inputs
        if (economyAddress == address(0)) revert ZeroAddress();
        if (registryAddress == address(0)) revert ZeroAddress();
        if (repTokenAddress == address(0)) revert ZeroAddress();
        if (timelockAddress == address(0)) revert ZeroAddress();

        // Verify deployer is a jurisdiction member (has RepToken balance)
        if (IERC20(repTokenAddress).balanceOf(msg.sender) == 0) {
            revert DeployerNotJurisdictionMember();
        }

        // Check if already deployed for this Economy (if enforced)
        if (enforceOnePerEconomy && autonetForEconomy[economyAddress] != address(0)) {
            revert AlreadyDeployedForEconomy(economyAddress, autonetForEconomy[economyAddress]);
        }

        // Deploy full contract (not a clone)
        // Clones don't run constructors, so we deploy fresh instances
        // This costs more gas but ensures proper initialization
        Autonet newAutonet = new Autonet(
            economyAddress,
            registryAddress,
            repTokenAddress,
            timelockAddress,
            name,
            symbol
        );

        autonet = address(newAutonet);

        // Transfer admin to deployer
        newAutonet.transferAdmin(msg.sender);

        // Track deployment
        deployedAutonets.push(autonet);
        autonetsByDeployer[msg.sender].push(autonet);
        if (enforceOnePerEconomy) {
            autonetForEconomy[economyAddress] = autonet;
        }

        emit AutonetCreated(autonet, msg.sender, economyAddress, name, symbol);
    }

    /**
     * @notice Get all Autonets deployed by an address
     */
    function getAutonetsByDeployer(address deployer) external view returns (address[] memory) {
        return autonetsByDeployer[deployer];
    }

    /**
     * @notice Get total number of deployed Autonets
     */
    function getDeployedCount() external view returns (uint256) {
        return deployedAutonets.length;
    }

    /**
     * @notice Get Autonet for a specific Economy (if enforced)
     */
    function getAutonetForEconomy(address economy) external view returns (address) {
        return autonetForEconomy[economy];
    }

    /**
     * @notice Toggle one-per-economy enforcement (admin only - deployer)
     * @dev In production, this should be governance-controlled
     */
    function setEnforcement(bool enforced) external {
        // For simplicity, anyone can toggle. In production, add access control.
        enforceOnePerEconomy = enforced;
        emit EnforcementChanged(enforced);
    }
}
