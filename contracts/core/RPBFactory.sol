// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {RPB} from "./RPB.sol";
import {Registry} from "./Registry.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title RPBFactory
 * @notice Factory for deploying RPB instances linked to jurisdictions.
 *
 * Anyone can deploy an RPB for a jurisdiction. The factory validates that the
 * target Registry belongs to a real jurisdiction (has a jurisdictionAddress set).
 * The RPB is live immediately but not endorsed by the jurisdiction until the
 * DAO passes a proposal to add it to the Registry.
 */
contract RPBFactory {

    address[] public deployedRPBs;
    mapping(address => address[]) public rpbsByDeployer;
    mapping(address => address) public rpbByRegistry;  // registry -> RPB (latest)

    event RPBDeployed(
        address indexed rpb,
        address indexed registry,
        address indexed deployer
    );

    /**
     * @notice Deploy an RPB for a jurisdiction.
     * @dev The RPB accepts any token with a parity entry in the jurisdiction's
     *      Registry. No specific payment token needed at deploy time.
     * @param _registry The jurisdiction's Registry contract address
     * @return rpb Address of the deployed RPB contract
     */
    function createRPB(
        address _registry,
        address _atnToken
    ) external returns (address rpb) {
        // RPB constructor validates the jurisdiction
        RPB newRPB = new RPB(_registry, _atnToken);

        // Transfer ownership to the deployer
        newRPB.transferOwnership(msg.sender);

        rpb = address(newRPB);
        deployedRPBs.push(rpb);
        rpbsByDeployer[msg.sender].push(rpb);
        rpbByRegistry[_registry] = rpb;

        emit RPBDeployed(rpb, _registry, msg.sender);
    }

    function getRPBCount() external view returns (uint256) {
        return deployedRPBs.length;
    }

    function getRPBsByDeployer(address deployer) external view returns (address[] memory) {
        return rpbsByDeployer[deployer];
    }
}
