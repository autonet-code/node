// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {RPB} from "./RPB.sol";
import {Registry} from "./Registry.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title RPBFactory
 * @notice Registry for RPB instances linked to jurisdictions.
 *
 * RPB contracts are deployed directly (e.g. via Remix) and then registered
 * here. The factory validates that the RPB points to a real jurisdiction.
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
     * @notice Register a pre-deployed RPB for a jurisdiction.
     * @param _rpb Address of the already-deployed RPB contract
     * @return rpb Address of the registered RPB
     */
    function registerRPB(
        address _rpb
    ) external returns (address rpb) {
        RPB rpbContract = RPB(_rpb);
        address registryAddr = address(rpbContract.registry());

        rpb = _rpb;
        deployedRPBs.push(rpb);
        rpbsByDeployer[msg.sender].push(rpb);
        rpbByRegistry[registryAddr] = rpb;

        emit RPBDeployed(rpb, registryAddr, msg.sender);
    }

    /**
     * @notice Deploy and register an RPB for a jurisdiction.
     * @param _registry The jurisdiction's Registry contract address
     * @return rpb Address of the deployed RPB contract
     */
    function createRPB(
        address _registry
    ) external returns (address rpb) {
        RPB newRPB = new RPB(_registry);
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
