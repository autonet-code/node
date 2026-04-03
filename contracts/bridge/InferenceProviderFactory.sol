// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./InferenceProviderBridge.sol";
import "../core/RPB.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title InferenceProviderFactory
 * @dev Factory for deploying InferenceProviderBridge instances per RPB jurisdiction.
 *
 *      When an RPB's model is deployed (setMatureModel called), the RPB owner
 *      or governance can deploy a bridge to make the model available as
 *      an IInferenceProvider for the Jurisdiction economic layer.
 *
 *      Each RPB gets exactly one bridge. The bridge address is registered
 *      with Jurisdiction's Autonet.sol as an INFERENCE_PROVIDER service.
 */
contract InferenceProviderFactory is Ownable {
    address public immutable staking;

    /// @notice Mapping from RPB address to its bridge (if deployed)
    mapping(address => address) public rpbBridges;

    /// @notice All deployed bridges
    address[] public allBridges;

    event BridgeDeployed(address indexed rpb, address indexed bridge, address indexed deployer);

    constructor(address _staking, address _owner) Ownable(_owner) {
        staking = _staking;
    }

    /// @notice Deploy a bridge for an RPB jurisdiction
    /// @param rpbAddress The RPB contract address
    /// @param bridgeOwner Who should own the bridge (typically RPB owner or DAO)
    /// @return bridge The deployed bridge address
    function deployBridge(address rpbAddress, address bridgeOwner)
        external
        returns (address bridge)
    {
        require(rpbBridges[rpbAddress] == address(0), "Bridge already deployed");

        // Verify RPB has a deployed model
        (string memory weightsCid, ) = RPB(rpbAddress).getMatureModel();
        require(bytes(weightsCid).length > 0, "RPB has no mature model");

        // Deploy bridge
        InferenceProviderBridge newBridge = new InferenceProviderBridge(
            rpbAddress,
            staking,
            bridgeOwner
        );

        bridge = address(newBridge);
        rpbBridges[rpbAddress] = bridge;
        allBridges.push(bridge);

        emit BridgeDeployed(rpbAddress, bridge, msg.sender);
    }

    /// @notice Get the number of deployed bridges
    function getBridgeCount() external view returns (uint256) {
        return allBridges.length;
    }

    /// @notice Check if an RPB has a bridge
    function hasBridge(address rpbAddress) external view returns (bool) {
        return rpbBridges[rpbAddress] != address(0);
    }
}
