// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./InferenceProviderBridge.sol";
import "../core/Project.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title InferenceProviderFactory
 * @dev Factory for deploying InferenceProviderBridge instances per Autonet project.
 *
 *      When a project reaches DEPLOYED status (setMatureModel called), the project
 *      founder or governance can deploy a bridge to make the model available as
 *      an IInferenceProvider for the Jurisdiction economic layer.
 *
 *      Each project gets exactly one bridge. The bridge address is registered
 *      with Jurisdiction's Autonet.sol as a INFERENCE_PROVIDER service.
 */
contract InferenceProviderFactory is Ownable {
    Project public immutable project;

    /// @notice Mapping from project ID to its bridge (if deployed)
    mapping(uint256 => address) public projectBridges;

    /// @notice All deployed bridges
    address[] public allBridges;

    event BridgeDeployed(uint256 indexed projectId, address indexed bridge, address indexed deployer);

    constructor(address _project, address _owner) Ownable(_owner) {
        project = Project(_project);
    }

    /// @notice Deploy a bridge for a project
    /// @param projectId The Autonet project ID
    /// @param bridgeOwner Who should own the bridge (typically project founder or DAO)
    /// @return bridge The deployed bridge address
    function deployBridge(uint256 projectId, address bridgeOwner)
        external
        returns (address bridge)
    {
        require(projectBridges[projectId] == address(0), "Bridge already deployed");

        // Verify project exists and is deployed
        (string memory weightsCid, ) = project.getMatureModel(projectId);
        require(bytes(weightsCid).length > 0, "Project has no mature model");

        // Deploy bridge
        InferenceProviderBridge newBridge = new InferenceProviderBridge(
            address(project),
            projectId,
            bridgeOwner
        );

        bridge = address(newBridge);
        projectBridges[projectId] = bridge;
        allBridges.push(bridge);

        emit BridgeDeployed(projectId, bridge, msg.sender);
    }

    /// @notice Get the number of deployed bridges
    function getBridgeCount() external view returns (uint256) {
        return allBridges.length;
    }

    /// @notice Check if a project has a bridge
    function hasBridge(uint256 projectId) external view returns (bool) {
        return projectBridges[projectId] != address(0);
    }
}
