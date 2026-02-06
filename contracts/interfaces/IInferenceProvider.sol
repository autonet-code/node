// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IInferenceProvider
 * @dev Interface for decentralized inference services.
 *      Compatible with On-Chain Jurisdiction's Autonet.sol service registry.
 *
 *      This interface enables Autonet PoI-trained models to be registered
 *      as inference providers in the Jurisdiction economic layer.
 */
interface IInferenceProvider {
    /// @notice Request inference, burning ATN as payment
    /// @param input The inference request (model-specific encoding, typically IPFS CID)
    /// @param maxCredits Maximum ATN to spend (will burn actual cost)
    /// @return requestId Unique ID for this inference request
    function requestInference(bytes calldata input, uint256 maxCredits) external returns (bytes32 requestId);

    /// @notice Check if inference result is ready
    /// @param requestId The request ID from requestInference
    /// @return ready True if result is available
    function isResultReady(bytes32 requestId) external view returns (bool ready);

    /// @notice Get inference result (reverts if not ready)
    /// @param requestId The request ID from requestInference
    /// @return output The inference output
    /// @return creditsUsed Actual ATN burned for this request
    function getResult(bytes32 requestId) external view returns (bytes memory output, uint256 creditsUsed);

    /// @notice Get current price per inference unit
    /// @return pricePerUnit ATN cost per unit (provider-defined unit)
    function getPricePerUnit() external view returns (uint256 pricePerUnit);

    /// @notice Emitted when inference is requested
    event InferenceRequested(bytes32 indexed requestId, address indexed requester, uint256 maxCredits);

    /// @notice Emitted when inference completes
    event InferenceCompleted(bytes32 indexed requestId, uint256 creditsUsed);
}
