// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../interfaces/IInferenceProvider.sol";
import "../core/Project.sol";
import "../tokens/ATNToken.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title InferenceProviderBridge
 * @dev Bridges Autonet's Project.sol inference to Jurisdiction's IInferenceProvider interface.
 *
 *      This contract enables Autonet-trained models (deployed via setMatureModel) to be
 *      registered as decentralized inference providers in the On-Chain Jurisdiction
 *      economic layer, replacing centralized LLM providers.
 *
 *      Flow:
 *      1. Jurisdiction user calls requestInference(input, maxCredits)
 *      2. Bridge forwards to Project.requestInference(projectId, inputCid)
 *      3. Off-chain inference nodes process request (listening to InferenceRequested events)
 *      4. Nodes call submitResult(requestId, outputCid) when complete
 *      5. Jurisdiction user calls getResult(requestId) to retrieve output
 *
 *      Economic Integration:
 *      - ATN from Jurisdiction is used as payment (same token, cross-contract)
 *      - Inference fees flow to Project's PT holders as revenue
 *      - Creates incentive alignment: useful models → more inference → more rewards
 */
contract InferenceProviderBridge is IInferenceProvider, Ownable, ReentrancyGuard {
    Project public immutable project;
    ATNToken public immutable atnToken;
    uint256 public immutable projectId;

    /// @notice Authorized inference nodes that can submit results
    mapping(address => bool) public authorizedNodes;

    /// @notice Request data storage
    struct InferenceRequest {
        address requester;
        string inputCid;
        uint256 maxCredits;
        uint256 creditsUsed;
        bytes output;        // Output bytes (typically encoded CID or direct result)
        bool completed;
        uint256 timestamp;
    }

    mapping(bytes32 => InferenceRequest) public requests;
    uint256 public nextRequestNonce;

    /// @notice Timeout for requests (default 1 hour)
    uint256 public requestTimeout = 3600;

    event NodeAuthorized(address indexed node, bool authorized);
    event ResultSubmitted(bytes32 indexed requestId, address indexed node, string outputCid);
    event RequestTimedOut(bytes32 indexed requestId);

    modifier onlyAuthorizedNode() {
        require(authorizedNodes[msg.sender], "Not authorized node");
        _;
    }

    constructor(
        address _project,
        uint256 _projectId,
        address _owner
    ) Ownable(_owner) {
        project = Project(_project);
        projectId = _projectId;
        atnToken = project.atnToken();
    }

    // ============ Admin Functions ============

    /// @notice Authorize or revoke an inference node
    function setAuthorizedNode(address _node, bool _authorized) external onlyOwner {
        authorizedNodes[_node] = _authorized;
        emit NodeAuthorized(_node, _authorized);
    }

    /// @notice Set request timeout
    function setRequestTimeout(uint256 _timeout) external onlyOwner {
        requestTimeout = _timeout;
    }

    // ============ IInferenceProvider Implementation ============

    /// @notice Request inference from this provider
    /// @param input Encoded input (expected: abi.encode(string inputCid))
    /// @param maxCredits Maximum ATN willing to spend
    function requestInference(bytes calldata input, uint256 maxCredits)
        external
        override
        nonReentrant
        returns (bytes32 requestId)
    {
        // Decode input CID from bytes
        string memory inputCid = abi.decode(input, (string));

        // Generate unique request ID
        requestId = keccak256(abi.encodePacked(
            block.chainid,
            address(this),
            msg.sender,
            nextRequestNonce++,
            block.timestamp
        ));

        // Get effective price for requester
        uint256 fee = project.getEffectivePrice(projectId, msg.sender);
        require(fee <= maxCredits, "Price exceeds maxCredits");

        // Transfer ATN from requester to this contract (we'll forward to Project)
        require(atnToken.transferFrom(msg.sender, address(this), fee), "ATN transfer failed");

        // Approve Project to spend ATN
        atnToken.approve(address(project), fee);

        // Forward to Project.requestInference
        // Note: Project.requestInference returns uint256, we track with bytes32
        project.requestInference(projectId, inputCid);

        // Store request
        requests[requestId] = InferenceRequest({
            requester: msg.sender,
            inputCid: inputCid,
            maxCredits: maxCredits,
            creditsUsed: fee,
            output: "",
            completed: false,
            timestamp: block.timestamp
        });

        emit InferenceRequested(requestId, msg.sender, maxCredits);
    }

    /// @notice Check if result is ready
    function isResultReady(bytes32 requestId) external view override returns (bool ready) {
        InferenceRequest storage req = requests[requestId];
        return req.completed || _isTimedOut(requestId);
    }

    /// @notice Get inference result
    function getResult(bytes32 requestId)
        external
        view
        override
        returns (bytes memory output, uint256 creditsUsed)
    {
        InferenceRequest storage req = requests[requestId];

        if (_isTimedOut(requestId) && !req.completed) {
            // Return empty with 0 credits on timeout (refund logic would be separate)
            return ("", 0);
        }

        require(req.completed, "Result not ready");
        return (req.output, req.creditsUsed);
    }

    /// @notice Get current price per inference unit
    function getPricePerUnit() external view override returns (uint256 pricePerUnit) {
        (, uint256 price) = project.getMatureModel(projectId);
        return price;
    }

    // ============ Node Functions ============

    /// @notice Submit inference result (called by authorized nodes)
    /// @param requestId The request to fulfill
    /// @param outputCid IPFS CID of the result
    function submitResult(bytes32 requestId, string calldata outputCid)
        external
        onlyAuthorizedNode
    {
        InferenceRequest storage req = requests[requestId];
        require(req.timestamp > 0, "Request does not exist");
        require(!req.completed, "Already completed");
        require(!_isTimedOut(requestId), "Request timed out");

        req.output = abi.encode(outputCid);
        req.completed = true;

        emit ResultSubmitted(requestId, msg.sender, outputCid);
        emit InferenceCompleted(requestId, req.creditsUsed);
    }

    /// @notice Submit raw bytes result (for direct results, not CIDs)
    function submitRawResult(bytes32 requestId, bytes calldata output)
        external
        onlyAuthorizedNode
    {
        InferenceRequest storage req = requests[requestId];
        require(req.timestamp > 0, "Request does not exist");
        require(!req.completed, "Already completed");
        require(!_isTimedOut(requestId), "Request timed out");

        req.output = output;
        req.completed = true;

        emit InferenceCompleted(requestId, req.creditsUsed);
    }

    // ============ View Functions ============

    /// @notice Check if request has timed out
    function _isTimedOut(bytes32 requestId) internal view returns (bool) {
        InferenceRequest storage req = requests[requestId];
        return req.timestamp > 0 && block.timestamp > req.timestamp + requestTimeout;
    }

    /// @notice Get request details
    function getRequest(bytes32 requestId) external view returns (
        address requester,
        string memory inputCid,
        uint256 maxCredits,
        uint256 creditsUsed,
        bool completed,
        uint256 timestamp
    ) {
        InferenceRequest storage req = requests[requestId];
        return (
            req.requester,
            req.inputCid,
            req.maxCredits,
            req.creditsUsed,
            req.completed,
            req.timestamp
        );
    }

    /// @notice Get the model weights CID for this provider
    function getModelCid() external view returns (string memory) {
        (string memory weightsCid, ) = project.getMatureModel(projectId);
        return weightsCid;
    }
}
