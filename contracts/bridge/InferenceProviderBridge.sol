// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../interfaces/IInferenceProvider.sol";
import "../core/Project.sol";
import "../core/ParticipantStaking.sol";
import "../tokens/ATNToken.sol";
import "../utils/AutonetLib.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title InferenceProviderBridge
 * @dev Bridges Autonet's Project.sol inference to Jurisdiction's IInferenceProvider interface
 *      with a Burn-Mint Equilibrium (BME) payment flow.
 *
 * BME Payment Flow (Phase C Step 5):
 *   1. User calls requestInference(input, maxCredits)
 *   2. Bridge burns ATN from user immediately (no token holds/transfers)
 *   3. Burned amount is tracked in the internal burn pool
 *   4. Off-chain inference nodes process the request
 *   5. Probabilistic spot-check (VERIFICATION_RATE_BPS = 700 = 7%):
 *      - Spot-check triggered: payment staged, verifier reviews output
 *      - No spot-check (93%): immediate provider payout
 *   6. On completion, bridge mints ATN to provider from burn pool
 *      (requires this contract is an authorized minter on ATNToken)
 *   7. PROTOCOL_FEE_BPS is retained as jackpot pool for verification rewards
 *
 * Key design decisions:
 * - Immediate payment (not epoch-delayed like Bittensor)
 * - Providers must be staked INFERENCE_PROVIDER participants (stake-backed guarantee)
 * - Probabilistic verification deters cheating without full re-execution overhead
 * - Protocol fee funds jackpots that incentivize honest spot-checking
 */
contract InferenceProviderBridge is IInferenceProvider, Ownable, ReentrancyGuard {
    Project public immutable project;
    ATNToken public immutable atnToken;
    ParticipantStaking public immutable staking;
    uint256 public immutable projectId;

    // ============ BME Economics Parameters ============

    /// @notice Protocol fee retained from each payment (funds jackpot pool). 300 = 3%
    uint256 public protocolFeeBps = 300;

    /// @notice Probability of triggering a verification spot-check. 700 = 7%
    uint256 public verificationRateBps = 700;

    /// @notice Jackpot paid to honest verifiers who catch bad results
    uint256 public verificationJackpotAmount = 10 * 1e18;

    /// @notice Total ATN burned by users (burn pool accounting)
    uint256 public totalBurned;

    /// @notice Accumulated protocol fees available as jackpots
    uint256 public jackpotPool;

    /// @notice Nonce for pseudo-random spot-check selection
    uint256 private _verifyNonce;

    // ============ Request Storage ============

    struct InferenceRequest {
        address requester;
        address provider;          // Assigned provider (set when node claims request)
        string inputCid;
        uint256 burnedAmount;      // ATN burned by requester
        uint256 providerAmount;    // ATN to mint to provider (after protocol fee)
        bytes output;
        bool completed;
        bool pendingVerification;  // True if selected for spot-check
        uint256 timestamp;
    }

    mapping(bytes32 => InferenceRequest) public requests;
    uint256 public nextRequestNonce;

    /// @notice Authorized inference nodes (off-chain executors)
    mapping(address => bool) public authorizedNodes;

    /// @notice Timeout for requests (default 1 hour)
    uint256 public requestTimeout = 3600;

    // ============ Events ============

    event NodeAuthorized(address indexed node, bool authorized);
    event ResultSubmitted(bytes32 indexed requestId, address indexed node, address indexed provider, string outputCid);
    event ProviderPaid(bytes32 indexed requestId, address indexed provider, uint256 amount);
    event VerificationTriggered(bytes32 indexed requestId, address indexed provider, uint256 stagedAmount);
    event VerificationResolved(bytes32 indexed requestId, address indexed provider, bool honest, uint256 amountPaid);
    event RequestTimedOut(bytes32 indexed requestId, uint256 refundAttempted);
    event JackpotAwarded(address indexed catcher, uint256 amount);
    event ProtocolFeeConfigured(uint256 feeBps, uint256 verificationRateBps);

    // ============ Modifiers ============

    modifier onlyAuthorizedNode() {
        require(authorizedNodes[msg.sender], "Not authorized node");
        _;
    }

    modifier onlyStakedProvider(address _provider) {
        require(
            staking.isActiveParticipant(_provider, AutonetLib.ParticipantRole.INFERENCE_PROVIDER),
            "Provider not staked"
        );
        _;
    }

    // ============ Constructor ============

    constructor(
        address _project,
        uint256 _projectId,
        address _staking,
        address _owner
    ) Ownable(_owner) {
        project = Project(_project);
        projectId = _projectId;
        staking = ParticipantStaking(_staking);
        atnToken = project.atnToken();
    }

    // ============ Admin Functions ============

    function setAuthorizedNode(address _node, bool _authorized) external onlyOwner {
        authorizedNodes[_node] = _authorized;
        emit NodeAuthorized(_node, _authorized);
    }

    function setRequestTimeout(uint256 _timeout) external onlyOwner {
        requestTimeout = _timeout;
    }

    function setProtocolFee(uint256 _feeBps, uint256 _verificationRateBps) external onlyOwner {
        require(_feeBps <= 2000, "Fee too high");                    // Max 20%
        require(_verificationRateBps <= 1000, "Rate too high");      // Max 10%
        protocolFeeBps = _feeBps;
        verificationRateBps = _verificationRateBps;
        emit ProtocolFeeConfigured(_feeBps, _verificationRateBps);
    }

    function setVerificationJackpot(uint256 _amount) external onlyOwner {
        verificationJackpotAmount = _amount;
    }

    // ============ IInferenceProvider Implementation ============

    /**
     * @notice Request inference — burns ATN immediately (BME burn step).
     * @param input Encoded input (abi.encode(string inputCid))
     * @param maxCredits Maximum ATN willing to spend
     * @return requestId Unique ID for this inference request
     */
    function requestInference(bytes calldata input, uint256 maxCredits)
        external
        override
        nonReentrant
        returns (bytes32 requestId)
    {
        string memory inputCid = abi.decode(input, (string));

        requestId = keccak256(abi.encodePacked(
            block.chainid,
            address(this),
            msg.sender,
            nextRequestNonce++,
            block.timestamp
        ));

        // Get effective price
        uint256 fee = project.getEffectivePrice(projectId, msg.sender);
        require(fee <= maxCredits, "Price exceeds maxCredits");
        require(fee > 0, "Zero fee");

        // BME Step 1: Burn ATN from user immediately
        // No token holds — tokens are destroyed, provider gets fresh minted tokens
        atnToken.burnFrom(msg.sender, fee);
        totalBurned += fee;

        // Split: provider amount vs protocol fee (goes to jackpot pool)
        uint256 protocolFee = (fee * protocolFeeBps) / 10000;
        uint256 providerAmount = fee - protocolFee;
        jackpotPool += protocolFee;

        requests[requestId] = InferenceRequest({
            requester: msg.sender,
            provider: address(0),     // Assigned when node claims/submits
            inputCid: inputCid,
            burnedAmount: fee,
            providerAmount: providerAmount,
            output: "",
            completed: false,
            pendingVerification: false,
            timestamp: block.timestamp
        });

        emit InferenceRequested(requestId, msg.sender, maxCredits);
    }

    /// @notice Check if result is ready
    function isResultReady(bytes32 requestId) external view override returns (bool) {
        InferenceRequest storage req = requests[requestId];
        if (req.pendingVerification) return false;  // Blocked pending verification
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
            return ("", 0);
        }

        require(req.completed, "Result not ready");
        require(!req.pendingVerification, "Pending verification");
        return (req.output, req.burnedAmount);
    }

    /// @notice Get current price per inference unit
    function getPricePerUnit() external view override returns (uint256) {
        (, uint256 price) = project.getMatureModel(projectId);
        return price;
    }

    // ============ Node Functions (BME verify→mint steps) ============

    /**
     * @notice Submit inference result. Implements BME verify→mint step.
     *
     *         Probabilistic spot-check (verificationRateBps / 10000):
     *         - If spot-check triggered: payment staged, VerificationTriggered emitted.
     *           An authorized verifier must call resolveVerification() within timeout.
     *         - Otherwise: provider is paid immediately via ATN mint.
     *
     * @param requestId The request to fulfill
     * @param outputCid Content hash of the result
     * @param provider  The inference provider to pay (must be staked INFERENCE_PROVIDER)
     */
    function submitResult(
        bytes32 requestId,
        string calldata outputCid,
        address provider
    )
        external
        onlyAuthorizedNode
        onlyStakedProvider(provider)
    {
        InferenceRequest storage req = requests[requestId];
        require(req.timestamp > 0, "Request does not exist");
        require(!req.completed, "Already completed");
        require(!_isTimedOut(requestId), "Request timed out");

        req.output = abi.encode(outputCid);
        req.provider = provider;

        // Probabilistic spot-check: ~7% of requests undergo verification
        if (_shouldSpotCheck()) {
            req.pendingVerification = true;
            emit VerificationTriggered(requestId, provider, req.providerAmount);
            emit ResultSubmitted(requestId, msg.sender, provider, outputCid);
            // Payment is staged — resolveVerification() will finalize
        } else {
            // BME Step 2: Mint ATN to provider from burn pool
            req.completed = true;
            _mintToProvider(provider, req.providerAmount);
            emit ResultSubmitted(requestId, msg.sender, provider, outputCid);
            emit InferenceCompleted(requestId, req.burnedAmount);
            emit ProviderPaid(requestId, provider, req.providerAmount);
        }
    }

    /**
     * @notice Submit raw bytes result (for direct results, not CIDs)
     */
    function submitRawResult(
        bytes32 requestId,
        bytes calldata output,
        address provider
    )
        external
        onlyAuthorizedNode
        onlyStakedProvider(provider)
    {
        InferenceRequest storage req = requests[requestId];
        require(req.timestamp > 0, "Request does not exist");
        require(!req.completed, "Already completed");
        require(!_isTimedOut(requestId), "Request timed out");

        req.output = output;
        req.provider = provider;

        if (_shouldSpotCheck()) {
            req.pendingVerification = true;
            emit VerificationTriggered(requestId, provider, req.providerAmount);
        } else {
            req.completed = true;
            _mintToProvider(provider, req.providerAmount);
            emit InferenceCompleted(requestId, req.burnedAmount);
            emit ProviderPaid(requestId, provider, req.providerAmount);
        }
    }

    /**
     * @notice Resolve a spot-checked inference request.
     *
     *         Called by an authorized node after independently verifying the output.
     *         - honest=true:  Provider was correct → mint payment, award no jackpot
     *         - honest=false: Provider cheated → slash provider stake, award jackpot to verifier
     *
     * @param requestId The spot-checked request
     * @param honest    Whether the provider's result was correct
     */
    function resolveVerification(bytes32 requestId, bool honest)
        external
        onlyAuthorizedNode
    {
        InferenceRequest storage req = requests[requestId];
        require(req.pendingVerification, "Not pending verification");
        require(!req.completed, "Already completed");

        req.pendingVerification = false;
        req.completed = true;

        address provider = req.provider;

        if (honest) {
            // Provider was honest: pay them in full
            _mintToProvider(provider, req.providerAmount);
            emit VerificationResolved(requestId, provider, true, req.providerAmount);
            emit ProviderPaid(requestId, provider, req.providerAmount);
        } else {
            // Provider cheated: slash stake, award jackpot to verifier, burn provider amount
            staking.slash(provider, staking.minStakeAmount(AutonetLib.ParticipantRole.INFERENCE_PROVIDER), "Inference fraud: bad result submitted");
            // Provider amount returns to jackpot pool (not minted)
            jackpotPool += req.providerAmount;

            // Award jackpot to the verifier (msg.sender)
            if (jackpotPool >= verificationJackpotAmount) {
                jackpotPool -= verificationJackpotAmount;
                atnToken.mint(msg.sender, verificationJackpotAmount);
                emit JackpotAwarded(msg.sender, verificationJackpotAmount);
            }

            emit VerificationResolved(requestId, provider, false, 0);
        }

        emit InferenceCompleted(requestId, req.burnedAmount);
    }

    /**
     * @notice Handle a timed-out request.
     *         The burned ATN is already gone (BME: burn is unconditional).
     *         The provider amount returns to the jackpot pool for future users.
     */
    function expireRequest(bytes32 requestId) external {
        InferenceRequest storage req = requests[requestId];
        require(req.timestamp > 0, "Request does not exist");
        require(!req.completed, "Already completed");
        require(_isTimedOut(requestId), "Not timed out");

        req.completed = true;
        jackpotPool += req.providerAmount;  // Recycle to jackpot pool

        emit RequestTimedOut(requestId, req.providerAmount);
    }

    // ============ BME Internal Helpers ============

    /**
     * @dev Mint ATN to provider from burn pool (BME mint step).
     *      Requires this contract is an authorized minter on ATNToken.
     */
    function _mintToProvider(address provider, uint256 amount) internal {
        if (amount > 0) {
            atnToken.mint(provider, amount);
        }
    }

    /**
     * @dev Probabilistic spot-check selector.
     *      Returns true with probability = verificationRateBps / 10000.
     *      Uses block data + nonce for pseudo-randomness.
     */
    function _shouldSpotCheck() internal returns (bool) {
        if (verificationRateBps == 0) return false;
        _verifyNonce++;
        uint256 rand = uint256(keccak256(abi.encodePacked(
            block.timestamp,
            block.prevrandao,
            msg.sender,
            _verifyNonce
        ))) % 10000;
        return rand < verificationRateBps;
    }

    function _isTimedOut(bytes32 requestId) internal view returns (bool) {
        InferenceRequest storage req = requests[requestId];
        return req.timestamp > 0 && block.timestamp > req.timestamp + requestTimeout;
    }

    // ============ View Functions ============

    function getRequest(bytes32 requestId) external view returns (
        address requester,
        address provider,
        string memory inputCid,
        uint256 burnedAmount,
        uint256 providerAmount,
        bool completed,
        bool pendingVerification,
        uint256 timestamp
    ) {
        InferenceRequest storage req = requests[requestId];
        return (
            req.requester,
            req.provider,
            req.inputCid,
            req.burnedAmount,
            req.providerAmount,
            req.completed,
            req.pendingVerification,
            req.timestamp
        );
    }

    function getModelCid() external view returns (string memory) {
        (string memory weightsCid, ) = project.getMatureModel(projectId);
        return weightsCid;
    }

    function getBMEStats() external view returns (
        uint256 burned,
        uint256 jackpot,
        uint256 feeBps,
        uint256 verifyBps
    ) {
        return (totalBurned, jackpotPool, protocolFeeBps, verificationRateBps);
    }
}
