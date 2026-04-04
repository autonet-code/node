// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Registry} from "../core/Registry.sol";

/**
 * @title EvolutionProposal
 * @notice Open-ended evolution mechanism for the Autonet network.
 *
 * Anyone can propose improvements (architectural changes, new paradigms,
 * hyperparameter tweaks, etc.) by submitting a CID + staking ATN.
 *
 * Lifecycle:  Proposed -> Evaluating -> Trial -> Adopted / Rejected
 *
 * Proposals are evaluated by the Recursive Principled Body (RPB):
 * nodes load the constitutional prompt from Registry, evaluate the
 * proposal through an AI provider, and submit structured recommendations
 * on-chain. Yuma-style consensus resolves non-deterministic outputs.
 *
 * Trial funding comes from network budget. Adoption rewards the proposer
 * and trial participants. Rejection refunds stake minus processing fee.
 *
 * Contribution types (diagnosis, proposal, validation, compute) carry
 * different reward weights.
 */
contract EvolutionProposal is ReentrancyGuard {

    // =========================================================================
    // ENUMS & STRUCTS
    // =========================================================================

    enum ProposalStatus {
        Proposed,      // Submitted, awaiting RPB evaluation
        Evaluating,    // RPB nodes are evaluating
        Trial,         // Approved for trial, funded
        Adopted,       // Trial succeeded, rewards distributed
        Rejected       // RPB/DAO rejected, stake refunded minus fee
    }

    enum ContributionType {
        Compute,       // Base training work
        Diagnosis,     // Identifying problems / capability gaps
        Proposal,      // Proposing solutions (highest on adoption)
        Validation     // Evaluating / validating proposals
    }

    struct Proposal {
        uint256 id;
        address proposer;
        string contentCid;          // CID pointing to description + spec
        uint256 stakeAmount;        // ATN staked by proposer
        ProposalStatus status;
        uint256 createdAt;
        uint256 statusChangedAt;
        uint256 trialBudget;        // ATN allocated for trial
        uint256 trialSpent;         // ATN spent during trial
        uint256 adoptionRewardPool; // Snapshot of remaining budget at adoption
        uint256 evaluationCount;    // Number of RPB evaluations received
        int256  evaluationScore;    // Net score from RPB evaluations
        uint256 daoProposalId;      // HomebaseDAO proposal ID (0 if none)
    }

    struct RPBEvaluation {
        address evaluator;
        bool approve;               // True = recommend adoption
        uint256 confidence;         // 0-10000 bps
        string reasonCid;           // CID pointing to structured recommendation
        uint256 timestamp;
    }

    // Contribution spectrum reward weights (in basis points, 10000 = 1x)
    struct ContributionWeights {
        uint256 compute;            // Base rate
        uint256 diagnosis;          // Problem identification
        uint256 proposal;           // Solution proposal (highest)
        uint256 validation;         // Evaluation work
    }

    // Track participant contributions during trials
    struct TrialParticipant {
        address participant;
        ContributionType contributionType;
        uint256 units;              // Work units contributed
        bool rewardClaimed;
    }

    // =========================================================================
    // STATE
    // =========================================================================

    address public admin;
    address public immutable rpbTokenAddress;
    address payable public immutable registryAddress;
    address public immutable timelockAddress;

    // Proposal storage
    uint256 public nextProposalId = 1;
    mapping(uint256 => Proposal) public proposals;
    uint256[] public proposalIds;

    // RPB evaluations per proposal
    mapping(uint256 => RPBEvaluation[]) public evaluations;
    mapping(uint256 => mapping(address => bool)) public hasEvaluated;

    // Trial participants per proposal
    mapping(uint256 => TrialParticipant[]) public trialParticipants;
    mapping(uint256 => mapping(address => bool)) public isTrialParticipant;

    // Configuration
    uint256 public minStake = 100 * 1e18;              // 100 ATN minimum stake
    uint256 public processingFeeBps = 500;              // 5% processing fee on rejection
    uint256 public evaluationQuorum = 3;                // Minimum RPB evaluations needed
    uint256 public approvalThresholdBps = 6000;         // 60% approval needed
    uint256 public maxTrialBudgetBps = 500;             // 5% of epoch budget for trial
    uint256 public evaluationPeriod = 3 days;           // Time for RPB evaluation

    // Contribution spectrum weights (defaults)
    ContributionWeights public contributionWeights = ContributionWeights({
        compute: 10000,     // 1x base
        diagnosis: 15000,   // 1.5x
        proposal: 30000,    // 3x (highest on adoption)
        validation: 12000   // 1.2x
    });

    // Who can submit RPB evaluations (nodes that run the evaluator)
    mapping(address => bool) public rpbEvaluators;

    // RPB prompt version tracking
    uint256 public currentPromptVersion;

    // =========================================================================
    // EVENTS
    // =========================================================================

    event ProposalSubmitted(uint256 indexed proposalId, address indexed proposer, string contentCid, uint256 stakeAmount);
    event ProposalStatusChanged(uint256 indexed proposalId, ProposalStatus oldStatus, ProposalStatus newStatus);
    event RPBEvaluationSubmitted(uint256 indexed proposalId, address indexed evaluator, bool approve, uint256 confidence);
    event TrialFunded(uint256 indexed proposalId, uint256 budget);
    event TrialParticipantAdded(uint256 indexed proposalId, address indexed participant, ContributionType contributionType);
    event ProposalAdopted(uint256 indexed proposalId, uint256 totalReward);
    event ProposalRejected(uint256 indexed proposalId, uint256 stakeRefunded, uint256 feeTaken);
    event ParticipantRewardClaimed(uint256 indexed proposalId, address indexed participant, uint256 amount);
    event RPBEvaluatorSet(address indexed evaluator, bool enabled);
    event ContributionWeightsUpdated(uint256 compute, uint256 diagnosis, uint256 proposal, uint256 validation);
    event RPBPromptUpdated(uint256 version, string promptCid);

    // =========================================================================
    // ERRORS
    // =========================================================================

    error NotAuthorized();
    error InvalidProposal();
    error ProposalNotFound();
    error InvalidStatus();
    error InsufficientStake();
    error AlreadyEvaluated();
    error QuorumNotReached();
    error EvaluationPeriodActive();
    error EvaluationPeriodExpired();
    error AlreadyClaimed();
    error NothingToClaim();
    error ZeroAddress();
    error InvalidAmount();

    // =========================================================================
    // MODIFIERS
    // =========================================================================

    modifier onlyAdmin() {
        if (msg.sender != admin && msg.sender != timelockAddress) revert NotAuthorized();
        _;
    }

    modifier onlyTimelock() {
        if (msg.sender != timelockAddress) revert NotAuthorized();
        _;
    }

    modifier onlyRPBEvaluator() {
        if (!rpbEvaluators[msg.sender] && msg.sender != admin) revert NotAuthorized();
        _;
    }

    // =========================================================================
    // CONSTRUCTOR
    // =========================================================================

    constructor(
        address _rpbToken,
        address payable _registry,
        address _timelock
    ) {
        if (_rpbToken == address(0)) revert ZeroAddress();
        if (_registry == address(0)) revert ZeroAddress();
        if (_timelock == address(0)) revert ZeroAddress();

        rpbTokenAddress = _rpbToken;
        registryAddress = _registry;
        timelockAddress = _timelock;
        admin = msg.sender;
    }

    // =========================================================================
    // PROPOSAL SUBMISSION
    // =========================================================================

    /**
     * @notice Submit an evolution proposal by staking ATN.
     * @param contentCid CID pointing to the proposal description + spec
     * @param stakeAmount ATN to stake (must be >= minStake)
     * @return proposalId The new proposal's ID
     */
    function submitProposal(
        string calldata contentCid,
        uint256 stakeAmount
    ) external nonReentrant returns (uint256 proposalId) {
        if (bytes(contentCid).length == 0) revert InvalidProposal();
        if (stakeAmount < minStake) revert InsufficientStake();

        // Transfer stake from proposer
        bool transferred = IERC20(rpbTokenAddress).transferFrom(
            msg.sender, address(this), stakeAmount
        );
        require(transferred, "EvolutionProposal: ATN transfer failed");

        proposalId = nextProposalId++;

        proposals[proposalId] = Proposal({
            id: proposalId,
            proposer: msg.sender,
            contentCid: contentCid,
            stakeAmount: stakeAmount,
            status: ProposalStatus.Proposed,
            createdAt: block.timestamp,
            statusChangedAt: block.timestamp,
            trialBudget: 0,
            trialSpent: 0,
            adoptionRewardPool: 0,
            evaluationCount: 0,
            evaluationScore: 0,
            daoProposalId: 0
        });

        proposalIds.push(proposalId);

        emit ProposalSubmitted(proposalId, msg.sender, contentCid, stakeAmount);
    }

    // =========================================================================
    // RPB EVALUATION
    // =========================================================================

    /**
     * @notice Begin RPB evaluation of a proposal.
     * @dev Moves proposal from Proposed to Evaluating. Admin or automatic.
     */
    function beginEvaluation(uint256 proposalId) external onlyAdmin {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        if (prop.status != ProposalStatus.Proposed) revert InvalidStatus();

        ProposalStatus oldStatus = prop.status;
        prop.status = ProposalStatus.Evaluating;
        prop.statusChangedAt = block.timestamp;

        emit ProposalStatusChanged(proposalId, oldStatus, ProposalStatus.Evaluating);
    }

    /**
     * @notice Submit an RPB evaluation for a proposal.
     * @dev Each evaluator can only evaluate once per proposal.
     * @param proposalId The proposal to evaluate
     * @param approve Whether to recommend adoption
     * @param confidence Confidence level (0-10000 bps)
     * @param reasonCid CID pointing to structured recommendation
     */
    function submitEvaluation(
        uint256 proposalId,
        bool approve,
        uint256 confidence,
        string calldata reasonCid
    ) external onlyRPBEvaluator {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        if (prop.status != ProposalStatus.Evaluating) revert InvalidStatus();
        if (hasEvaluated[proposalId][msg.sender]) revert AlreadyEvaluated();
        if (confidence > 10000) revert InvalidAmount();

        hasEvaluated[proposalId][msg.sender] = true;

        evaluations[proposalId].push(RPBEvaluation({
            evaluator: msg.sender,
            approve: approve,
            confidence: confidence,
            reasonCid: reasonCid,
            timestamp: block.timestamp
        }));

        prop.evaluationCount++;

        // Weighted score: +confidence for approve, -confidence for reject
        if (approve) {
            prop.evaluationScore += int256(confidence);
        } else {
            prop.evaluationScore -= int256(confidence);
        }

        emit RPBEvaluationSubmitted(proposalId, msg.sender, approve, confidence);
    }

    /**
     * @notice Resolve evaluation: approve for trial or reject.
     * @dev Requires quorum reached and evaluation period elapsed.
     *      Can be called by anyone (permissionless resolution).
     */
    function resolveEvaluation(uint256 proposalId) external {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        if (prop.status != ProposalStatus.Evaluating) revert InvalidStatus();
        if (prop.evaluationCount < evaluationQuorum) revert QuorumNotReached();

        // Check if evaluation period has elapsed
        if (block.timestamp < prop.statusChangedAt + evaluationPeriod) {
            revert EvaluationPeriodActive();
        }

        // Calculate approval percentage from weighted scores
        // Sum all confidence values, compute approval ratio
        RPBEvaluation[] storage evals = evaluations[proposalId];
        uint256 totalConfidence = 0;
        uint256 approvalConfidence = 0;

        for (uint256 i = 0; i < evals.length; i++) {
            totalConfidence += evals[i].confidence;
            if (evals[i].approve) {
                approvalConfidence += evals[i].confidence;
            }
        }

        ProposalStatus oldStatus = prop.status;

        if (totalConfidence > 0 &&
            (approvalConfidence * 10000) / totalConfidence >= approvalThresholdBps
        ) {
            // Approved for trial
            prop.status = ProposalStatus.Trial;
            prop.statusChangedAt = block.timestamp;
            emit ProposalStatusChanged(proposalId, oldStatus, ProposalStatus.Trial);
        } else {
            // Rejected
            _rejectProposal(proposalId);
        }
    }

    // =========================================================================
    // TRIAL MANAGEMENT
    // =========================================================================

    /**
     * @notice Fund a trial from network budget.
     * @param proposalId The proposal in Trial status
     * @param budget ATN to allocate for the trial
     */
    function fundTrial(uint256 proposalId, uint256 budget) external onlyAdmin {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        if (prop.status != ProposalStatus.Trial) revert InvalidStatus();
        if (budget == 0) revert InvalidAmount();

        // Transfer trial budget from admin/treasury
        bool transferred = IERC20(rpbTokenAddress).transferFrom(
            msg.sender, address(this), budget
        );
        require(transferred, "EvolutionProposal: Trial funding transfer failed");

        prop.trialBudget += budget;

        emit TrialFunded(proposalId, budget);
    }

    /**
     * @notice Record a trial participant's contribution.
     * @param proposalId The proposal in Trial status
     * @param participant The contributor address
     * @param contributionType Type of contribution
     * @param units Work units contributed
     */
    function recordTrialContribution(
        uint256 proposalId,
        address participant,
        ContributionType contributionType,
        uint256 units
    ) external onlyAdmin {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        if (prop.status != ProposalStatus.Trial) revert InvalidStatus();
        if (participant == address(0)) revert ZeroAddress();
        if (units == 0) revert InvalidAmount();

        trialParticipants[proposalId].push(TrialParticipant({
            participant: participant,
            contributionType: contributionType,
            units: units,
            rewardClaimed: false
        }));

        if (!isTrialParticipant[proposalId][participant]) {
            isTrialParticipant[proposalId][participant] = true;
        }

        emit TrialParticipantAdded(proposalId, participant, contributionType);
    }

    // =========================================================================
    // ADOPTION / REJECTION
    // =========================================================================

    /**
     * @notice Adopt a proposal after successful trial.
     * @dev Distributes rewards to proposer and trial participants.
     *      Can be triggered by admin or after DAO vote.
     */
    function adoptProposal(uint256 proposalId) external onlyAdmin {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        if (prop.status != ProposalStatus.Trial) revert InvalidStatus();

        ProposalStatus oldStatus = prop.status;
        prop.status = ProposalStatus.Adopted;
        prop.statusChangedAt = block.timestamp;

        // Snapshot the reward pool at adoption time
        prop.adoptionRewardPool = prop.trialBudget - prop.trialSpent;

        // Return proposer's stake
        IERC20(rpbTokenAddress).transfer(prop.proposer, prop.stakeAmount);

        emit ProposalAdopted(proposalId, prop.adoptionRewardPool);
        emit ProposalStatusChanged(proposalId, oldStatus, ProposalStatus.Adopted);
    }

    /**
     * @notice Reject a proposal (admin or after failed evaluation).
     */
    function rejectProposal(uint256 proposalId) external onlyAdmin {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        if (prop.status == ProposalStatus.Adopted || prop.status == ProposalStatus.Rejected) {
            revert InvalidStatus();
        }

        _rejectProposal(proposalId);
    }

    function _rejectProposal(uint256 proposalId) internal {
        Proposal storage prop = proposals[proposalId];
        ProposalStatus oldStatus = prop.status;

        prop.status = ProposalStatus.Rejected;
        prop.statusChangedAt = block.timestamp;

        // Refund stake minus processing fee
        uint256 fee = (prop.stakeAmount * processingFeeBps) / 10000;
        uint256 refund = prop.stakeAmount - fee;

        if (refund > 0) {
            IERC20(rpbTokenAddress).transfer(prop.proposer, refund);
        }
        // Fee stays in contract (treasury can collect)

        // Return unspent trial budget
        uint256 unspent = prop.trialBudget - prop.trialSpent;
        if (unspent > 0) {
            // Unspent trial funds stay in contract for treasury to reclaim
            prop.trialSpent = prop.trialBudget; // Mark as resolved
        }

        emit ProposalRejected(proposalId, refund, fee);
        emit ProposalStatusChanged(proposalId, oldStatus, ProposalStatus.Rejected);
    }

    // =========================================================================
    // REWARD CLAIMING (Contribution Spectrum)
    // =========================================================================

    /**
     * @notice Claim trial participation reward after adoption.
     * @dev Reward = (participant_weighted_units / total_weighted_units) * remaining_budget
     *      Weight is determined by contribution type.
     * @param proposalId The adopted proposal
     * @param participantIndex Index in the trialParticipants array
     */
    function claimTrialReward(
        uint256 proposalId,
        uint256 participantIndex
    ) external nonReentrant {
        Proposal storage prop = proposals[proposalId];
        if (prop.status != ProposalStatus.Adopted) revert InvalidStatus();

        TrialParticipant[] storage participants = trialParticipants[proposalId];
        if (participantIndex >= participants.length) revert InvalidAmount();

        TrialParticipant storage participant = participants[participantIndex];
        if (participant.participant != msg.sender) revert NotAuthorized();
        if (participant.rewardClaimed) revert AlreadyClaimed();

        // Calculate weighted contribution
        uint256 weight = _getContributionWeight(participant.contributionType);
        uint256 myWeighted = participant.units * weight;

        // Calculate total weighted
        uint256 totalWeighted = 0;
        for (uint256 i = 0; i < participants.length; i++) {
            uint256 w = _getContributionWeight(participants[i].contributionType);
            totalWeighted += participants[i].units * w;
        }

        if (totalWeighted == 0) revert NothingToClaim();

        // Use the snapshotted reward pool from adoption time
        uint256 reward = (myWeighted * prop.adoptionRewardPool) / totalWeighted;

        if (reward == 0) revert NothingToClaim();

        participant.rewardClaimed = true;
        prop.trialSpent += reward;

        IERC20(rpbTokenAddress).transfer(msg.sender, reward);

        emit ParticipantRewardClaimed(proposalId, msg.sender, reward);
    }

    function _getContributionWeight(ContributionType ct) internal view returns (uint256) {
        if (ct == ContributionType.Compute) return contributionWeights.compute;
        if (ct == ContributionType.Diagnosis) return contributionWeights.diagnosis;
        if (ct == ContributionType.Proposal) return contributionWeights.proposal;
        if (ct == ContributionType.Validation) return contributionWeights.validation;
        return contributionWeights.compute; // default
    }

    // =========================================================================
    // RPB PROMPT MANAGEMENT (via Registry)
    // =========================================================================

    /**
     * @notice Update the RPB constitutional prompt CID in Registry.
     * @dev Timelock-protected. Higher quorum enforced at DAO level.
     * @param promptCid New prompt CID
     */
    function updateRPBPrompt(string calldata promptCid) external onlyTimelock {
        if (bytes(promptCid).length == 0) revert InvalidProposal();

        currentPromptVersion++;

        // Store versioned prompt in Registry
        string memory versionKey = string.concat("rpb.prompt.v", _uint2str(currentPromptVersion));
        Registry(registryAddress).editRegistry(versionKey, promptCid);

        // Update current pointer
        Registry(registryAddress).editRegistry("rpb.prompt.current", promptCid);

        // Store version number
        Registry(registryAddress).editRegistry("rpb.prompt.version", _uint2str(currentPromptVersion));

        emit RPBPromptUpdated(currentPromptVersion, promptCid);
    }

    /**
     * @notice Get the current RPB prompt CID from Registry.
     */
    function getCurrentPromptCid() external view returns (string memory) {
        return Registry(registryAddress).getRegistryValue("rpb.prompt.current");
    }

    /**
     * @notice Get a specific version of the RPB prompt.
     */
    function getPromptVersion(uint256 version) external view returns (string memory) {
        string memory key = string.concat("rpb.prompt.v", _uint2str(version));
        return Registry(registryAddress).getRegistryValue(key);
    }

    // =========================================================================
    // DAO INTEGRATION
    // =========================================================================

    /**
     * @notice Link a proposal to a HomebaseDAO proposal for human voting.
     * @param proposalId Evolution proposal ID
     * @param daoProposalId DAO proposal ID
     */
    function linkDAOProposal(
        uint256 proposalId,
        uint256 daoProposalId
    ) external onlyAdmin {
        Proposal storage prop = proposals[proposalId];
        if (prop.createdAt == 0) revert ProposalNotFound();
        prop.daoProposalId = daoProposalId;
    }

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    function setMinStake(uint256 _minStake) external onlyAdmin {
        if (_minStake == 0) revert InvalidAmount();
        minStake = _minStake;
    }

    function setProcessingFeeBps(uint256 _feeBps) external onlyAdmin {
        if (_feeBps > 5000) revert InvalidAmount(); // Max 50%
        processingFeeBps = _feeBps;
    }

    function setEvaluationQuorum(uint256 _quorum) external onlyAdmin {
        if (_quorum == 0) revert InvalidAmount();
        evaluationQuorum = _quorum;
    }

    function setApprovalThreshold(uint256 _thresholdBps) external onlyAdmin {
        if (_thresholdBps == 0 || _thresholdBps > 10000) revert InvalidAmount();
        approvalThresholdBps = _thresholdBps;
    }

    function setEvaluationPeriod(uint256 _period) external onlyAdmin {
        if (_period == 0) revert InvalidAmount();
        evaluationPeriod = _period;
    }

    function setContributionWeights(
        uint256 compute,
        uint256 diagnosis,
        uint256 proposal_,
        uint256 validation
    ) external onlyAdmin {
        if (compute == 0) revert InvalidAmount();
        contributionWeights = ContributionWeights(compute, diagnosis, proposal_, validation);
        emit ContributionWeightsUpdated(compute, diagnosis, proposal_, validation);
    }

    function setRPBEvaluator(address evaluator, bool enabled) external onlyAdmin {
        if (evaluator == address(0)) revert ZeroAddress();
        rpbEvaluators[evaluator] = enabled;
        emit RPBEvaluatorSet(evaluator, enabled);
    }

    function transferAdmin(address newAdmin) external onlyAdmin {
        if (newAdmin == address(0)) revert ZeroAddress();
        admin = newAdmin;
    }

    /**
     * @notice Recover ATN from contract (fees, unspent trial budgets).
     */
    function recoverTokens(address to, uint256 amount) external onlyAdmin {
        if (to == address(0)) revert ZeroAddress();
        IERC20(rpbTokenAddress).transfer(to, amount);
    }

    // =========================================================================
    // VIEW FUNCTIONS
    // =========================================================================

    function getProposal(uint256 proposalId) external view returns (
        address proposer,
        string memory contentCid,
        uint256 stakeAmount,
        ProposalStatus status,
        uint256 createdAt,
        uint256 trialBudget,
        uint256 evaluationCount,
        int256 evaluationScore,
        uint256 daoProposalId
    ) {
        Proposal storage p = proposals[proposalId];
        return (
            p.proposer, p.contentCid, p.stakeAmount, p.status,
            p.createdAt, p.trialBudget, p.evaluationCount,
            p.evaluationScore, p.daoProposalId
        );
    }

    function getProposalCount() external view returns (uint256) {
        return proposalIds.length;
    }

    function getEvaluations(uint256 proposalId) external view returns (
        address[] memory evaluatorAddrs,
        bool[] memory approvals,
        uint256[] memory confidences,
        string[] memory reasonCids
    ) {
        RPBEvaluation[] storage evals = evaluations[proposalId];
        uint256 len = evals.length;

        evaluatorAddrs = new address[](len);
        approvals = new bool[](len);
        confidences = new uint256[](len);
        reasonCids = new string[](len);

        for (uint256 i = 0; i < len; i++) {
            evaluatorAddrs[i] = evals[i].evaluator;
            approvals[i] = evals[i].approve;
            confidences[i] = evals[i].confidence;
            reasonCids[i] = evals[i].reasonCid;
        }
    }

    function getTrialParticipants(uint256 proposalId) external view returns (
        address[] memory participants,
        uint8[] memory types,
        uint256[] memory units,
        bool[] memory claimed
    ) {
        TrialParticipant[] storage parts = trialParticipants[proposalId];
        uint256 len = parts.length;

        participants = new address[](len);
        types = new uint8[](len);
        units = new uint256[](len);
        claimed = new bool[](len);

        for (uint256 i = 0; i < len; i++) {
            participants[i] = parts[i].participant;
            types[i] = uint8(parts[i].contributionType);
            units[i] = parts[i].units;
            claimed[i] = parts[i].rewardClaimed;
        }
    }

    function getContributionWeights() external view returns (
        uint256 compute, uint256 diagnosis, uint256 proposal_, uint256 validation
    ) {
        return (
            contributionWeights.compute,
            contributionWeights.diagnosis,
            contributionWeights.proposal,
            contributionWeights.validation
        );
    }

    // =========================================================================
    // INTERNAL HELPERS
    // =========================================================================

    function _uint2str(uint256 _i) internal pure returns (string memory) {
        if (_i == 0) return "0";
        uint256 j = _i;
        uint256 len;
        while (j != 0) {
            len++;
            j /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint256 k = len;
        while (_i != 0) {
            k--;
            bstr[k] = bytes1(uint8(48 + _i % 10));
            _i /= 10;
        }
        return string(bstr);
    }
}
