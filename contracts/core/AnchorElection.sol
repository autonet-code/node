// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";
import {Strings} from "@openzeppelin/contracts/utils/Strings.sol";
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";

/// @dev The Substrate surface an election needs.
interface ISubstrateAnchor {
    function anchor(
        string calldata epochId,
        bytes32 epochRoot,
        bytes32 prevEpochRoot,
        bytes32 prevAnchorHash,
        string calldata agentMintCid,
        bytes32 payloadHash,
        bytes32 agentMintRoot,
        uint256 cutoff
    ) external;
    function anchorCount() external view returns (uint256);
    function isLive(address agent) external view returns (bool);
}

/// @dev The DAO Governor surface: OpenZeppelin Governor plus the options
///      counting module's winningOption (trustless-contracts HomebaseDAO).
interface IOptionGovernor {
    function state(uint256 proposalId) external view returns (uint8);
    function hashProposal(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) external pure returns (uint256);
    function propose(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description
    ) external returns (uint256);
    function proposalEta(uint256 proposalId) external view returns (uint256);
    function proposalSnapshot(uint256 proposalId) external view returns (uint256);
    function votingDelay() external view returns (uint256);
    function winningOption(uint256 proposalId)
        external view returns (bool decided, uint256 index);
    function timelock() external view returns (address);
}

/// @dev Timestamp-clocked REP votes (RepToken, ERC20Votes).
interface IPastVotes {
    function getPastVotes(address account, uint256 timepoint) external view returns (uint256);
    function getPastTotalSupply(uint256 timepoint) external view returns (uint256);
}

/// @title AnchorElection
/// @notice Single-choice election of each epoch's anchor (docs/
///         tool_substrate.md, "Items 5-6 DESIGN", 2026-10-26).
///
/// Each epoch's election is one DAO proposal whose only action is
/// `anchorWinner(epochNumber, attempt)` on this contract, with the
/// description `descriptionOf(epochNumber, attempt)`, so every daemon
/// derives the same proposal id. While the proposal is Pending or Active,
/// the agents drawn as proposers each register one option, from their own
/// wallet, so a proposal opened early locks no close out. Voting index 0 is
/// reject ("none of these"); registered options start at 1. The Governor's
/// options counting module tallies one vote per account by index; on
/// execution the winning option is anchored on Substrate, or, if reject won,
/// nothing is and the daemons open the next attempt.
///
/// The draw is threshold sortition, checked locally per account: a live
/// agent with REP share s (votes at the proposal's creation / total REP)
/// is drawn when hash(chain, this, epoch, attempt, agent) falls below
/// `proposers * s` of the hash range, so about `proposers` agents are
/// drawn per epoch. From attempt `openAfterAttempts` on, any live agent with REP
/// may register: an epoch never stalls because no drawn agent showed up.
contract AnchorElection is Initializable {
    struct Option {
        string epochId;
        bytes32 epochRoot;
        bytes32 prevEpochRoot;
        bytes32 prevAnchorHash;
        string agentMintCid;
        bytes32 payloadHash;
        bytes32 agentMintRoot;
        uint256 cutoff;
    }

    uint8 private constant PENDING = 0;
    uint256 private constant PPM = 1_000_000;

    // Set once at setup (the constructor, or initialize() on a clone); no
    // setters.
    ISubstrateAnchor public substrate;
    IOptionGovernor public governor;
    IPastVotes public repToken;

    /// @notice Expected number of drawn proposers per epoch (3 at setup).
    uint256 public proposers;
    uint8 private constant ACTIVE_STATE = 1;
    /// @notice Minimum REP share (parts per million) to be drawn (0 at setup).
    uint256 public proposerFloorPpm;
    /// @notice Most options one election may hold (8 at setup).
    uint256 public maxOptions;
    /// @notice The attempt from which any live agent may register (2 at setup).
    uint256 public openAfterAttempts;

    mapping(uint256 => Option[]) private _options;
    mapping(uint256 => address[]) private _proposerOf;
    mapping(uint256 => mapping(bytes32 => bool)) private _seenOption;
    mapping(uint256 => mapping(address => bool)) private _registered;

    /// @notice How many attempts were opened for an epoch's election. The
    ///         live (or latest) one is attemptsOpened - 1.
    mapping(uint256 => uint256) public attemptsOpened;
    /// @notice When this contract last anchored a winner.
    uint256 public lastAnchoredAt;

    // Governor proposal states (OpenZeppelin IGovernor.ProposalState).
    uint8 private constant CANCELED_STATE = 2;
    uint8 private constant DEFEATED_STATE = 3;
    uint8 private constant QUEUED_STATE = 5;
    uint8 private constant EXPIRED_STATE = 6;
    uint8 private constant EXECUTED_STATE = 7;
    /// @notice How long a queued winner may sit unexecuted, once its epoch
    ///         is next to anchor and its timelock delay has passed, before
    ///         a new attempt may open. Anyone can execute a queued winner,
    ///         so one still queued after this cannot anchor (its option
    ///         does not fit the chain as it now stands).
    uint256 public constant EXECUTE_GRACE = 1 hours;

    event OptionRegistered(
        uint256 indexed proposalId, uint256 indexed epochNumber,
        uint256 index, address indexed proposer, bytes32 optionHash);
    event WinnerAnchored(
        uint256 indexed proposalId, uint256 indexed epochNumber,
        uint256 index, bytes32 optionHash);
    event ElectionOpened(
        uint256 indexed proposalId, uint256 indexed epochNumber, uint256 attempt);
    event ParamsSet(uint256 proposers, uint256 proposerFloorPpm,
                    uint256 maxOptions, uint256 openAfterAttempts);

    error NotTimelock();
    error NotPending(uint256 proposalId);
    error NotLiveAgent(address account);
    error NotDrawn(address account);
    error AlreadyRegistered(address account);
    error DuplicateOption(bytes32 optionHash);
    error TooManyOptions();
    error SnapshotNotPast();
    error WrongEpoch(uint256 expected, uint256 got);
    error Undecided(uint256 proposalId);
    error EpochNotOpenable(uint256 epochNumber);
    error AttemptLive(uint256 proposalId);
    error BadParams();

    /// @dev Sets up the instance it creates, so a directly deployed
    ///      election works as a plain contract and can never be initialized
    ///      again; that also locks the implementation behind the clones.
    constructor(address substrate_, address governor_, address repToken_) initializer {
        _init(substrate_, governor_, repToken_);
    }

    /// @notice Set up a clone (EIP-1167): the constructor's arguments, in
    ///         the same order. Once per clone, in the transaction that
    ///         creates it.
    function initialize(address substrate_, address governor_, address repToken_)
        external initializer
    {
        _init(substrate_, governor_, repToken_);
    }

    function _init(address substrate_, address governor_, address repToken_) private {
        substrate = ISubstrateAnchor(substrate_);
        governor = IOptionGovernor(governor_);
        repToken = IPastVotes(repToken_);
        proposers = 3;
        maxOptions = 8;
        openAfterAttempts = 2;
    }

    modifier onlyTimelock() {
        if (msg.sender != governor.timelock()) revert NotTimelock();
        _;
    }

    // ---------------------------------------------------------------------
    // Proposal identity
    // ---------------------------------------------------------------------

    function descriptionOf(uint256 epochNumber, uint256 attempt)
        public pure returns (string memory)
    {
        return string.concat(
            "Anchor election: epoch ", Strings.toString(epochNumber),
            ", attempt ", Strings.toString(attempt));
    }

    function callOf(uint256 epochNumber, uint256 attempt)
        public pure returns (bytes memory)
    {
        return abi.encodeCall(AnchorElection.anchorWinner, (epochNumber, attempt));
    }

    /// @notice IOptionSource: whether `data` is this source's election call.
    ///         The governor counts a proposal as a single-choice election
    ///         only when its one action is such a call; any other call on
    ///         this contract (setParams) is voted For, Against or Abstain
    ///         like any proposal. True only for a call to anchorWinner.
    function isElectionCall(bytes calldata data) external pure returns (bool) {
        return data.length >= 4 && bytes4(data[:4]) == AnchorElection.anchorWinner.selector;
    }

    function proposalIdOf(uint256 epochNumber, uint256 attempt)
        public view returns (uint256)
    {
        address[] memory targets = new address[](1);
        targets[0] = address(this);
        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);
        calldatas[0] = callOf(epochNumber, attempt);
        return governor.hashProposal(
            targets, values, calldatas,
            keccak256(bytes(descriptionOf(epochNumber, attempt))));
    }

    // ---------------------------------------------------------------------
    // The draw
    // ---------------------------------------------------------------------

    /// @notice The timepoint REP is read at for a proposal: its creation
    ///         (the snapshot minus the voting delay).
    function drawTimepoint(uint256 proposalId) public view returns (uint256) {
        return governor.proposalSnapshot(proposalId) - governor.votingDelay();
    }

    function isDrawn(uint256 epochNumber, uint256 attempt, address account,
                     uint256 timepoint) public view returns (bool) {
        if (!substrate.isLive(account)) return false;
        uint256 supply = repToken.getPastTotalSupply(timepoint);
        if (supply == 0) return false;
        uint256 votes = repToken.getPastVotes(account, timepoint);
        if (votes == 0 || votes * PPM < proposerFloorPpm * supply) return false;
        if (attempt >= openAfterAttempts) return true;
        uint256 h = uint256(keccak256(abi.encode(
            block.chainid, address(this), epochNumber, attempt, account)));
        // h / 2^256 < proposers * votes / supply, without overflow.
        return Math.mulDiv(h, supply, type(uint256).max) < proposers * votes;
    }

    // ---------------------------------------------------------------------
    // Opening an election
    //
    // Only this contract proposes its elections (the governor refuses an
    // election call from anyone else), and it never cancels one. So nobody
    // can take an election's proposal id and kill it, now or ahead of time:
    // an attempt exists only once open() makes it, and there is always a
    // next one.
    // ---------------------------------------------------------------------

    /// @notice Open the election for an epoch: its first attempt, or the
    ///         next one once the latest ended without an anchor. Any live
    ///         agent may call it. The epoch must be the next to anchor, or
    ///         follow an epoch whose election is already open (closes are
    ///         pipelined: an epoch's election may run before its parent's
    ///         winner lands). The attempt number is this contract's count,
    ///         never the caller's. Registration stays open while the vote
    ///         runs, so an attempt opened before the closes exist still
    ///         takes them.
    function open(uint256 epochNumber) external returns (uint256 proposalId) {
        if (!substrate.isLive(msg.sender)) revert NotLiveAgent(msg.sender);
        uint256 anchored = substrate.anchorCount();
        if (epochNumber <= anchored
                || (epochNumber != anchored + 1 && attemptsOpened[epochNumber - 1] == 0)) {
            revert EpochNotOpenable(epochNumber);
        }
        uint256 attempt = attemptsOpened[epochNumber];
        if (attempt > 0) {
            uint256 latest = proposalIdOf(epochNumber, attempt - 1);
            if (!_endedWithoutAnchor(latest, epochNumber == anchored + 1)) {
                revert AttemptLive(latest);
            }
        }
        attemptsOpened[epochNumber] = attempt + 1;

        address[] memory targets = new address[](1);
        targets[0] = address(this);
        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);
        calldatas[0] = callOf(epochNumber, attempt);
        proposalId = governor.propose(
            targets, values, calldatas, descriptionOf(epochNumber, attempt));
        emit ElectionOpened(proposalId, epochNumber, attempt);
    }

    /// @dev Whether an unanchored epoch's attempt is over. Defeated, expired
    ///      or cancelled: yes. Executed: yes, reject won (the epoch is still
    ///      unanchored). Queued: only when the epoch is next to anchor and
    ///      the winner has stayed unexecuted for EXECUTE_GRACE past both its
    ///      timelock eta and the previous anchor.
    function _endedWithoutAnchor(uint256 proposalId, bool isNext) private view returns (bool) {
        uint8 st = governor.state(proposalId);
        if (st == CANCELED_STATE || st == DEFEATED_STATE || st == EXPIRED_STATE
                || st == EXECUTED_STATE) {
            return true;
        }
        if (st == QUEUED_STATE && isNext) {
            uint256 eta = governor.proposalEta(proposalId);
            uint256 since = eta > lastAnchoredAt ? eta : lastAnchoredAt;
            return eta != 0 && block.timestamp >= since + EXECUTE_GRACE;
        }
        return false;
    }

    // ---------------------------------------------------------------------
    // Options
    // ---------------------------------------------------------------------

    function optionHashOf(Option memory o) public pure returns (bytes32) {
        return keccak256(abi.encode(
            o.epochId, o.epochRoot, o.prevEpochRoot, o.prevAnchorHash,
            o.agentMintCid, o.payloadHash, o.agentMintRoot, o.cutoff));
    }

    /// @notice Register the caller's option for an epoch's election, while
    ///         its proposal is Pending or Active. One option per drawn agent;
    ///         identical content registers once.
    function registerOption(uint256 epochNumber, uint256 attempt,
                            Option calldata option) external returns (uint256) {
        uint256 pid = proposalIdOf(epochNumber, attempt);
        uint8 st = governor.state(pid);
        if (st != PENDING && st != ACTIVE_STATE) revert NotPending(pid);
        if (!substrate.isLive(msg.sender)) revert NotLiveAgent(msg.sender);
        uint256 timepoint = drawTimepoint(pid);
        if (timepoint >= block.timestamp) revert SnapshotNotPast();
        if (!isDrawn(epochNumber, attempt, msg.sender, timepoint)) {
            revert NotDrawn(msg.sender);
        }
        if (_registered[pid][msg.sender]) revert AlreadyRegistered(msg.sender);
        bytes32 h = optionHashOf(option);
        if (_seenOption[pid][h]) revert DuplicateOption(h);
        if (_options[pid].length >= maxOptions) revert TooManyOptions();

        _registered[pid][msg.sender] = true;
        _seenOption[pid][h] = true;
        _options[pid].push(option);
        _proposerOf[pid].push(msg.sender);
        uint256 index = _options[pid].length;   // voting index: 0 is reject
        emit OptionRegistered(pid, epochNumber, index, msg.sender, h);
        return index;
    }

    /// @notice IOptionSource: the options of a proposal.
    function optionCount(uint256 proposalId) external view returns (uint256) {
        return _options[proposalId].length + 1;
    }

    /// @notice IOptionSource: an option's content hash (ties go to the
    ///         lowest).
    function optionHash(uint256 proposalId, uint256 index)
        external view returns (bytes32)
    {
        if (index == 0) return bytes32(0);
        return optionHashOf(_options[proposalId][index - 1]);
    }

    function getOption(uint256 proposalId, uint256 index)
        external view returns (Option memory option, address proposer)
    {
        return (_options[proposalId][index - 1], _proposerOf[proposalId][index - 1]);
    }

    // ---------------------------------------------------------------------
    // Execution
    // ---------------------------------------------------------------------

    /// @notice Anchor the election's winning option. Called by the
    ///         timelock when the proposal executes. Reverts while the
    ///         previous epoch is unanchored; the proposal stays Queued and
    ///         can be executed again once it lands.
    function anchorWinner(uint256 epochNumber, uint256 attempt) external onlyTimelock {
        uint256 expected = substrate.anchorCount() + 1;
        if (epochNumber != expected) revert WrongEpoch(expected, epochNumber);
        uint256 pid = proposalIdOf(epochNumber, attempt);
        (bool decided, uint256 index) = governor.winningOption(pid);
        if (!decided) revert Undecided(pid);
        if (index == 0) return;   // reject won: nothing anchors, a new attempt follows
        Option storage o = _options[pid][index - 1];
        substrate.anchor(
            o.epochId, o.epochRoot, o.prevEpochRoot, o.prevAnchorHash,
            o.agentMintCid, o.payloadHash, o.agentMintRoot, o.cutoff);
        lastAnchoredAt = block.timestamp;
        emit WinnerAnchored(pid, epochNumber, index, optionHashOf(o));
    }

    // ---------------------------------------------------------------------
    // Governance
    // ---------------------------------------------------------------------

    function setParams(uint256 proposers_, uint256 proposerFloorPpm_,
                       uint256 maxOptions_, uint256 openAfterAttempts_)
        external onlyTimelock
    {
        // A vote names its option in a uint8 and index 0 is reject, so at
        // most 255 registered options can be voted for.
        if (proposers_ == 0 || maxOptions_ == 0 || maxOptions_ > 255
                || proposerFloorPpm_ > PPM) {
            revert BadParams();
        }
        proposers = proposers_;
        proposerFloorPpm = proposerFloorPpm_;
        maxOptions = maxOptions_;
        openAfterAttempts = openAfterAttempts_;
        emit ParamsSet(proposers_, proposerFloorPpm_, maxOptions_,
                       openAfterAttempts_);
    }
}
