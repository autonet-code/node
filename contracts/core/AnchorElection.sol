// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";
import {Strings} from "@openzeppelin/contracts/utils/Strings.sol";

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
/// derives the same proposal id. While the proposal is Pending (the voting
/// delay), the agents drawn as proposers each register one option, from
/// their own wallet. The Governor's options counting module tallies one
/// vote per account by option index; on execution the winning option is
/// anchored on Substrate.
///
/// The draw is threshold sortition, checked locally per account: a live
/// agent with REP share s (votes at the proposal's creation / total REP)
/// is drawn when hash(chain, this, epoch, attempt, agent) falls below
/// `proposers * s` of the hash range, so about `proposers` agents are
/// drawn per epoch. From attempt `openAfterAttempts` on, any live agent
/// may register: an epoch never stalls because no drawn agent showed up.
contract AnchorElection {
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

    ISubstrateAnchor public immutable substrate;
    IOptionGovernor public immutable governor;
    IPastVotes public immutable repToken;

    /// @notice Expected number of drawn proposers per epoch.
    uint256 public proposers = 3;
    /// @notice Minimum REP share (parts per million) to be drawn.
    uint256 public proposerFloorPpm = 0;
    /// @notice Most options one election may hold.
    uint256 public maxOptions = 8;
    /// @notice The attempt from which any live agent may register.
    uint256 public openAfterAttempts = 2;

    mapping(uint256 => Option[]) private _options;
    mapping(uint256 => address[]) private _proposerOf;
    mapping(uint256 => mapping(bytes32 => bool)) private _seenOption;
    mapping(uint256 => mapping(address => bool)) private _registered;

    event OptionRegistered(
        uint256 indexed proposalId, uint256 indexed epochNumber,
        uint256 index, address indexed proposer, bytes32 optionHash);
    event WinnerAnchored(
        uint256 indexed proposalId, uint256 indexed epochNumber,
        uint256 index, bytes32 optionHash);
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
    error BadParams();

    constructor(address substrate_, address governor_, address repToken_) {
        substrate = ISubstrateAnchor(substrate_);
        governor = IOptionGovernor(governor_);
        repToken = IPastVotes(repToken_);
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
        if (attempt >= openAfterAttempts) return true;
        uint256 supply = repToken.getPastTotalSupply(timepoint);
        if (supply == 0) return false;
        uint256 votes = repToken.getPastVotes(account, timepoint);
        if (votes == 0 || votes * PPM < proposerFloorPpm * supply) return false;
        uint256 h = uint256(keccak256(abi.encode(
            block.chainid, address(this), epochNumber, attempt, account)));
        // h / 2^256 < proposers * votes / supply, without overflow.
        return Math.mulDiv(h, supply, type(uint256).max) < proposers * votes;
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
    ///         its proposal is Pending. One option per drawn agent;
    ///         identical content registers once.
    function registerOption(uint256 epochNumber, uint256 attempt,
                            Option calldata option) external returns (uint256) {
        uint256 pid = proposalIdOf(epochNumber, attempt);
        if (governor.state(pid) != PENDING) revert NotPending(pid);
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
        uint256 index = _options[pid].length - 1;
        emit OptionRegistered(pid, epochNumber, index, msg.sender, h);
        return index;
    }

    /// @notice IOptionSource: the options of a proposal.
    function optionCount(uint256 proposalId) external view returns (uint256) {
        return _options[proposalId].length;
    }

    /// @notice IOptionSource: an option's content hash (ties go to the
    ///         lowest).
    function optionHash(uint256 proposalId, uint256 index)
        external view returns (bytes32)
    {
        return optionHashOf(_options[proposalId][index]);
    }

    function getOption(uint256 proposalId, uint256 index)
        external view returns (Option memory option, address proposer)
    {
        return (_options[proposalId][index], _proposerOf[proposalId][index]);
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
        Option storage o = _options[pid][index];
        substrate.anchor(
            o.epochId, o.epochRoot, o.prevEpochRoot, o.prevAnchorHash,
            o.agentMintCid, o.payloadHash, o.agentMintRoot, o.cutoff);
        emit WinnerAnchored(pid, epochNumber, index, optionHashOf(o));
    }

    // ---------------------------------------------------------------------
    // Governance
    // ---------------------------------------------------------------------

    function setParams(uint256 proposers_, uint256 proposerFloorPpm_,
                       uint256 maxOptions_, uint256 openAfterAttempts_)
        external onlyTimelock
    {
        if (proposers_ == 0 || maxOptions_ == 0 || maxOptions_ > 256
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
