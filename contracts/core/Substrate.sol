// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Substrate — chain surface for the world-model substrate.
///
/// Substrate-native replacement for the pre-substrate RPB / Registry /
/// AutonetDAO contracts. Designed from scratch around the substrate's
/// data shapes (epoch_id, epoch_root, agent_mint blob CID,
/// authoritative payload hash) rather than retro-fitting them into
/// task-driven training contracts.
///
/// Two responsibilities live here, intentionally folded into one
/// contract because they share state:
///
///   1. **Epoch anchoring** — at each network epoch close, one daemon
///      submits the canonical anchor (epoch_id, epoch_root,
///      prev_epoch_root, prev_anchor_hash, agent_mint_cid, payload_hash).
///      The chain verifies the anchor links into the chain correctly
///      and rejects forgeries.
///
///   2. **Agent training records** — registered agents read their
///      authoritative mint from the off-chain agent_mint blob (CID is
///      on chain), then call recordTrainingForEpoch(amount, epochId)
///      from their own keypair. The contract dedupes per (agent,
///      epochId), binds the record to a real anchored epoch, and
///      tracks cumulative per-agent training contribution.
///
/// What is deliberately NOT here
/// -----------------------------
///
///   - ATN token / ERC20 / share purchases. The substrate doesn't
///     need a token tied to its operation. Token economics will
///     come back as a separate substrate-native contract when we get
///     to incentive distribution. This contract is just the
///     attribution layer.
///   - Heartbeat / DAO governance. Separate concern.
///   - Capability scoring, evolution proposals, sponsorship
///     hierarchies. All pre-substrate concepts.
///
/// What an agent's chain interaction looks like
/// --------------------------------------------
///
/// 1. Once: ``registerAgent(lineageHash)``. Sets msg.sender as an
///    active agent.
///
/// 2. Per epoch close it minted in:
///      a. Off-chain: read EpochAnchored event for the epoch_id.
///      b. Off-chain: fetch agent_mint blob via the cid.
///      c. Off-chain: decode own mint amount, scale to uint256.
///      d. On-chain: ``recordTrainingForEpoch(amount, epochIdHash)``.
///         Reverts if (msg.sender, epochIdHash) already submitted.
///
/// 3. Anytime: read ``agentMintTotal(address)`` for cumulative or
///    ``mintForEpoch(address, bytes32)`` for per-epoch.
contract Substrate {
    // =========================================================================
    // Anchor records
    // =========================================================================

    struct Anchor {
        string epochId;
        bytes32 epochRoot;
        bytes32 prevEpochRoot;
        bytes32 prevAnchorHash;
        string agentMintCid;
        bytes32 payloadHash;
        address submitter;
        uint256 blockNumber;
        uint256 timestamp;
    }

    /// @dev Sequential storage of anchors as submitted.
    Anchor[] private anchors;
    mapping(string => uint256) private anchorIndexByEpochId;  // 1-indexed
    mapping(bytes32 => string) private epochIdByHash;          // for hash → string lookup
    bytes32 public latestAnchorHash;
    bytes32 public latestEpochRoot;

    event EpochAnchored(
        string indexed epochId,
        bytes32 indexed epochIdHash,
        bytes32 indexed epochRoot,
        bytes32 prevEpochRoot,
        bytes32 prevAnchorHash,
        bytes32 payloadHash,
        string agentMintCid,
        address submitter,
        uint256 blockNumber
    );

    error EpochIdEmpty();
    error EpochAlreadyAnchored(string epochId);
    error PrevEpochRootMismatch(bytes32 expected, bytes32 got);
    error PrevAnchorHashMismatch(bytes32 expected, bytes32 got);

    /// @notice Submit an anchor for a network epoch close.
    function submitAnchor(
        string calldata epochId,
        bytes32 epochRoot,
        bytes32 prevEpochRoot,
        bytes32 prevAnchorHash,
        string calldata agentMintCid,
        bytes32 payloadHash
    ) external {
        if (bytes(epochId).length == 0) revert EpochIdEmpty();
        if (anchorIndexByEpochId[epochId] != 0) {
            revert EpochAlreadyAnchored(epochId);
        }
        if (prevEpochRoot != latestEpochRoot) {
            revert PrevEpochRootMismatch(latestEpochRoot, prevEpochRoot);
        }
        if (prevAnchorHash != latestAnchorHash) {
            revert PrevAnchorHashMismatch(latestAnchorHash, prevAnchorHash);
        }

        bytes32 epochIdHash = keccak256(bytes(epochId));

        anchors.push(Anchor({
            epochId: epochId,
            epochRoot: epochRoot,
            prevEpochRoot: prevEpochRoot,
            prevAnchorHash: prevAnchorHash,
            agentMintCid: agentMintCid,
            payloadHash: payloadHash,
            submitter: msg.sender,
            blockNumber: block.number,
            timestamp: block.timestamp
        }));
        anchorIndexByEpochId[epochId] = anchors.length;
        epochIdByHash[epochIdHash] = epochId;

        bytes32 newAnchorHash = keccak256(abi.encode(
            epochId,
            epochRoot,
            prevEpochRoot,
            prevAnchorHash,
            agentMintCid,
            payloadHash,
            msg.sender,
            block.number
        ));
        latestAnchorHash = newAnchorHash;
        latestEpochRoot = epochRoot;

        emit EpochAnchored(
            epochId,
            epochIdHash,
            epochRoot,
            prevEpochRoot,
            prevAnchorHash,
            payloadHash,
            agentMintCid,
            msg.sender,
            block.number
        );
    }

    function anchorCount() external view returns (uint256) {
        return anchors.length;
    }

    function getAnchor(uint256 index) external view returns (Anchor memory) {
        return anchors[index];
    }

    function getAnchorByEpochId(string calldata epochId) external view returns (Anchor memory) {
        uint256 idx = anchorIndexByEpochId[epochId];
        require(idx > 0, "epoch not anchored");
        return anchors[idx - 1];
    }

    function isAnchored(string calldata epochId) external view returns (bool) {
        return anchorIndexByEpochId[epochId] != 0;
    }

    /// @notice True iff an epochIdHash maps to an anchored epoch.
    /// @dev    Convenience for the per-epoch training submission path
    ///         where agents reference epochs by hash, not string.
    function isAnchoredByHash(bytes32 epochIdHash) external view returns (bool) {
        return bytes(epochIdByHash[epochIdHash]).length > 0;
    }

    // =========================================================================
    // Agent registration
    // =========================================================================

    struct Agent {
        bytes32 lineageHash;
        uint256 registeredAt;
        bool active;
        uint256 totalTrainingMint;
        uint256 trainingSubmissionCount;
    }

    mapping(address => Agent) public agents;
    mapping(bytes32 => address) public agentByLineage;
    address[] private registeredAgentList;

    event AgentRegistered(address indexed agent, bytes32 indexed lineageHash, uint256 timestamp);

    error AlreadyRegistered();
    error LineageHashRequired();
    error LineageHashAlreadyUsed();
    error AgentNotActive();

    /// @notice Register the caller as a substrate agent.
    /// @param lineageHash Caller-supplied identity hash (e.g.
    ///                    keccak256 of the agent's manifest CID).
    function registerAgent(bytes32 lineageHash) external {
        if (agents[msg.sender].registeredAt != 0) revert AlreadyRegistered();
        if (lineageHash == bytes32(0)) revert LineageHashRequired();
        if (agentByLineage[lineageHash] != address(0)) revert LineageHashAlreadyUsed();

        agents[msg.sender] = Agent({
            lineageHash: lineageHash,
            registeredAt: block.timestamp,
            active: true,
            totalTrainingMint: 0,
            trainingSubmissionCount: 0
        });
        agentByLineage[lineageHash] = msg.sender;
        registeredAgentList.push(msg.sender);

        emit AgentRegistered(msg.sender, lineageHash, block.timestamp);
    }

    function isRegistered(address agent) external view returns (bool) {
        return agents[agent].registeredAt != 0;
    }

    function registeredAgentCount() external view returns (uint256) {
        return registeredAgentList.length;
    }

    function getRegisteredAgent(uint256 index) external view returns (address) {
        return registeredAgentList[index];
    }

    // =========================================================================
    // Per-epoch training records
    // =========================================================================

    /// @dev mintForEpoch[agent][epochIdHash] = scaled mint amount.
    ///      Zero means "agent did not submit for that epoch."
    mapping(address => mapping(bytes32 => uint256)) public mintForEpoch;

    /// @dev Cumulative training mint for each agent.
    mapping(address => uint256) public agentMintTotal;

    /// @dev Network-wide cumulative training mint.
    uint256 public networkMintTotal;

    event TrainingRecorded(
        address indexed agent,
        bytes32 indexed epochIdHash,
        uint256 amount,
        uint256 cumulativeForAgent
    );

    error EpochNotAnchored(bytes32 epochIdHash);
    error AlreadySubmittedForEpoch(bytes32 epochIdHash);

    /// @notice Record this agent's authoritative mint for an
    ///         already-anchored epoch.
    /// @dev    Per-(agent, epochIdHash) idempotent. The epoch must
    ///         have been anchored via submitAnchor() first — this
    ///         binds the record to a federation-ratified close.
    /// @param amount The mint amount, integer-scaled (default ×1e6
    ///               at the agent-side submitter, see Phase 5.6 docs).
    /// @param epochIdHash keccak256 of the epoch_id string.
    function recordTrainingForEpoch(uint256 amount, bytes32 epochIdHash) external {
        if (!agents[msg.sender].active) revert AgentNotActive();
        if (bytes(epochIdByHash[epochIdHash]).length == 0) {
            revert EpochNotAnchored(epochIdHash);
        }
        if (mintForEpoch[msg.sender][epochIdHash] != 0) {
            revert AlreadySubmittedForEpoch(epochIdHash);
        }
        // Allow zero submissions to be skipped silently — the agent
        // had no mint share for this epoch but may still want a
        // marker on chain. Here we treat zero as a no-op (no event,
        // no state change).
        if (amount == 0) {
            return;
        }

        mintForEpoch[msg.sender][epochIdHash] = amount;
        agentMintTotal[msg.sender] += amount;
        networkMintTotal += amount;
        agents[msg.sender].totalTrainingMint += amount;
        agents[msg.sender].trainingSubmissionCount += 1;

        emit TrainingRecorded(
            msg.sender,
            epochIdHash,
            amount,
            agentMintTotal[msg.sender]
        );
    }

    /// @notice Has (agent, epochIdHash) already been submitted?
    function hasSubmittedForEpoch(address agent, bytes32 epochIdHash) external view returns (bool) {
        return mintForEpoch[agent][epochIdHash] != 0;
    }
}
