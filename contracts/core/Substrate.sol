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
///   1. **Epoch anchoring** — at each network epoch close, one
///      participant (typically an agent acting as the rotating
///      canonical-ordering coordinator) submits the canonical anchor
///      (epoch_id, epoch_root, prev_epoch_root, prev_anchor_hash,
///      agent_mint_cid, payload_hash). The chain verifies the anchor
///      links into the chain correctly and rejects forgeries.
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
///      c. Off-chain: decode own mint amount, scale to uint256, and
///         build the merkle proof from the full mint map.
///      d. On-chain: ``recordTrainingForEpoch(amount, epochIdHash,
///         proof)``. Reverts if (msg.sender, epochIdHash) already
///         submitted, or if (msg.sender, amount) doesn't verify
///         against the anchor's agentMintRoot.
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
        // Merkle root over (agent address, scaled mint amount) leaves
        // for this epoch. recordTrainingForEpoch verifies the caller's
        // claim against it — mint amounts are federation-ratified, not
        // self-reported. Appended last so prior tuple indexes hold.
        bytes32 agentMintRoot;
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
    /// @param agentMintRoot Merkle root over the epoch's
    ///        (agent, scaledAmount) mint map; sorted-pair hashing,
    ///        double-hashed leaves (see recordTrainingForEpoch).
    function submitAnchor(
        string calldata epochId,
        bytes32 epochRoot,
        bytes32 prevEpochRoot,
        bytes32 prevAnchorHash,
        string calldata agentMintCid,
        bytes32 payloadHash,
        bytes32 agentMintRoot
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
            timestamp: block.timestamp,
            agentMintRoot: agentMintRoot
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
            agentMintRoot,
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

    /// @dev libp2p PeerId bytes (multihash-encoded). Stored separately
    ///      from the Agent struct so updates don't churn the struct's
    ///      storage layout. Variable length to accommodate ed25519
    ///      (~38 bytes) and secp256k1 keys.
    mapping(address => bytes) private agentPeerId;

    /// @dev Browser-reachable WebSocket endpoint (wss://...) the daemon
    ///      hosting this agent currently answers on. DISTINCT from peerId:
    ///      peerId is the libp2p mesh identity (daemon↔daemon, ~immutable);
    ///      this is where a *web app* dials (mutable presence — changes on
    ///      every daemon restart when fronted by an ephemeral tunnel). Stored
    ///      as a string (it's a URL). Off-chain indexers mirror EndpointUpdated
    ///      to the agent directory so a browser can resolve agent → wss.
    mapping(address => string) private agentEndpoint;

    event AgentRegistered(
        address indexed agent,
        bytes32 indexed lineageHash,
        bytes peerId,
        uint256 timestamp
    );

    event PeerIdUpdated(address indexed agent, bytes peerId);

    /// @notice Emitted when an agent's browser-reachable wss endpoint changes.
    ///         Indexers mirror this to the off-chain agent directory.
    event EndpointUpdated(address indexed agent, string wsEndpoint);

    error AlreadyRegistered();
    error LineageHashRequired();
    error LineageHashAlreadyUsed();
    error AgentNotActive();
    error PeerIdRequired();

    /// @notice Register the caller as a substrate agent.
    /// @param lineageHash Caller-supplied identity hash (e.g.
    ///                    keccak256 of the agent's manifest CID).
    /// @param peerId libp2p PeerId bytes for off-chain discovery.
    ///        Daemons resolve this to a current multiaddr via the
    ///        libp2p Kademlia DHT — the chain stores *who*, the DHT
    ///        stores *where right now*.
    function registerAgent(bytes32 lineageHash, bytes calldata peerId) external {
        if (agents[msg.sender].registeredAt != 0) revert AlreadyRegistered();
        if (lineageHash == bytes32(0)) revert LineageHashRequired();
        if (agentByLineage[lineageHash] != address(0)) revert LineageHashAlreadyUsed();
        if (peerId.length == 0) revert PeerIdRequired();

        agents[msg.sender] = Agent({
            lineageHash: lineageHash,
            registeredAt: block.timestamp,
            active: true,
            totalTrainingMint: 0,
            trainingSubmissionCount: 0
        });
        agentByLineage[lineageHash] = msg.sender;
        agentPeerId[msg.sender] = peerId;
        registeredAgentList.push(msg.sender);

        emit AgentRegistered(msg.sender, lineageHash, peerId, block.timestamp);
    }

    /// @notice Update this agent's libp2p PeerId. Used when a daemon
    ///         rotates its libp2p keypair (e.g. fresh install on a new
    ///         box) without losing on-chain identity / reputation.
    function updatePeerId(bytes calldata peerId) external {
        if (agents[msg.sender].registeredAt == 0) revert AgentNotActive();
        if (peerId.length == 0) revert PeerIdRequired();
        agentPeerId[msg.sender] = peerId;
        emit PeerIdUpdated(msg.sender, peerId);
    }

    /// @notice Read an agent's PeerId for DHT discovery.
    function getAgentPeerId(address agent) external view returns (bytes memory) {
        return agentPeerId[agent];
    }

    /// @notice Publish/refresh this agent's browser-reachable wss endpoint.
    ///         Agent-signed (msg.sender), so no one can set another agent's
    ///         endpoint — this is what makes the off-chain directory
    ///         hijack-proof. An empty string clears it (going dark).
    ///         Daemons that expose a remote listener call this on startup when
    ///         the endpoint differs from what's on-chain (ephemeral tunnels
    ///         change it every restart).
    function updateEndpoint(string calldata wsEndpoint) external {
        if (agents[msg.sender].registeredAt == 0) revert AgentNotActive();
        agentEndpoint[msg.sender] = wsEndpoint;
        emit EndpointUpdated(msg.sender, wsEndpoint);
    }

    /// @notice Read an agent's current browser-reachable wss endpoint.
    function getAgentEndpoint(address agent) external view returns (string memory) {
        return agentEndpoint[agent];
    }

    function isRegistered(address agent) external view returns (bool) {
        return agents[agent].registeredAt != 0;
    }

    /// @notice Batch variant of ``isRegistered`` for daemons reconciling
    ///         many cached agents against chain in a single call.
    function areRegistered(address[] calldata addrs) external view returns (bool[] memory) {
        bool[] memory out = new bool[](addrs.length);
        for (uint256 i = 0; i < addrs.length; i++) {
            out[i] = agents[addrs[i]].registeredAt != 0;
        }
        return out;
    }

    function registeredAgentCount() external view returns (uint256) {
        return registeredAgentList.length;
    }

    function getRegisteredAgent(uint256 index) external view returns (address) {
        return registeredAgentList[index];
    }

    // =========================================================================
    // Tool registry (tool-substrate v2, docs/tool_substrate.md §On-chain)
    //
    // A published tool's manifest is a sha256-addressed blob in the blob
    // store (NOT an IPFS CID). registerTool anchors AUTHORSHIP on chain:
    // msg.sender is the agent key, so "who authored this manifest digest"
    // becomes chain-verified truth. The federated epoch close keeps reading
    // the gossiped manifest_meta (stays chain-free + deterministic); the
    // chain is the dispute arbiter — a gossip/chain mismatch is CON-able.
    //
    // Chain = truth, blob = storage, indexer mirrors ToolRegistered into
    // the Firestore `tools` collection for the web2 surface. Exact same
    // doctrine as agents (AgentRegistered) and services (ServiceRegistered).
    //
    // Duplicate policy: REVERT on duplicate manifestDigest. A manifest
    // digest is content-addressed, so the same digest is byte-identical
    // content; letting a second agent "register" it would falsely claim
    // authorship of the first agent's blob. First registrant wins; the
    // digest → author binding is permanent. (A genuinely new tool has a
    // different digest, even a one-byte edit, so this never blocks real
    // authorship — only theft of an existing blob.)
    // =========================================================================

    /// @dev manifestDigest → author agent address. address(0) = unregistered.
    mapping(bytes32 => address) public toolAuthor;
    /// @dev Registration timestamp per digest (0 = unregistered).
    mapping(bytes32 => uint256) public toolRegisteredAt;
    /// @dev Per-agent count of tools authored (for cheap directory paging).
    mapping(address => uint256) public toolCountByAgent;

    event ToolRegistered(
        address indexed agent,
        bytes32 indexed manifestDigest,
        uint256 timestamp
    );

    error ManifestDigestRequired();
    error ToolAlreadyRegistered(bytes32 manifestDigest, address author);

    /// @notice Register authorship of a published tool manifest.
    /// @dev    Callable only by an active registered agent (reuses the
    ///         agent registry). Reverts on a digest already registered by
    ///         anyone (content-addressed ⇒ same digest is the same blob).
    /// @param manifestDigest sha256 of the canonical manifest blob bytes.
    function registerTool(bytes32 manifestDigest) external {
        if (!agents[msg.sender].active) revert AgentNotActive();
        if (manifestDigest == bytes32(0)) revert ManifestDigestRequired();
        address existing = toolAuthor[manifestDigest];
        if (existing != address(0)) {
            revert ToolAlreadyRegistered(manifestDigest, existing);
        }
        toolAuthor[manifestDigest] = msg.sender;
        toolRegisteredAt[manifestDigest] = block.timestamp;
        toolCountByAgent[msg.sender] += 1;
        emit ToolRegistered(msg.sender, manifestDigest, block.timestamp);
    }

    /// @notice True iff a manifest digest has a chain-verified author.
    function isToolRegistered(bytes32 manifestDigest) external view returns (bool) {
        return toolAuthor[manifestDigest] != address(0);
    }

    // =========================================================================
    // Per-epoch training records
    // =========================================================================

    /// @dev mintForEpoch[agent][epochIdHash] = scaled mint amount.
    ///      Zero means "agent did not submit for that epoch."
    mapping(address => mapping(bytes32 => uint256)) public mintForEpoch;

    /// @dev Cumulative training mint for each agent.
    /// @notice Phase 7.1 reframes this as the **reputation** ledger:
    ///         monotonic, soulbound, never decreases, no transfer
    ///         function. Reputation is the "I worked for this" tier.
    ///         The kept name ``agentMintTotal`` is back-compat with
    ///         Phase 5.6 callers; ``agentReputation`` is the
    ///         semantic alias used by Phase 7+ tokenomics code.
    mapping(address => uint256) public agentMintTotal;

    /// @dev Network-wide cumulative training mint (sum of all
    ///      agents' agentMintTotal).
    uint256 public networkMintTotal;

    /// @notice Reputation alias for agentMintTotal. Same storage,
    ///         different semantic — reputation is the soulbound
    ///         contribution measure, never decreases, untransferable.
    function agentReputation(address agent) external view returns (uint256) {
        return agentMintTotal[agent];
    }

    event TrainingRecorded(
        address indexed agent,
        bytes32 indexed epochIdHash,
        uint256 amount,
        uint256 cumulativeForAgent
    );

    error EpochNotAnchored(bytes32 epochIdHash);
    error AlreadySubmittedForEpoch(bytes32 epochIdHash);
    error MintRootMissing(bytes32 epochIdHash);
    error MintProofInvalid(bytes32 epochIdHash);

    /// @notice Record this agent's authoritative mint for an
    ///         already-anchored epoch.
    /// @dev    Per-(agent, epochIdHash) idempotent. The epoch must
    ///         have been anchored via submitAnchor() first — this
    ///         binds the record to a federation-ratified close. The
    ///         amount is NOT self-reported: it must verify against
    ///         the anchor's agentMintRoot, so the only claimable
    ///         amount is the one the federation ratified.
    /// @param amount The mint amount, integer-scaled (default ×1e6
    ///               at the agent-side submitter, see Phase 5.6 docs).
    /// @param epochIdHash keccak256 of the epoch_id string.
    /// @param proof Merkle proof for the leaf
    ///              keccak256(bytes.concat(keccak256(abi.encode(
    ///              msg.sender, amount)))) under the anchor's
    ///              agentMintRoot (sorted-pair hashing; a single-leaf
    ///              tree has root == leaf and an empty proof).
    function recordTrainingForEpoch(
        uint256 amount,
        bytes32 epochIdHash,
        bytes32[] calldata proof
    ) external {
        if (!agents[msg.sender].active) revert AgentNotActive();
        string storage epochId = epochIdByHash[epochIdHash];
        if (bytes(epochId).length == 0) {
            revert EpochNotAnchored(epochIdHash);
        }
        if (mintForEpoch[msg.sender][epochIdHash] != 0) {
            revert AlreadySubmittedForEpoch(epochIdHash);
        }
        // Allow zero submissions to be skipped silently — the agent
        // had no mint share for this epoch but may still want a
        // marker on chain. Here we treat zero as a no-op (no event,
        // no state change). Zero shares are not in the mint tree.
        if (amount == 0) {
            return;
        }

        bytes32 root = anchors[anchorIndexByEpochId[epochId] - 1].agentMintRoot;
        if (root == bytes32(0)) revert MintRootMissing(epochIdHash);
        bytes32 leaf = keccak256(
            bytes.concat(keccak256(abi.encode(msg.sender, amount)))
        );
        if (!_verifyMintProof(proof, root, leaf)) {
            revert MintProofInvalid(epochIdHash);
        }

        mintForEpoch[msg.sender][epochIdHash] = amount;
        agentMintTotal[msg.sender] += amount;
        networkMintTotal += amount;
        agents[msg.sender].totalTrainingMint += amount;
        agents[msg.sender].trainingSubmissionCount += 1;

        // Phase 7.1: training mints both ledgers at the same amount.
        // Reputation (agentMintTotal, just bumped above) is soulbound.
        // ATN balance is transferable — the agent can spend it on
        // inference fees once the inference-as-a-service path lands.
        _atnBalance[msg.sender] += amount;
        atnTotalSupply += amount;
        emit ATNTransfer(address(0), msg.sender, amount);

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

    /// @dev Sorted-pair merkle verification (OpenZeppelin-compatible):
    ///      at each level the pair is hashed in ascending byte order,
    ///      so proofs carry no left/right index bits.
    function _verifyMintProof(
        bytes32[] calldata proof,
        bytes32 root,
        bytes32 leaf
    ) private pure returns (bool) {
        bytes32 computed = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 p = proof[i];
            computed = computed <= p
                ? keccak256(abi.encodePacked(computed, p))
                : keccak256(abi.encodePacked(p, computed));
        }
        return computed == root;
    }

    // =========================================================================
    // ATN: transferable token (Phase 7.1)
    //
    // Distinct from reputation. Training mints both reputation and ATN at
    // the same amount; ATN can move between addresses, reputation can't.
    //
    // ERC20-shaped surface: balanceOf, transfer, approve, transferFrom,
    // allowance. Deliberately minimal — no name/symbol/decimals/totalSupply
    // (those belong on a wrapper if we ever want a standard ERC20 facade).
    // No external mint or burn function — the only mint path is through
    // recordTrainingForEpoch. No admin keys.
    // =========================================================================

    mapping(address => uint256) private _atnBalance;
    mapping(address => mapping(address => uint256)) private _atnAllowance;
    uint256 public atnTotalSupply;

    event ATNTransfer(address indexed from, address indexed to, uint256 amount);
    event ATNApproval(address indexed owner, address indexed spender, uint256 amount);

    error InsufficientATN(uint256 requested, uint256 available);
    error InsufficientAllowance(uint256 requested, uint256 available);
    error TransferToZero();

    function balanceOf(address agent) external view returns (uint256) {
        return _atnBalance[agent];
    }

    function allowance(address owner, address spender) external view returns (uint256) {
        return _atnAllowance[owner][spender];
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        if (to == address(0)) revert TransferToZero();
        uint256 bal = _atnBalance[msg.sender];
        if (bal < amount) revert InsufficientATN(amount, bal);
        unchecked { _atnBalance[msg.sender] = bal - amount; }
        _atnBalance[to] += amount;
        emit ATNTransfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        _atnAllowance[msg.sender][spender] = amount;
        emit ATNApproval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        if (to == address(0)) revert TransferToZero();
        uint256 allowed = _atnAllowance[from][msg.sender];
        if (allowed < amount) revert InsufficientAllowance(amount, allowed);
        uint256 bal = _atnBalance[from];
        if (bal < amount) revert InsufficientATN(amount, bal);
        // Allowance decreases unless set to max (the unlimited-approval
        // convention used by ERC20 implementations everywhere).
        if (allowed != type(uint256).max) {
            unchecked { _atnAllowance[from][msg.sender] = allowed - amount; }
        }
        unchecked { _atnBalance[from] = bal - amount; }
        _atnBalance[to] += amount;
        emit ATNTransfer(from, to, amount);
        return true;
    }

    // =========================================================================
    // Inference payments (Phase 7.2)
    //
    // payForInference is a thin wrapper around transfer that emits a
    // structured event tagged with an off-chain request id. The event
    // lets the network audit which on-chain payments matched which
    // inference requests; the request id is opaque to the contract
    // (typically a sha256/keccak of the request body).
    //
    // No special pricing logic on chain. The price is whatever the
    // caller pays. Daemons advertise their price off-chain (libp2p
    // capability gossip); requesting agents call payForInference with
    // that amount before the serving agent agrees to serve.
    //
    // The contract does not enforce that the payment matches a real
    // inference request — that's the off-chain protocol's job. This
    // function just provides a labeled payment rail with audit-grade
    // event tagging.
    // =========================================================================

    event InferencePayment(
        address indexed payer,
        address indexed recipient,
        uint256 amount,
        bytes32 indexed requestId
    );

    /// @notice Pay a serving agent for an inference request.
    /// @param recipient The serving agent's address (the on-chain
    ///                  identity of the agent that handled the
    ///                  request — not the daemon process that
    ///                  carried the bytes).
    /// @param amount    ATN to transfer.
    /// @param requestId Opaque off-chain request id (e.g. keccak256
    ///                  of the inference request body) — emitted in
    ///                  the event so the request can be matched.
    function payForInference(
        address recipient,
        uint256 amount,
        bytes32 requestId
    ) external returns (bool) {
        if (recipient == address(0)) revert TransferToZero();
        uint256 bal = _atnBalance[msg.sender];
        if (bal < amount) revert InsufficientATN(amount, bal);
        unchecked { _atnBalance[msg.sender] = bal - amount; }
        _atnBalance[recipient] += amount;
        emit ATNTransfer(msg.sender, recipient, amount);
        emit InferencePayment(msg.sender, recipient, amount, requestId);
        return true;
    }
}
