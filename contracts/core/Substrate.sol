// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {EIP712} from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import {Checkpoints} from "@openzeppelin/contracts/utils/structs/Checkpoints.sol";

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
contract Substrate is EIP712 {
    using ECDSA for bytes32;

    /// @dev EIP712 domain: name + version. The domain separator folds in
    ///      chainId + this contract's address, so an owner-binding signature is
    ///      bound to THIS Substrate deployment on THIS chain — it cannot be
    ///      replayed against another Substrate instance or a fork.
    /// @param treasury_ The DAO treasury address receiving the treasury share
    ///        of the service fee (see payForService). Immutable — there is
    ///        no admin key to repoint it; changing it is a redeploy.
    /// @param vaultMinter_ The single privileged address (the DAO-repo
    ///        VentureVault) allowed to mint ATN for purchased funding via
    ///        mintFromVault. Immutable — no admin key to repoint it.
    ///        Zero address = no vault: mintFromVault always reverts and the
    ///        purchased-money feature is off.
    constructor(address treasury_, address vaultMinter_)
        EIP712("AutonetSubstrate", "1")
    {
        treasury = treasury_;
        vaultMinter = vaultMinter_;
    }

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
        // Merkle root over (agent address, scaled ATN amount, scaled
        // reputation amount) leaves for this epoch (v4.1: the leaf
        // commits BOTH decoupled ledgers). recordTrainingForEpoch
        // verifies the caller's claim against it — mint AND reputation
        // amounts are federation-ratified, not self-reported. Appended
        // last so prior tuple indexes hold.
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
    ///        (agent, scaledAmount, scaledRepAmount) mint map;
    ///        sorted-pair hashing, double-hashed leaves
    ///        (see recordTrainingForEpoch).
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
    // Owner-rooted registration (tool-substrate v2, docs/tool_substrate.md
    // §Owner-rooted registration)
    //
    // The AGENT is the only web3 entity; fleets root in a human WALLET (the
    // "owner"), never in an installation. The chain records *who and whose*:
    // (agent → owner) attributes an agent to an owner wallet, and
    // (agent → parent) records fleet topology. This is pure registration
    // data — recomputable by anyone, materialized off-chain into fleet trees.
    //
    // WHY CRYPTOGRAPHIC VERIFICATION. The off-chain mint damper excludes
    // attestations from agents under the SAME owner as a tool's author. If
    // owner were a self-declared field, a sybil operator could register many
    // agents each claiming a DIFFERENT fabricated owner wallet and thereby
    // evade the same-owner exclusion. So `owner` is proven: the owner wallet
    // must have signed an EIP-712 OwnerBinding over (agent, parent). The agent
    // key (msg.sender) countersigns implicitly by being the transaction sender.
    // Two independent keys must agree, exactly as intended: the human vouches
    // for the agent, the agent submits.
    //
    // OWNER BINDING IS AGENT-AUTHORIZED ROTATION, NOT WRITE-ONCE. The physical
    // host that holds the agent keys (msg.sender) is the root of trust. It can
    // (re)bind the fleet to any wallet that signs a fresh OwnerBinding. This
    // exists for KEY-LOSS RECOVERY: a human who leaks their owner-wallet key
    // recovers by re-binding every agent, from the daemon host, to a new
    // wallet. Crucially the OLD owner's signature is NOT required to rotate —
    // requiring it would hand a veto to the thief who stole the key. Custody of
    // the agent keys, not possession of the old owner key, authorizes rotation.
    //
    // REPLAY SAFETY. The OwnerBinding struct binds {agent, parent, nonce}; the
    // EIP712 domain separator binds {chainId, this contract}. Together these
    // make a signature usable ONLY to bind THIS agent to THIS owner (the
    // recovered signer) with THIS parent, at THIS binding nonce, on THIS
    // deployment.
    //
    // The nonce is NOW MANDATORY — the old "no nonce needed" argument is dead.
    // That argument leaned entirely on owner being write-once: once set, a
    // captured signature couldn't re-attribute the agent because the second
    // bind reverted. Dropping write-once to enable recovery resurrects
    // exactly that replay: a captured OLD owner-binding signature (e.g. the one
    // that first bound the now-leaked owner) could otherwise be replayed to
    // rotate the fleet BACK to the leaked wallet, undoing the recovery. The
    // per-agent `bindingNonce` closes this: it increments on every
    // successful bind/rotate, so every historical signature was signed against
    // a nonce that is now stale and no longer recovers to an accepted binding.
    // A fresh rotation requires the new owner to sign the CURRENT nonce, which
    // only a cooperating (new) owner can produce. `agent` is still msg.sender,
    // so a stolen signature can't be redirected to a different agent either.
    // =========================================================================

    /// @dev agent → owner wallet (address(0) = ownerless, the legacy
    ///      registerAgent path). Rotatable by the agent (custody of the
    ///      agent key authorizes re-binding to any wallet that signs).
    mapping(address => address) public agentOwner;
    /// @dev agent → parent agent in the fleet tree (address(0) = top-level).
    mapping(address => address) public agentParent;
    /// @dev agent → current binding nonce. Incremented on every successful
    ///      bind/rotate; the value a fresh OwnerBinding signature must target.
    mapping(address => uint256) public bindingNonce;

    /// @dev EIP-712 typehash for the owner's owner-binding signature.
    bytes32 private constant OWNER_BINDING_TYPEHASH =
        keccak256("OwnerBinding(address agent,address parent,uint256 nonce)");

    /// @dev Emitted on every bind AND every rotation. `prevOwner` is the owner
    ///      immediately before this binding (address(0) on first bind), so the
    ///      indexer reconstructs the full ownership history — including the
    ///      leaked→recovered transition — from the event stream alone. Chosen
    ///      over a separate AgentOwnerRotated event: one event type, one
    ///      subscription, and first-bind is just the prevOwner==0 case, so the
    ///      indexer needs no special-casing to fold rotations into fleet trees.
    event OwnerBound(
        address indexed agent,
        address indexed owner,
        address indexed parent,
        address prevOwner,
        uint256 nonce
    );

    error OwnerRequired();
    error BadOwnerBindingSignature();
    error ParentNotRegistered();
    error ParentOwnerMismatch();

    /// @notice EIP-712 digest the owner wallet signs to bind an agent.
    ///         Exposed for off-chain signers and tests. Bind the CURRENT
    ///         `bindingNonce(agent)` — a signature at a stale nonce will
    ///         not recover to an accepted binding.
    /// @param agent  The agent address being bound (must equal msg.sender
    ///               at binding time).
    /// @param parent The parent agent (address(0) for a top-level agent).
    /// @param nonce  The agent's binding nonce this signature is valid at.
    function bindingHash(address agent, address parent, uint256 nonce)
        public
        view
        returns (bytes32)
    {
        return _hashTypedDataV4(
            keccak256(abi.encode(OWNER_BINDING_TYPEHASH, agent, parent, nonce))
        );
    }

    /// @dev Verify the (new) owner's owner-binding signature and the parent/owner
    ///      constraints, then bind (msg.sender → owner, parent). Shared by the
    ///      fresh-registration, retrofit, and rotation paths. Consumes the
    ///      agent's current binding nonce (the signature must target it) and
    ///      increments it, so no signature is ever accepted twice. Caller must
    ///      have already ensured the agent is registered.
    function _bindOwner(
        address owner,
        address parentAgent,
        bytes calldata ownerSig
    ) private {
        if (owner == address(0)) revert OwnerRequired();

        // Recover the (new) owner's EIP-712 signature over
        // (this agent, parent, current-nonce). Binding the current nonce is
        // what makes a captured historical signature un-replayable: after any
        // successful bind the nonce has moved on, so old signatures recover to
        // a stale binding the contract no longer accepts.
        uint256 nonce = bindingNonce[msg.sender];
        bytes32 digestHash = bindingHash(msg.sender, parentAgent, nonce);
        address signer = digestHash.recover(ownerSig);
        if (signer != owner) revert BadOwnerBindingSignature();

        // Parent (if any) must be a registered agent under the SAME (new) owner
        // — the same-fleet constraint. A top-level agent passes parent = 0.
        // On rotation this is evaluated against the NEW owner: a fleet migrates
        // agent by agent, so a child rotated before its parent may temporarily
        // need parent = 0 (or a parent already rotated to the new owner).
        if (parentAgent != address(0)) {
            if (agents[parentAgent].registeredAt == 0) revert ParentNotRegistered();
            if (agentOwner[parentAgent] != owner) revert ParentOwnerMismatch();
        }

        address prevOwner = agentOwner[msg.sender];
        agentOwner[msg.sender] = owner;
        agentParent[msg.sender] = parentAgent;
        bindingNonce[msg.sender] = nonce + 1;
        emit OwnerBound(msg.sender, owner, parentAgent, prevOwner, nonce);
    }

    /// @notice Register the caller as a substrate agent WITH a verified owner
    ///         and optional parent (registerAgent v2). Identical registration
    ///         to registerAgent, plus the owner/parent attribution.
    /// @param lineageHash  Caller-supplied identity hash (see registerAgent).
    /// @param peerId       libp2p PeerId bytes (see registerAgent).
    /// @param owner        The owner human wallet. Must have signed the
    ///                     OwnerBinding — verified on-chain, never trusted as a
    ///                     bare claim.
    /// @param parentAgent  Optional parent agent (address(0) = top-level).
    ///                     If set, must be a registered agent whose owner is
    ///                     the same `owner` (same-fleet constraint).
    /// @param ownerSig     EIP-712 signature by `owner` over
    ///                     OwnerBinding{agent: msg.sender, parent: parentAgent,
    ///                     nonce: bindingNonce[msg.sender]} (0 for a fresh
    ///                     agent).
    function registerAgentBound(
        bytes32 lineageHash,
        bytes calldata peerId,
        address owner,
        address parentAgent,
        bytes calldata ownerSig
    ) external {
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

        // Fresh registration: nonce is 0, owner unset. The shared bind path
        // runs verification and the (agent→owner) binding.
        _bindOwner(owner, parentAgent, ownerSig);
    }

    /// @notice Attach owner + parent onto an ALREADY-registered agent — the
    ///         legacy-agent migration path (first bind) AND the recovery
    ///         rotation path (re-bind). Thin alias for rotateOwner: first
    ///         attach is simply a rotation from address(0). Retained as an
    ///         additive external API for callers that think in "attach" terms.
    /// @dev    msg.sender is the agent; it must already be registered. Whether
    ///         owner is currently set or not, the (new) owner's signature at
    ///         the current nonce authorizes the binding.
    function attachOwner(
        address owner,
        address parentAgent,
        bytes calldata ownerSig
    ) external {
        if (agents[msg.sender].registeredAt == 0) revert AgentNotActive();
        _bindOwner(owner, parentAgent, ownerSig);
    }

    /// @notice Rotate (or first-set) the owner+parent of the calling agent to
    ///         `newOwner`. Authorized by CUSTODY of the agent key (msg.sender)
    ///         plus the NEW owner's EIP-712 signature at the current nonce. The
    ///         OLD owner does NOT participate — this is the key-loss recovery
    ///         path, so requiring the (possibly leaked) old key would defeat
    ///         the purpose. First bind is the address(0)→newOwner case.
    ///
    ///         Per-agent rotation: children keep their own bindings. To migrate
    ///         a fleet after a key leak, rotate each agent; because a child's
    ///         parent must share the child's new owner, rotate parents before
    ///         children (or pass parent = 0 and re-parent afterward).
    /// @param newOwner   The new owner wallet (must sign; != address(0)).
    /// @param newParent  Parent under newOwner (registered & owned by newOwner)
    ///                    or address(0) for top-level.
    /// @param newOwnerSig EIP-712 signature by newOwner over
    ///                    OwnerBinding{agent: msg.sender, parent: newParent,
    ///                    nonce: bindingNonce[msg.sender]}.
    function rotateOwner(
        address newOwner,
        address newParent,
        bytes calldata newOwnerSig
    ) external {
        if (agents[msg.sender].registeredAt == 0) revert AgentNotActive();
        _bindOwner(newOwner, newParent, newOwnerSig);
    }

    /// @notice Read an agent's owner wallet (address(0) = unset).
    function getAgentOwner(address agent) external view returns (address) {
        return agentOwner[agent];
    }

    /// @notice Read an agent's parent in the fleet tree (address(0) = top-level
    ///         or owner unset).
    function getAgentParent(address agent) external view returns (address) {
        return agentParent[agent];
    }

    /// @notice True iff both agents have a set owner and it is the SAME owner.
    ///         The off-chain mint damper batch-reads this to exclude same-fleet
    ///         self-attestation. Returns false when EITHER owner is unset —
    ///         an unset owner is not "the same owner as" anything.
    function sameOwner(address agentA, address agentB) external view returns (bool) {
        address oa = agentOwner[agentA];
        if (oa == address(0)) return false;
        return oa == agentOwner[agentB];
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

    // Checkpointed reputation history (the IVotes MECHANISM without the
    // delegation layer, mirroring the ATN checkpoints below). Reputation
    // is the SOULBOUND voice measure: it only ever increases (one write
    // site — recordTrainingForEpoch) and never transfers, so unlike the
    // ATN pair there is exactly ONE checkpoint mutation per account, at
    // the mint. Historical reputation reads from CURRENT state:
    // reputationOfAt works on any node, no archive required. Used by the
    // federated close's voice weighting (nodes/common/voice_state.py) —
    // ratified 2026-07-08: ATN = money, reputation = voice, so voice
    // weight is priced from reputation, not balances.
    using Checkpoints for Checkpoints.Trace208;
    mapping(address => Checkpoints.Trace208) private _reputationHistory;
    Checkpoints.Trace208 private _reputationSupplyHistory;

    /// @notice Reputation alias for agentMintTotal. Same storage,
    ///         different semantic — reputation is the soulbound
    ///         contribution measure, never decreases, untransferable.
    function agentReputation(address agent) external view returns (uint256) {
        return agentMintTotal[agent];
    }

    /// @notice Reputation of `agent` as of the END of `blockNumber`
    ///         (latest checkpoint at or before it; 0 if none). Served
    ///         from current state — works on non-archive nodes. Same
    ///         upperLookup semantics as balanceOfAt, but the underlying
    ///         quantity is soulbound and monotonic (never decreases).
    function reputationOfAt(address agent, uint256 blockNumber)
        external
        view
        returns (uint256)
    {
        return _reputationHistory[agent].upperLookup(uint48(blockNumber));
    }

    /// @notice Total reputation (== networkMintTotal) as of the END of
    ///         `blockNumber`. The denominator for reputation-weighted
    ///         voice at a pinned snapshot block.
    function reputationTotalSupplyAt(uint256 blockNumber)
        external
        view
        returns (uint256)
    {
        return _reputationSupplyHistory.upperLookup(uint48(blockNumber));
    }

    /// @param amount   The ATN (money) minted in this training event.
    /// @param repAmount The reputation (soulbound voice) minted in this
    ///        training event. Distinct from ``amount`` since v4.1 "gradient
    ///        trust": the daemon-side close attributes ATN to ALL callers
    ///        (including zero-reputation ones), but reputation only to the
    ///        portion earned from reputation-holding callers, so
    ///        ``repAmount <= amount`` always (equality reproduces the
    ///        pre-v4.1 lockstep behavior). Kept as a distinct field rather
    ///        than reusing ``amount`` so indexers can separate the two
    ///        ledgers from the event stream.
    /// @param cumulativeForAgent The agent's cumulative reputation
    ///        (agentMintTotal / agentReputation) AFTER this event.
    event TrainingRecorded(
        address indexed agent,
        bytes32 indexed epochIdHash,
        uint256 amount,
        uint256 repAmount,
        uint256 cumulativeForAgent
    );

    error EpochNotAnchored(bytes32 epochIdHash);
    error AlreadySubmittedForEpoch(bytes32 epochIdHash);
    error MintRootMissing(bytes32 epochIdHash);
    error MintProofInvalid(bytes32 epochIdHash);
    error RepExceedsAmount(uint256 amount, uint256 repAmount);

    /// @notice Record this agent's authoritative mint for an
    ///         already-anchored epoch.
    /// @dev    Per-(agent, epochIdHash) idempotent. The epoch must
    ///         have been anchored via submitAnchor() first — this
    ///         binds the record to a federation-ratified close. The
    ///         ATN ``amount`` is NOT self-reported: it must verify
    ///         against the anchor's agentMintRoot, so the only
    ///         claimable ATN amount is the one the federation ratified.
    ///
    ///         v4.1 "gradient trust" — DECOUPLED LEDGERS. ATN (money) and
    ///         reputation (voice) are minted at potentially DIFFERENT
    ///         amounts in the same event. The daemon-side close computes
    ///         two numbers per agent: ``amount`` = ATN mint (weights in
    ///         usage from zero-reputation callers) and ``repAmount`` =
    ///         reputation mint (only the portion attributable to
    ///         reputation-holding callers). The invariant
    ///         ``repAmount <= amount`` is enforced on-chain: reputation
    ///         can never exceed the ATN earned in the same event, and
    ///         equality reproduces the pre-v4.1 lockstep behavior. This is
    ///         the sybil defense — mint must not grant the very resource
    ///         (reputation) that gates mint weight, or dust rings could
    ///         bootstrap voice from zero.
    /// @param amount The ATN (money) mint, integer-scaled (default ×1e6
    ///               at the agent-side submitter, see Phase 5.6 docs).
    /// @param repAmount The reputation (soulbound voice) mint, same scale
    ///               as ``amount``. Must be <= amount. Pass repAmount ==
    ///               amount for the legacy lockstep behavior.
    /// @param epochIdHash keccak256 of the epoch_id string.
    /// @param proof Merkle proof for the leaf
    ///              keccak256(bytes.concat(keccak256(abi.encode(
    ///              msg.sender, amount, repAmount)))) under the anchor's
    ///              agentMintRoot (sorted-pair hashing; a single-leaf
    ///              tree has root == leaf and an empty proof). The leaf
    ///              commits BOTH ledgers, so ``repAmount`` is
    ///              federation-ratified too: a claimant cannot inflate
    ///              its voice by self-reporting repAmount == amount when
    ///              the close attributed less — that leaf doesn't prove.
    ///              The repAmount <= amount require above stays as
    ///              defense-in-depth on top of the proof.
    function recordTrainingForEpoch(
        uint256 amount,
        uint256 repAmount,
        bytes32 epochIdHash,
        bytes32[] calldata proof
    ) external {
        if (!agents[msg.sender].active) revert AgentNotActive();
        // Reputation can never exceed the ATN earned in the same event
        // (v4.1: rep is the rep-holder-attributable SUBSET of the ATN
        // mint). Enforced before the zero-amount no-op so a caller can't
        // slip a positive repAmount past a zero amount either.
        if (repAmount > amount) revert RepExceedsAmount(amount, repAmount);
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
        // (repAmount <= amount == 0, so nothing to mint on either side.)
        if (amount == 0) {
            return;
        }

        bytes32 root = anchors[anchorIndexByEpochId[epochId] - 1].agentMintRoot;
        if (root == bytes32(0)) revert MintRootMissing(epochIdHash);
        // v4.1: the leaf commits (agent, amount, repAmount) — BOTH the
        // money and the voice amounts are federation-ratified. Same
        // double-hashed OpenZeppelin leaf convention, one more field in
        // the abi.encode (mirrored by nodes/common/mint_merkle.py).
        bytes32 leaf = keccak256(
            bytes.concat(keccak256(abi.encode(msg.sender, amount, repAmount)))
        );
        if (!_verifyMintProof(proof, root, leaf)) {
            revert MintProofInvalid(epochIdHash);
        }

        // mintForEpoch records the ATN amount (what the merkle proof
        // committed). Reputation is the decoupled repAmount below.
        mintForEpoch[msg.sender][epochIdHash] = amount;
        // Reputation ledger (soulbound voice) accrues repAmount.
        agentMintTotal[msg.sender] += repAmount;
        networkMintTotal += repAmount;
        // The Agent struct's totalTrainingMint tracks reputation
        // (soulbound contribution), so it too moves by repAmount.
        agents[msg.sender].totalTrainingMint += repAmount;
        agents[msg.sender].trainingSubmissionCount += 1;

        // Checkpoint the soulbound reputation ledger at this block. This
        // is the ONLY reputation checkpoint write site — reputation never
        // decreases and never transfers, so there is nothing to mirror on
        // any other path (mintFromVault deliberately mints ATN only). The
        // federated close reads reputationOfAt / reputationTotalSupplyAt
        // at the previous anchor block to price voice weights. We push
        // even when repAmount == 0 (amount > 0) so the supply checkpoint
        // stays block-aligned; the pushed values are simply unchanged.
        _reputationHistory[msg.sender].push(
            uint48(block.number), uint208(agentMintTotal[msg.sender])
        );
        _reputationSupplyHistory.push(
            uint48(block.number), uint208(networkMintTotal)
        );

        // v4.1: training mints the two ledgers at DECOUPLED amounts.
        // Reputation (agentMintTotal, bumped above by repAmount) is
        // soulbound voice. ATN balance is transferable money, minted at
        // the full `amount` — the agent can spend it on inference fees.
        _mintATN(msg.sender, amount);

        emit TrainingRecorded(
            msg.sender,
            epochIdHash,
            amount,
            repAmount,
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

    // Checkpointed balance history (the IVotes MECHANISM without the
    // delegation layer — deliberate: getPastVotes-style semantics
    // return 0 for accounts that never delegate, which would mute
    // every wallet by default; the substrate's voice weighting needs
    // past BALANCES, not past votes). Every balance mutation pushes a
    // (block, balance) checkpoint, so historical balances are read
    // from CURRENT state: balanceOfAt works on any node, no archive
    // state required. Used by the federated close's voice weighting,
    // which prices each household's review/usage credit from balances
    // at the previous epoch's anchor block (nodes/common/voice_state.py).
    // (``using Checkpoints for Checkpoints.Trace208`` is declared once,
    // with the reputation history above.)
    mapping(address => Checkpoints.Trace208) private _atnBalanceHistory;
    Checkpoints.Trace208 private _atnSupplyHistory;

    /// @dev Push the account's CURRENT balance as a checkpoint at this
    ///      block. Same-block pushes overwrite (Trace208 semantics), so
    ///      multiple transfers in one block leave one final checkpoint.
    ///      uint208 cannot overflow: balances are bounded by total mint
    ///      volume (1e6-scaled federation amounts).
    function _checkpointATN(address account) private {
        _atnBalanceHistory[account].push(
            uint48(block.number), uint208(_atnBalance[account])
        );
    }

    /// @dev The one ATN mint primitive: bumps balance + supply, checkpoints
    ///      both, and emits the canonical mint transfer (from address(0)).
    ///      Deliberately mints ATN ONLY — it does NOT touch reputation
    ///      (agentMintTotal). Callers that also mint reputation (the
    ///      training path) bump it themselves; purchased-money paths
    ///      (mintFromVault) must not, because ATN is money and reputation
    ///      is earned voice (ratified 2026-07-08).
    function _mintATN(address to, uint256 amount) private {
        _atnBalance[to] += amount;
        atnTotalSupply += amount;
        _checkpointATN(to);
        _atnSupplyHistory.push(uint48(block.number), uint208(atnTotalSupply));
        emit ATNTransfer(address(0), to, amount);
    }

    event ATNTransfer(address indexed from, address indexed to, uint256 amount);
    event ATNApproval(address indexed owner, address indexed spender, uint256 amount);

    error InsufficientATN(uint256 requested, uint256 available);
    error InsufficientAllowance(uint256 requested, uint256 available);
    error TransferToZero();

    function balanceOf(address agent) external view returns (uint256) {
        return _atnBalance[agent];
    }

    /// @notice ATN balance as of the END of `blockNumber` (latest
    ///         checkpoint at or before it; 0 if none). Served from
    ///         current state — works on non-archive nodes.
    function balanceOfAt(address agent, uint256 blockNumber)
        external
        view
        returns (uint256)
    {
        return _atnBalanceHistory[agent].upperLookup(uint48(blockNumber));
    }

    /// @notice Total ATN supply as of the END of `blockNumber`.
    function atnTotalSupplyAt(uint256 blockNumber)
        external
        view
        returns (uint256)
    {
        return _atnSupplyHistory.upperLookup(uint48(blockNumber));
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
        _checkpointATN(msg.sender);
        _checkpointATN(to);
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
        _checkpointATN(from);
        _checkpointATN(to);
        emit ATNTransfer(from, to, amount);
        return true;
    }

    // =========================================================================
    // Service payments (Phase 7.2)
    //
    // payForService is a thin wrapper around transfer that emits a
    // structured event tagged with an off-chain request id. The event
    // lets the network audit which on-chain payments matched which
    // service requests; the request id is opaque to the contract
    // (typically a sha256/keccak of the request body).
    //
    // No special pricing logic on chain. The price is whatever the
    // caller pays. Daemons advertise their price off-chain (libp2p
    // capability gossip); requesting agents call payForService with
    // that amount before the serving agent agrees to serve.
    //
    // The contract does not enforce that the payment matches a real
    // service request — that's the off-chain protocol's job. This
    // function just provides a labeled payment rail with audit-grade
    // event tagging.
    // =========================================================================

    event ServicePayment(
        address indexed payer,
        address indexed recipient,
        uint256 amount,
        bytes32 indexed requestId
    );

    // -------------------------------------------------------------------------
    // Service fee (fee-recycled emission, docs/epoch_economics.md).
    //
    // A fee on every labeled service payment, split two ways:
    //   - the BURNED share leaves supply here and re-enters as the recycled
    //     component of the next epoch's emission pool (the federated close
    //     reads ServiceFee logs between anchors) — burn-and-remint is
    //     recycling implemented on the existing mint rail, and it is what
    //     makes volume-linked emission wash-proof: pumping volume pays real
    //     fees into a pool you only ever share pro-rata;
    //   - the TREASURY share transfers to the immutable DAO treasury.
    //
    // PROVISIONAL VALUES pending user blessing (economics parameters).
    // -------------------------------------------------------------------------

    /// @notice Fee on payForService, in basis points of the gross amount.
    uint16 public constant SERVICE_FEE_BPS = 250;      // 2.5%
    /// @notice Share OF THE FEE that goes to the treasury; the rest burns.
    uint16 public constant FEE_TREASURY_BPS = 5000;    // half/half
    uint16 public constant BPS_DENOM = 10_000;

    /// @notice DAO treasury (set at construction; no admin key to repoint).
    address public immutable treasury;

    // -------------------------------------------------------------------------
    // Vault minter (purchased-money rail).
    //
    // ATN has exactly two mint origins: EARNED voice via
    // recordTrainingForEpoch (mints reputation + ATN in lockstep), and
    // PURCHASED money via mintFromVault (mints ATN ONLY). The two are kept
    // strictly separate on purpose (ratified 2026-07-08: ATN = money,
    // reputation = voice). A venture backer who buys ATN through the
    // DAO-repo VentureVault must never gain soulbound reputation from that
    // purchase — money can be bought, voice must be earned.
    //
    // vaultMinter is the single privileged address (the VentureVault
    // contract) allowed to call mintFromVault. It is immutable — there is
    // no admin key to repoint it; changing it is a redeploy. Zero address =
    // no vault: the feature is off and mintFromVault always reverts.
    // -------------------------------------------------------------------------

    /// @notice The single address allowed to mint purchased ATN via
    ///         mintFromVault (the DAO-repo VentureVault). Immutable;
    ///         zero address = feature off.
    address public immutable vaultMinter;

    /// @notice Emitted when the vault mints purchased ATN. Distinct from
    ///         ATNTransfer(address(0), ...) so indexers can separate
    ///         purchased money from earned mint. No reputation is minted.
    event VaultMint(address indexed to, uint256 amount);

    error NotVaultMinter();
    error VaultDisabled();

    /// @notice Mint purchased ATN to `to`. Callable ONLY by the immutable
    ///         vaultMinter (the DAO-repo VentureVault, through a minimal
    ///         IAutonetATN interface).
    /// @dev Mints ATN ONLY — balance, supply, and checkpoints move exactly
    ///      like the recordTrainingForEpoch mint path, but reputation
    ///      (agentMintTotal) is deliberately NOT touched: this is purchased
    ///      money, not earned voice (ratified 2026-07-08). If vaultMinter is
    ///      the zero address the feature is off and this always reverts.
    /// @param to     Recipient of the purchased ATN.
    /// @param amount ATN to mint.
    function mintFromVault(address to, uint256 amount) external {
        if (vaultMinter == address(0)) revert VaultDisabled();
        if (msg.sender != vaultMinter) revert NotVaultMinter();
        if (to == address(0)) revert TransferToZero();
        _mintATN(to, amount);
        emit VaultMint(to, amount);
    }

    /// @notice Emitted per fee collection. `burned` is what the federated
    ///         close sums into the next epoch's recycled emission pool.
    event ServiceFee(
        address indexed payer,
        address indexed recipient,
        uint256 amount,
        uint256 burned,
        uint256 toTreasury
    );

    /// @notice Pay a serving agent for a service request.
    /// @param recipient The serving agent's address (the on-chain
    ///                  identity of the agent that handled the
    ///                  request — not the daemon process that
    ///                  carried the bytes).
    /// @param amount    ATN to transfer.
    /// @param requestId Opaque off-chain request id (e.g. keccak256
    ///                  of the service request body) — emitted in
    ///                  the event so the request can be matched.
    function payForService(
        address recipient,
        uint256 amount,
        bytes32 requestId
    ) external returns (bool) {
        if (recipient == address(0)) revert TransferToZero();
        uint256 bal = _atnBalance[msg.sender];
        if (bal < amount) revert InsufficientATN(amount, bal);

        // Service fee: burned share exits supply (recycled into the next
        // epoch's emission pool by the close), treasury share transfers.
        uint256 fee = (amount * SERVICE_FEE_BPS) / BPS_DENOM;
        uint256 toTreasury = treasury != address(0)
            ? (fee * FEE_TREASURY_BPS) / BPS_DENOM
            : 0;
        uint256 burned = fee - toTreasury;
        uint256 net = amount - fee;

        unchecked { _atnBalance[msg.sender] = bal - amount; }
        _atnBalance[recipient] += net;
        _checkpointATN(msg.sender);
        _checkpointATN(recipient);
        if (toTreasury > 0) {
            _atnBalance[treasury] += toTreasury;
            _checkpointATN(treasury);
            emit ATNTransfer(msg.sender, treasury, toTreasury);
        }
        if (burned > 0) {
            atnTotalSupply -= burned;
            _atnSupplyHistory.push(
                uint48(block.number), uint208(atnTotalSupply)
            );
        }

        emit ATNTransfer(msg.sender, recipient, net);
        emit ServicePayment(msg.sender, recipient, amount, requestId);
        if (fee > 0) {
            emit ServiceFee(msg.sender, recipient, amount, burned, toTreasury);
        }
        return true;
    }
}
