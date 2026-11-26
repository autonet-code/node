// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {EIP712} from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import {Checkpoints} from "@openzeppelin/contracts/utils/structs/Checkpoints.sol";
import {SafeCast} from "@openzeppelin/contracts/utils/math/SafeCast.sol";
import {Strings} from "@openzeppelin/contracts/utils/Strings.sol";
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {ReentrancyGuardTransient} from "@openzeppelin/contracts/utils/ReentrancyGuardTransient.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/// @dev The DAO Registry's read surface: every parameter lives there,
///      the value index among them (jurisdiction.parity.<token>).
interface IRegistryRead {
    function getRegistryValue(string memory key) external view returns (string memory);
}

/// @title Substrate: a jurisdiction's chain surface for the tool economy.
///
/// Four concerns that share state:
///
///   1. **Epoch anchoring.** Each epoch's close is anchored here (by the
///      governor, or the AnchorElection as the anchorer): hash-chained,
///      cutoffs strictly increasing, with the merkle root of the epoch's
///      claims.
///   2. **Agents and tools.** The agent registry with owner binding, and the
///      authorship of published tools.
///   3. **Service payments in the value index's tokens.** A payment is made
///      in any token the jurisdiction's value index lists (a non-zero
///      Registry `jurisdiction.parity.<token>`). Each takes the service fee:
///      `feeTreasuryBps` of it is owed to the treasury and the rest, the tool
///      share, fills that token's pool. Nothing is burned and nothing is
///      minted: Substrate is not a token.
///   4. **Claims from the pools.** Each epoch's close commits, per earner,
///      an amount of every token's pool (the same fraction of each, the
///      basket). A claim credits the amounts, adds their value, at the
///      parities stored when the epoch was anchored, to `earnedValue`,
///      which the DAO's REP claim reads, and pays the whole basket in the
///      same transaction. A token whose transfer fails (paused, frozen)
///      stays owed and is withdrawn on its own later, so it blocks nothing
///      else. When the claims of an epoch are worth less than the REP
///      floor (Registry `jurisdiction.autonet.rep_floor`, stored at each
///      anchor for the epoch after it), each leaf also carries a seed, its
///      part of the difference, added to `earnedValue` too; no epoch's
///      seeds total more than the floor stored at the anchor before it.
///
/// Custody. Substrate holds only the pools and what is owed from them
/// (earners' claims, the treasury's share). Channel deposits stay in
/// PaymentChannel and bonds in ToolBond. Every token call is checked
/// (SafeERC20), every inflow must arrive in full (a token that takes a cut
/// is refused), every function that moves a token is guarded against
/// reentry, and the books are Substrate's own: `balanceOf` is read only to
/// measure inflows and to sweep what was sent outside a payment.
contract Substrate is EIP712, Initializable, UUPSUpgradeable, ReentrancyGuardTransient {
    using ECDSA for bytes32;
    using SafeERC20 for IERC20;
    using Checkpoints for Checkpoints.Trace208;

    // -------------------------------------------------------------------------
    // Deployment: one implementation, one proxy per jurisdiction
    //
    // A jurisdiction's Substrate is an ERC-1967 proxy over an implementation
    // deployed once, set up by initialize() in the transaction that creates
    // the proxy. Its governor (the DAO timelock) can point it at new code
    // (upgradeToAndCall), which keeps every pool, claim, anchor and
    // registration, until it calls renounceUpgrades(), after which nobody
    // can. The values initialize() sets have no setter, except the anchorer
    // and the fee settings, which the governor retunes.
    //
    // The constructor takes the same arguments and sets up the instance it
    // creates, so a Substrate deployed directly works as a plain contract
    // (local chains, tests) and can never be initialized again: that is
    // also what locks the implementation behind the proxies.
    // -------------------------------------------------------------------------

    /// @param treasury_ The DAO treasury: the fee's treasury share and any
    ///        token sent here outside a payment are owed to it. Zero = no
    ///        treasury share (the whole fee is the tool share).
    /// @param governor_ The DAO timelock: anchors, sets the anchorer, retunes
    ///        the fee, upgrades a proxy. Non-zero.
    /// @param anchorer_ The jurisdiction's AnchorElection, or address(0).
    /// @param registry_ The DAO Registry, where the value index lives.
    ///        Non-zero.
    constructor(
        address treasury_,
        address governor_,
        address anchorer_,
        address registry_,
        uint16 serviceFeeBps_,
        uint16 feeTreasuryBps_
    )
        EIP712("AutonetSubstrate", "1")
        initializer
    {
        _init(treasury_, governor_, anchorer_, registry_, serviceFeeBps_, feeTreasuryBps_);
    }

    /// @notice Set up a proxy's Substrate: the constructor's arguments, in
    ///         the same order. Once per proxy.
    function initialize(
        address treasury_,
        address governor_,
        address anchorer_,
        address registry_,
        uint16 serviceFeeBps_,
        uint16 feeTreasuryBps_
    ) external initializer {
        _init(treasury_, governor_, anchorer_, registry_, serviceFeeBps_, feeTreasuryBps_);
    }

    /// @notice True once the governor gave up upgrading this proxy.
    bool public upgradesRenounced;

    event UpgradesRenounced();

    error UpgradesAreRenounced();

    error NotDuringUpgrade();

    /// @dev Transient flag: set while an authorized upgrade runs, so
    ///      renounceUpgrades can only be the call of an upgradeToAndCall.
    bytes32 private constant _UPGRADING_SLOT =
        keccak256("autonet.substrate.upgrading");

    /// @dev Only the governor (the DAO timelock) may point a proxy at new
    ///      code, and only until it renounces upgrades. The DAO's governor
    ///      treats upgradeToAndCall as constitutional, by its selector.
    function _authorizeUpgrade(address) internal override {
        if (msg.sender != governor) revert NotGovernor();
        if (upgradesRenounced) revert UpgradesAreRenounced();
        bytes32 slot = _UPGRADING_SLOT;
        assembly ("memory-safe") { tstore(slot, 1) }
    }

    /// @notice Give up upgrades for good. One way, and only as the call of
    ///         an upgradeToAndCall by the governor (to the same or a new
    ///         implementation), so it takes the same constitutional proposal
    ///         as an upgrade.
    function renounceUpgrades() external {
        if (msg.sender != governor) revert NotGovernor();
        bytes32 slot = _UPGRADING_SLOT;
        uint256 upgrading;
        assembly ("memory-safe") { upgrading := tload(slot) }
        if (upgrading == 0) revert NotDuringUpgrade();
        upgradesRenounced = true;
        emit UpgradesRenounced();
    }

    function _init(
        address treasury_,
        address governor_,
        address anchorer_,
        address registry_,
        uint16 serviceFeeBps_,
        uint16 feeTreasuryBps_
    ) private {
        if (governor_ == address(0)) revert NoAnchorAuthority();
        if (registry_ == address(0)) revert RegistryRequired();
        if (serviceFeeBps_ < MIN_SERVICE_FEE_BPS || serviceFeeBps_ > MAX_SERVICE_FEE_BPS) {
            revert FeeOutOfBounds(serviceFeeBps_);
        }
        if (feeTreasuryBps_ > MAX_FEE_TREASURY_BPS) revert FeeOutOfBounds(feeTreasuryBps_);
        serviceFeeBps = serviceFeeBps_;
        feeTreasuryBps = feeTreasuryBps_;
        treasury = treasury_;
        governor = governor_;
        registry = registry_;
        deployedAt = block.timestamp;
        latestCutoff = block.timestamp;
        anchorer = anchorer_;
        emit AnchorerSet(address(0), anchorer_);
    }

    // =========================================================================
    // Anchor records
    // =========================================================================

    struct Anchor {
        string epochId;
        bytes32 epochRoot;
        bytes32 prevEpochRoot;
        bytes32 prevAnchorHash;
        // sha256 of the epoch's claim blob (the close's payload of claims),
        // served peer to peer; not an IPFS CID.
        string claimCid;
        bytes32 payloadHash;
        address submitter;
        uint256 blockNumber;
        uint256 timestamp;
        // Merkle root of the epoch's leaves: one claim per earner (see
        // claim) and, for ToolBond, one leaf per concluded tool, each type
        // under its own tag.
        bytes32 claimRoot;
        // The epoch's batches are those signed after the previous anchor's
        // cutoff and at or before this one; its fees are those collected in
        // the same span.
        uint256 cutoff;
    }

    Anchor[] private anchors;
    mapping(string => uint256) private anchorIndexByEpochId;  // 1-indexed
    mapping(bytes32 => string) private epochIdByHash;
    bytes32 public latestAnchorHash;
    bytes32 public latestEpochRoot;

    event EpochAnchored(
        string indexed epochId,
        bytes32 indexed epochIdHash,
        bytes32 indexed epochRoot,
        bytes32 prevEpochRoot,
        bytes32 prevAnchorHash,
        bytes32 payloadHash,
        string claimCid,
        address submitter,
        uint256 blockNumber
    );

    error EpochIdEmpty();
    error EpochAlreadyAnchored(string epochId);
    error PrevEpochRootMismatch(bytes32 expected, bytes32 got);
    error PrevAnchorHashMismatch(bytes32 expected, bytes32 got);
    error NoAnchorAuthority();
    error RegistryRequired();
    error CutoffOutOfOrder(uint256 latest, uint256 got);

    event AnchorerSet(address indexed oldAnchorer, address indexed newAnchorer);

    /// @notice Deployment timestamp: the REP snapshot for the first anchor.
    uint256 public deployedAt;

    /// @notice The contract allowed to anchor besides the governor: the
    ///         jurisdiction's AnchorElection. Set by the governor;
    ///         address(0) = governor only.
    address public anchorer;

    /// @notice The latest anchored epoch's cutoff (deployedAt before any).
    uint256 public latestCutoff;

    /// @notice The cutoff the pools are capped at: the cutoff of the anchor
    ///         before the latest one (0 until two epochs are anchored). Claims
    ///         of a token can never total more than its tool share collected
    ///         up to it, so the pool an epoch allocates is the previous
    ///         epoch's fees (lag one), and whatever a close leaves unallocated
    ///         carries over.
    uint256 public poolCutoff;

    /// @notice Set the anchorer (the AnchorElection). Governor only.
    function setAnchorer(address newAnchorer) external {
        if (msg.sender != governor) revert NotGovernor();
        emit AnchorerSet(anchorer, newAnchorer);
        anchorer = newAnchorer;
    }

    /// @notice The REP timepoint the close weighs the NEXT epoch at (the
    ///         previous anchor's timestamp; deployedAt before any anchor).
    function anchorSnapshotTime() public view returns (uint256) {
        uint256 n = anchors.length;
        return n == 0 ? deployedAt : anchors[n - 1].timestamp;
    }

    /// @notice Anchor a network epoch close. Governor or the anchorer only.
    /// @param claimRoot Merkle root of the epoch's leaves (sorted-pair
    ///        hashing, double-hashed leaves; see claim).
    /// @param cutoff The epoch's cutoff: after the latest one and before
    ///        this block, so every fee of the epoch is already counted.
    function anchor(
        string calldata epochId,
        bytes32 epochRoot,
        bytes32 prevEpochRoot,
        bytes32 prevAnchorHash,
        string calldata claimCid,
        bytes32 payloadHash,
        bytes32 claimRoot,
        uint256 cutoff
    ) external {
        if (msg.sender != governor
                && (anchorer == address(0) || msg.sender != anchorer)) {
            revert NotGovernor();
        }
        if (cutoff <= latestCutoff || cutoff >= block.timestamp) {
            revert CutoffOutOfOrder(latestCutoff, cutoff);
        }
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

        // The pools may now pay out the fees up to the previous cutoff.
        if (anchors.length > 0) poolCutoff = latestCutoff;
        latestCutoff = cutoff;

        Anchor storage a = anchors.push();
        a.epochId = epochId;
        a.epochRoot = epochRoot;
        a.prevEpochRoot = prevEpochRoot;
        a.prevAnchorHash = prevAnchorHash;
        a.claimCid = claimCid;
        a.payloadHash = payloadHash;
        a.submitter = msg.sender;
        a.blockNumber = block.number;
        a.timestamp = block.timestamp;
        a.claimRoot = claimRoot;
        a.cutoff = cutoff;
        _index(a);
        _storeParities(anchors.length - 1);
        // Stored for the NEXT epoch: its close reads the floor pinned to
        // this anchor, and its seeds are capped at this value.
        uint256 floor = repFloorValue();
        if (floor != 0) _anchorFloor[anchors.length - 1] = floor;
    }

    /// @dev Store every pool token's parity as the epoch is anchored: the
    ///      price its claims are valued at.
    function _storeParities(uint256 index) private {
        uint256 n = _poolTokens.length;
        for (uint256 i = 0; i < n; i++) {
            address t = _poolTokens[i];
            uint256 p = parityOf(t);
            if (p != 0) _anchorParity[index][t] = p;
        }
    }

    /// @dev Index the anchor just stored, chain to it, and announce it.
    function _index(Anchor storage a) private {
        bytes32 epochIdHash = keccak256(bytes(a.epochId));
        anchorIndexByEpochId[a.epochId] = anchors.length;
        epochIdByHash[epochIdHash] = a.epochId;
        // Content only: no sender or block number, so the next epoch's
        // prevAnchorHash is known before this anchor lands.
        latestAnchorHash = _hashStored(a);
        latestEpochRoot = a.epochRoot;
        emit EpochAnchored(
            a.epochId, epochIdHash, a.epochRoot, a.prevEpochRoot, a.prevAnchorHash,
            a.payloadHash, a.claimCid, msg.sender, block.number
        );
    }

    /// @notice The hash an anchor with these fields chains to.
    function anchorHashOf(
        string calldata epochId,
        bytes32 epochRoot,
        bytes32 prevEpochRoot,
        bytes32 prevAnchorHash,
        string calldata claimCid,
        bytes32 payloadHash,
        bytes32 claimRoot,
        uint256 cutoff
    ) public pure returns (bytes32) {
        return keccak256(abi.encode(
            epochId, epochRoot, prevEpochRoot, prevAnchorHash, claimCid,
            payloadHash, claimRoot, cutoff));
    }

    function _hashStored(Anchor storage a) private view returns (bytes32) {
        return keccak256(abi.encode(
            a.epochId, a.epochRoot, a.prevEpochRoot, a.prevAnchorHash,
            a.claimCid, a.payloadHash, a.claimRoot, a.cutoff));
    }

    function anchorCount() external view returns (uint256) {
        return anchors.length;
    }

    function getAnchor(uint256 index) external view returns (Anchor memory) {
        return anchors[index];
    }

    /// @notice The anchor's block timestamp (the next epoch's REP snapshot).
    function anchorTimestamp(uint256 index) external view returns (uint256) {
        return anchors[index].timestamp;
    }

    /// @notice The anchor's merkle root (claims and concluded tools).
    function anchorRoot(uint256 index) external view returns (bytes32) {
        return anchors[index].claimRoot;
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
    function isAnchoredByHash(bytes32 epochIdHash) external view returns (bool) {
        return bytes(epochIdByHash[epochIdHash]).length > 0;
    }

    // =========================================================================
    // Agent registration
    // =========================================================================

    struct Agent {
        uint256 registeredAt;
        bool active;
    }

    mapping(address => Agent) public agents;
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
        bytes peerId,
        uint256 timestamp
    );

    event PeerIdUpdated(address indexed agent, bytes peerId);

    /// @notice Emitted when an agent's browser-reachable wss endpoint changes.
    ///         Indexers mirror this to the off-chain agent directory.
    event EndpointUpdated(address indexed agent, string wsEndpoint);

    error AlreadyRegistered();
    error AgentNotActive();
    error PeerIdRequired();

    /// @notice Update this agent's libp2p PeerId. Used when a daemon
    ///         rotates its libp2p keypair (e.g. fresh install on a new
    ///         box) without losing on-chain identity / earnings.
    function updatePeerId(bytes calldata peerId) external {
        if (!isLive(msg.sender)) revert AgentNotActive();
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
        if (!isLive(msg.sender)) revert AgentNotActive();
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

    /// @dev agent → owner wallet. Every registered agent has one (there is
    ///      no unbound registration). Rotatable by the agent (custody of the
    ///      agent key authorizes re-binding to any wallet that signs).
    mapping(address => address) public agentOwner;
    /// @dev agent → parent agent in the fleet tree (address(0) = top-level).
    mapping(address => address) public agentParent;
    /// @dev agent → current binding nonce. Incremented on every successful
    ///      bind/rotate; the value a fresh OwnerBinding signature must target.
    mapping(address => uint256) public bindingNonce;
    /// @dev agent → depth in the fleet tree (0 = top-level). Bounds the
    ///      ancestor walk in ``isLive``.
    mapping(address => uint256) public agentDepth;

    /// @notice Deepest allowed fleet tree (a top-level agent is depth 0).
    uint256 public constant MAX_FLEET_DEPTH = 16;

    /// @dev EIP-712 typehash for the owner's revocation of an agent subtree.
    bytes32 private constant REVOKE_TYPEHASH =
        keccak256("RevokeAgent(address agent)");

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

    /// @dev The owner revoked `agent`. Every descendant (via the OwnerBound
    ///      parent links) is revoked with it: ``isLive`` walks the ancestors.
    ///      Revocation is final.
    event AgentRevoked(address indexed agent, address indexed owner);

    error OwnerRequired();
    error BadOwnerBindingSignature();
    error ParentNotRegistered();
    error ParentOwnerMismatch();
    error FleetTooDeep();
    error FleetCycle();
    error BadRevokeSignature();

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

    /// @dev Verify the binding signature and the parent/owner constraints, then
    ///      bind (msg.sender → owner, parent). Shared by registration and
    ///      rotation. Consumes the agent's current binding nonce (the signature
    ///      must target it) and increments it, so no signature is ever
    ///      accepted twice. Caller must have already ensured the agent is
    ///      registered.
    ///
    ///      Signer: the owner wallet, or (``parentMaySign``, registration only)
    ///      the live parent agent's key. A parent-signed child lands in the
    ///      parent's household, so agents can spawn registered children
    ///      without a human, and spawning never adds a household.
    function _bindOwner(
        address owner,
        address parentAgent,
        bytes calldata sig,
        bool parentMaySign
    ) private {
        if (owner == address(0)) revert OwnerRequired();

        // Parent (if any) must be a live agent under the SAME owner, the
        // same-fleet constraint. A top-level agent passes parent = 0.
        uint256 depth = 0;
        if (parentAgent != address(0)) {
            if (agents[parentAgent].registeredAt == 0) revert ParentNotRegistered();
            if (!isLive(parentAgent)) revert AgentNotActive();
            if (agentOwner[parentAgent] != owner) revert ParentOwnerMismatch();
            // The parent's depth is counted by walking up from it, never
            // read from agentDepth: a rotation re-parents one agent and
            // leaves its descendants' recorded depths stale. The walk also
            // refuses a cycle (binding under one's own descendant).
            address a = parentAgent;
            while (a != address(0)) {
                if (a == msg.sender) revert FleetCycle();
                depth += 1;
                if (depth > MAX_FLEET_DEPTH) revert FleetTooDeep();
                a = agentParent[a];
            }
        }

        // Recover the signature over (this agent, parent, current nonce).
        // Binding the current nonce makes a captured historical signature
        // un-replayable: after any bind the nonce has moved on.
        uint256 nonce = bindingNonce[msg.sender];
        address signer = bindingHash(msg.sender, parentAgent, nonce).recover(sig);
        if (signer != owner
            && !(parentMaySign && parentAgent != address(0) && signer == parentAgent)) {
            revert BadOwnerBindingSignature();
        }

        address prevOwner = agentOwner[msg.sender];
        agentOwner[msg.sender] = owner;
        agentParent[msg.sender] = parentAgent;
        agentDepth[msg.sender] = depth;
        bindingNonce[msg.sender] = nonce + 1;
        emit OwnerBound(msg.sender, owner, parentAgent, prevOwner, nonce);
    }

    /// @notice Register the caller as a substrate agent, bound to its owner
    ///         wallet and optional parent. The only registration path.
    /// @param peerId       libp2p PeerId bytes for off-chain discovery.
    ///                     Daemons resolve it to a current multiaddr via the
    ///                     libp2p DHT: the chain stores who, the DHT where.
    /// @param owner        The owner human wallet (the household).
    /// @param parentAgent  Optional parent agent (address(0) = top-level).
    ///                     If set, must be a live agent of the same `owner`.
    /// @param sig          EIP-712 OwnerBinding{agent: msg.sender, parent:
    ///                     parentAgent, nonce: 0}, signed by `owner`, or by
    ///                     `parentAgent` when it is set.
    function registerAgentBound(
        bytes calldata peerId,
        address owner,
        address parentAgent,
        bytes calldata sig
    ) external {
        if (agents[msg.sender].registeredAt != 0) revert AlreadyRegistered();
        if (peerId.length == 0) revert PeerIdRequired();

        agents[msg.sender] = Agent({
            registeredAt: block.timestamp,
            active: true
        });
        agentPeerId[msg.sender] = peerId;
        registeredAgentList.push(msg.sender);

        emit AgentRegistered(msg.sender, peerId, block.timestamp);

        _bindOwner(owner, parentAgent, sig, true);
    }

    /// @notice Rotate the owner+parent of the calling agent to `newOwner`.
    ///         Authorized by CUSTODY of the agent key (msg.sender) plus the
    ///         NEW owner's EIP-712 signature at the current nonce. The OLD
    ///         owner does NOT participate: this is the key-loss recovery path,
    ///         so requiring the (possibly leaked) old key would defeat it.
    ///         A revoked agent (or one under a revoked ancestor) cannot rotate,
    ///         so revocation can't be escaped by re-binding.
    ///
    ///         Per-agent rotation: children keep their own bindings. To migrate
    ///         a fleet after a key leak, rotate each agent; because a child's
    ///         parent must share the child's new owner, rotate parents before
    ///         children (or pass parent = 0 and re-parent afterward).
    /// @param newOwner   The new owner wallet (must sign; != address(0)).
    /// @param newParent  Live parent under newOwner, or address(0).
    /// @param newOwnerSig EIP-712 signature by newOwner over
    ///                    OwnerBinding{agent: msg.sender, parent: newParent,
    ///                    nonce: bindingNonce[msg.sender]}.
    function rotateOwner(
        address newOwner,
        address newParent,
        bytes calldata newOwnerSig
    ) external {
        if (!isLive(msg.sender)) revert AgentNotActive();
        _bindOwner(newOwner, newParent, newOwnerSig, false);
    }

    /// @notice EIP-712 digest the owner wallet signs to revoke an agent and
    ///         its whole subtree.
    function revokeHash(address agent) public view returns (bytes32) {
        return _hashTypedDataV4(keccak256(abi.encode(REVOKE_TYPEHASH, agent)));
    }

    /// @notice Revoke `agent` and, through the parent links, every
    ///         descendant. Final. Authorized by the agent's owner wallet:
    ///         either it sends the transaction (empty `ownerSig`), or anyone
    ///         relays its RevokeAgent signature (so the wallet needs no gas).
    ///         A leaked owner key can therefore destroy its fleet, but never
    ///         take it: taking needs the agent keys.
    function revokeAgent(address agent, bytes calldata ownerSig) external {
        address owner = agentOwner[agent];
        if (owner == address(0) || !agents[agent].active) revert AgentNotActive();
        if (msg.sender != owner) {
            if (ownerSig.length != 65 || revokeHash(agent).recover(ownerSig) != owner) {
                revert BadRevokeSignature();
            }
        }
        agents[agent].active = false;
        emit AgentRevoked(agent, owner);
    }

    /// @notice True iff `agent` is registered, not revoked, and has no
    ///         revoked ancestor. The gate for every agent action.
    ///         An agent whose chain of parents does not end within
    ///         MAX_FLEET_DEPTH steps is NOT live: a chain that long can only
    ///         come from re-parenting an agent that already had descendants,
    ///         and a revoked ancestor beyond the walk must never be missed.
    ///         Re-parenting that agent higher brings its descendants back.
    function isLive(address agent) public view returns (bool) {
        if (agents[agent].registeredAt == 0) return false;
        address a = agent;
        for (uint256 i = 0; i <= MAX_FLEET_DEPTH; i++) {
            if (!agents[a].active) return false;
            a = agentParent[a];
            if (a == address(0)) return true;
        }
        return false;
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
        if (!isLive(msg.sender)) revert AgentNotActive();
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
    // The value index
    //
    // A token is in the jurisdiction's value index while the DAO Registry
    // holds a non-zero `jurisdiction.parity.<token>` (0x and 40 lowercase hex
    // digits, a decimal string), the same key the economy module reads.
    // Editing it is constitutional (the governor protects the prefix).
    // The value of an amount is amount x parity / 1e30, in 6-decimal value
    // units: 1 USDC (6 decimals) at parity 1e30 is 1e6, and so is 1 DAI
    // (18 decimals) at parity 1e18. REP is that value x 1e12, DAO side.
    // =========================================================================

    /// @notice The DAO Registry (the value index's source). Set once.
    address public registry;

    uint256 private constant PARITY_SCALE = 1e30;
    /// @notice Decimals of every token Substrate has accepted, stored when
    ///         it was first listed and never read again (any number of
    ///         decimals; tokenListed says whether it was listed).
    mapping(address => uint8) public tokenDecimals;

    /// @notice Has `token` been listed (its fees can be collected)?
    mapping(address => bool) public tokenListed;

    /// @dev Every token ever listed, in listing order (the pools' tokens).
    address[] private _poolTokens;

    event TokenListed(address indexed token, uint8 decimals);

    error NotInIndex(address token);
    error BadTokenDecimals(address token);
    error TokenNotListed(address token);

    /// @notice The token's parity in the index, 0 = not in it (or malformed).
    function parityOf(address token) public view returns (uint256) {
        return _registryUint(
            string.concat("jurisdiction.parity.", Strings.toHexString(token)));
    }

    /// @dev A Registry value as a decimal number: 1 to 77 ASCII digits and
    ///      nothing else, any other value (unset, malformed, too long to fit
    ///      for sure in a uint256, whose parse would revert) reading as 0,
    ///      so a bad value never blocks an anchor, a payment or a claim.
    function _registryUint(string memory key) private view returns (uint256) {
        string memory v = IRegistryRead(registry).getRegistryValue(key);
        if (bytes(v).length == 0 || bytes(v).length > 77) return 0;
        (bool ok, uint256 n) = Strings.tryParseUint(v);
        return ok ? n : 0;
    }

    /// @notice True iff a payment can be made in `token` now: it is in the
    ///         index and reports its decimals.
    function acceptsToken(address token) public view returns (bool) {
        if (parityOf(token) == 0) return false;
        if (tokenListed[token]) return true;
        (bool ok, ) = _readDecimals(token);
        return ok;
    }

    /// @notice The value of `amount` of `token` at today's index, in 6-decimal
    ///         value units; 0 when the token is not accepted. The one place
    ///         the parity is turned into value (ToolBond values bonds here).
    function valueOf(address token, uint256 amount) external view returns (uint256) {
        if (!acceptsToken(token)) return 0;
        return (amount * parityOf(token)) / PARITY_SCALE;
    }

    /// @notice List `token` (store its decimals) so its fees can be
    ///         collected even after it leaves the index: a payment channel
    ///         lists its token when it opens. Anyone; only an accepted token.
    function listToken(address token) public {
        if (tokenListed[token]) return;
        if (parityOf(token) == 0) revert NotInIndex(token);
        (bool ok, uint8 dec) = _readDecimals(token);
        if (!ok) revert BadTokenDecimals(token);
        tokenListed[token] = true;
        tokenDecimals[token] = dec;
        _poolTokens.push(token);
        emit TokenListed(token, dec);
    }

    function poolTokenCount() external view returns (uint256) {
        return _poolTokens.length;
    }

    function poolTokens() external view returns (address[] memory) {
        return _poolTokens;
    }

    function _readDecimals(address token) private view returns (bool, uint8) {
        if (token.code.length == 0) return (false, 0);
        (bool ok, bytes memory ret) = token.staticcall(abi.encodeWithSignature("decimals()"));
        if (!ok || ret.length < 32) return (false, 0);
        uint256 d = abi.decode(ret, (uint256));
        if (d > type(uint8).max) return (false, 0);
        return (true, uint8(d));
    }

    // =========================================================================
    // Service payments and the fee
    //
    // The fee is serviceFeeBps of the gross, rounded up, so no split of a
    // payment escapes it. feeTreasuryBps of the fee is owed to the treasury;
    // the rest, the tool share, fills the token's pool, which pays the
    // authors and reviewers of the tools that conclude. Both are set at
    // deploy and retuned by the governor; at least a tenth of every fee is
    // the tool share. Pricing is off chain (the provider's signed list).
    // =========================================================================

    /// @notice Fee on every service payment, in basis points of the gross.
    uint16 public serviceFeeBps;
    /// @notice Share OF THE FEE owed to the treasury; the rest is the tool
    ///         share.
    uint16 public feeTreasuryBps;
    uint16 public constant BPS_DENOM = 10_000;

    /// @notice Bounds on the governed service fee: 0.5% to 10%.
    uint16 public constant MIN_SERVICE_FEE_BPS = 50;
    uint16 public constant MAX_SERVICE_FEE_BPS = 1000;
    /// @notice At least 10% of every fee is the tool share, so no vote can
    ///         switch the tool pool off.
    uint16 public constant MAX_FEE_TREASURY_BPS = 9000;

    /// @notice DAO treasury (set once; no setter).
    address public treasury;

    /// @notice The DAO timelock. Set once; no setter.
    address public governor;

    /// @dev Cumulative tool share collected per token, checkpointed by block
    ///      timestamp: the pools' cap and what the close allocates.
    mapping(address => Checkpoints.Trace208) private _toolShareHistory;
    /// @dev Cumulative fees collected per token (treasury share included),
    ///      checkpointed by block timestamp: the epoch-length measure.
    mapping(address => Checkpoints.Trace208) private _feeHistory;

    /// @notice All tool share collected in `token` so far.
    mapping(address => uint256) public toolShareTotal;
    /// @notice All fees collected in `token` so far.
    mapping(address => uint256) public feeTotal;

    event ServicePayment(
        address indexed payer,
        address indexed recipient,
        address indexed token,
        uint256 amount,
        bytes32 requestId
    );
    /// @notice One per fee collected: `payer` is who paid the fee (the buyer
    ///         on a direct payment, the PaymentChannel on a channel payout).
    event ServiceFee(
        address indexed payer,
        address indexed recipient,
        address indexed token,
        uint256 amount,
        uint256 toolShare,
        uint256 toTreasury
    );
    event ServiceFeeBpsSet(uint16 oldBps, uint16 newBps);
    event FeeTreasuryBpsSet(uint16 oldBps, uint16 newBps);

    error NotGovernor();
    error FeeOutOfBounds(uint16 bps);
    error ZeroAddress();
    error ZeroAmount();
    error ShortReceipt(address token, uint256 expected, uint256 received);

    function setServiceFeeBps(uint16 newBps) external {
        if (msg.sender != governor) revert NotGovernor();
        if (newBps < MIN_SERVICE_FEE_BPS || newBps > MAX_SERVICE_FEE_BPS) {
            revert FeeOutOfBounds(newBps);
        }
        emit ServiceFeeBpsSet(serviceFeeBps, newBps);
        serviceFeeBps = newBps;
    }

    function setFeeTreasuryBps(uint16 newBps) external {
        if (msg.sender != governor) revert NotGovernor();
        if (newBps > MAX_FEE_TREASURY_BPS) revert FeeOutOfBounds(newBps);
        emit FeeTreasuryBpsSet(feeTreasuryBps, newBps);
        feeTreasuryBps = newBps;
    }

    /// @notice The fee on a payment of `gross`, rounded up.
    function feeOf(uint256 gross) public view returns (uint256) {
        return (gross * serviceFeeBps + BPS_DENOM - 1) / BPS_DENOM;
    }

    /// @notice Pay `recipient` for a service, in `token` (in the index):
    ///         the recipient receives gross less the fee, straight from the
    ///         payer; Substrate takes the fee. Approve this contract for
    ///         `gross` first.
    function payForService(
        address token,
        address recipient,
        uint256 gross,
        bytes32 requestId
    ) external nonReentrant {
        if (recipient == address(0)) revert ZeroAddress();
        if (gross == 0) revert ZeroAmount();
        if (parityOf(token) == 0) revert NotInIndex(token);
        listToken(token);
        uint256 fee = feeOf(gross);
        _pullFee(token, msg.sender, recipient, gross, fee);
        if (gross > fee) {
            IERC20(token).safeTransferFrom(msg.sender, recipient, gross - fee);
        }
        emit ServicePayment(msg.sender, recipient, token, gross, requestId);
    }

    /// @notice Pay the fee on a payment of `gross` to `recipient` that the
    ///         caller made itself (a PaymentChannel paying a provider the net
    ///         from its deposit). Pulls exactly feeOf(gross) from the caller
    ///         (approve first). The token must have been listed; it need no
    ///         longer be in the index, so a channel opened in a token the DAO
    ///         later removed still settles.
    function collectFee(
        address token,
        address recipient,
        uint256 gross,
        bytes32 requestId
    ) external nonReentrant {
        if (!tokenListed[token]) revert TokenNotListed(token);
        if (gross == 0) revert ZeroAmount();
        _pullFee(token, msg.sender, recipient, gross, feeOf(gross));
        requestId;
    }

    function _pullFee(
        address token,
        address payer,
        address recipient,
        uint256 gross,
        uint256 fee
    ) private {
        if (fee == 0) return;
        _pullExact(token, payer, fee);
        uint256 toTreasury = treasury != address(0)
            ? (fee * feeTreasuryBps) / BPS_DENOM
            : 0;
        uint256 toolShare = fee - toTreasury;

        _book(token, fee, toolShare, toTreasury);
        emit ServiceFee(payer, recipient, token, gross, toolShare, toTreasury);
    }

    function _book(address token, uint256 fee, uint256 toolShare, uint256 toTreasury) private {
        if (toTreasury > 0) treasuryOwed[token] += toTreasury;
        uint48 now_ = SafeCast.toUint48(block.timestamp);
        uint256 f = feeTotal[token] + fee;
        feeTotal[token] = f;
        _feeHistory[token].push(now_, SafeCast.toUint208(f));
        if (toolShare > 0) {
            uint256 t = toolShareTotal[token] + toolShare;
            toolShareTotal[token] = t;
            _toolShareHistory[token].push(now_, SafeCast.toUint208(t));
        }
    }

    /// @dev Pull exactly `amount` of `token` from `from` into Substrate's
    ///      books. A token that delivers less (a transfer fee) is refused.
    function _pullExact(address token, address from, uint256 amount) private {
        uint256 before = IERC20(token).balanceOf(address(this));
        IERC20(token).safeTransferFrom(from, address(this), amount);
        uint256 got = IERC20(token).balanceOf(address(this)) - before;
        if (got != amount) revert ShortReceipt(token, amount, got);
        held[token] += amount;
    }

    /// @notice Tool share collected in `token` at or before `timestamp`.
    function toolShareUpTo(address token, uint256 timestamp) public view returns (uint256) {
        return _toolShareHistory[token].upperLookup(SafeCast.toUint48(timestamp));
    }

    /// @notice Tool share collected in `token` after `fromTs`, at or before `toTs`.
    function toolShareBetween(address token, uint256 fromTs, uint256 toTs)
        external view returns (uint256)
    {
        if (toTs <= fromTs) return 0;
        return toolShareUpTo(token, toTs) - toolShareUpTo(token, fromTs);
    }

    /// @notice Fees collected in `token` at or before `timestamp`.
    function feesUpTo(address token, uint256 timestamp) public view returns (uint256) {
        return _feeHistory[token].upperLookup(SafeCast.toUint48(timestamp));
    }

    /// @notice Fees collected in `token` after `fromTs`, at or before `toTs`.
    function feesBetween(address token, uint256 fromTs, uint256 toTs)
        external view returns (uint256)
    {
        if (toTs <= fromTs) return 0;
        return feesUpTo(token, toTs) - feesUpTo(token, fromTs);
    }

    // =========================================================================
    // The pools and the claims
    //
    // One pool per token. Each epoch's close commits one leaf per earner (a
    // household's owner wallet): the amounts it takes of each token's pool
    // (the same fraction of every pool) and its seed of the REP floor:
    //
    //   keccak256(bytes.concat(keccak256(abi.encode(
    //       CLAIM_TAG, earner, tokens, amounts, seed))))
    //
    // tokens strictly ascending, one amount each (a leaf may have none and
    // only a seed). A claim credits the
    // amounts as owed to the earner and adds their REP value to
    // earnedValue: each amount at its token's parity stored when the epoch
    // was anchored (anchorParity), so a parity edit after the anchor
    // changes nothing. The value is computed here from amounts the pool cap
    // already bounds, never taken from the leaf, so no anchor can grant
    // more REP than the fees collected are worth. The claim then pays every
    // token of the basket to the earner; a token whose transfer fails stays
    // owed and leaves later by withdraw, one token at a time, so it blocks
    // nothing else. Claims of a token never total more
    // than its tool share collected up to poolCutoff.
    //
    // The REP floor. The DAO Registry's `jurisdiction.autonet.rep_floor`, in
    // REP base units (18 decimals; unset or malformed = 0 = no floor), is the
    // least REP an epoch in which tools conclude mints. Each anchor stores it
    // in value units (divided by 1e12, rounded down: REP is value x 1e12, DAO
    // side) for the epoch after it, whose close reads the floor pinned to
    // that anchor: an epoch's seeds are capped at the floor stored at the
    // anchor before it (seedCap; 0 for the first epoch), so the close and
    // the cap read the same number and an edit after that anchor changes
    // nothing for the epoch. When the epoch's claims are worth less, the
    // close gives the difference to the same earners in the same
    // proportions: each leaf's seed, in value units. A claim adds its seed
    // to earnedValue (the DAO's REP claim mints it with the rest) and to
    // seededValue. Should a leaf's seed ever reach past what is left of the
    // cap (a bad anchor), the seed is cut to it and the claim's tokens are
    // paid all the same.
    // =========================================================================

    bytes32 public constant CLAIM_TAG = keccak256("autonet.claim");

    /// @notice Cumulative REP value an earner has claimed, in 6-decimal value
    ///         units. Monotonic: the DAO's REP claim mints the difference to
    ///         what it already paid (x1e12).
    mapping(address => uint256) public earnedValue;
    /// @notice All REP value claimed so far.
    uint256 public totalEarnedValue;

    /// @dev anchor index => token => its parity when the epoch was anchored.
    mapping(uint256 => mapping(address => uint256)) private _anchorParity;

    /// @notice A pool token's parity stored when anchor `index` landed
    ///         (0 = not in the index then): what its claims are valued at.
    function anchorParity(uint256 index, address token) external view returns (uint256) {
        return _anchorParity[index][token];
    }

    /// @notice Has `earner` claimed its leaf of the epoch?
    mapping(address => mapping(bytes32 => bool)) public claimedEpoch;

    /// @notice All of `token`'s pool claimed so far.
    mapping(address => uint256) public claimedTotal;

    /// @notice Claimed and not yet withdrawn, per earner and token.
    mapping(address => mapping(address => uint256)) public owed;
    /// @notice The treasury's fee share and swept surplus, not yet withdrawn.
    mapping(address => uint256) public treasuryOwed;

    /// @dev What Substrate's books say it holds of each token: every fee
    ///      pulled and every surplus swept, less every withdrawal.
    mapping(address => uint256) public held;

    /// @dev anchor index => the REP floor stored when it landed, in value
    ///      units (0 = no floor): the cap of the NEXT epoch's seeds.
    mapping(uint256 => uint256) private _anchorFloor;

    /// @notice The seed claimed so far of anchor `index`'s epoch.
    mapping(uint256 => uint256) public seedClaimed;

    /// @notice The part of earnedValue that came from the REP floor, per
    ///         earner (value units).
    mapping(address => uint256) public seededValue;
    /// @notice All seed claimed so far.
    uint256 public totalSeededValue;

    /// @notice The Registry key of the REP floor (REP base units).
    string public constant REP_FLOOR_KEY = "jurisdiction.autonet.rep_floor";
    /// @dev REP base units per value unit (REP is value x 1e12, DAO side).
    uint256 private constant REP_PER_VALUE = 1e12;

    /// @notice `value` is everything the claim added to earnedValue: its
    ///         tokens' value and its seed.
    event Claimed(address indexed earner, bytes32 indexed epochIdHash, uint256 value, uint256 earnedValueAfter);
    /// @notice The REP floor's part of a claim (value units), as credited.
    event Seeded(address indexed earner, bytes32 indexed epochIdHash, uint256 seed, uint256 seededValueAfter);
    event ClaimedToken(address indexed earner, bytes32 indexed epochIdHash, address indexed token, uint256 amount);
    event Withdrawn(address indexed earner, address indexed token, address to, uint256 amount);
    /// @notice A claim's transfer of `token` failed; the amount stays owed.
    event PayDeferred(address indexed earner, address indexed token, uint256 amount);
    event TreasuryWithdrawn(address indexed token, uint256 amount);
    event Swept(address indexed token, uint256 amount);

    error EpochNotAnchored(bytes32 epochIdHash);
    error AlreadyClaimed(bytes32 epochIdHash);
    error ClaimRootMissing(bytes32 epochIdHash);
    error ClaimProofInvalid(bytes32 epochIdHash);
    error TokensNotSorted();
    error LengthMismatch();
    error PoolExceeded(address token, uint256 cap, uint256 claimedAfter);
    error NothingOwed();

    /// @notice The most that can ever have been claimed of `token`'s pool by
    ///         now: its tool share collected up to poolCutoff.
    function poolCap(address token) public view returns (uint256) {
        uint256 c = poolCutoff;
        return c == 0 ? 0 : toolShareUpTo(token, c);
    }

    /// @notice The REP floor in the Registry now, in value units (its REP
    ///         base units divided by 1e12, rounded down); 0 when unset or
    ///         malformed (_registryUint), so it can never block an anchor.
    function repFloorValue() public view returns (uint256) {
        return _registryUint(REP_FLOOR_KEY) / REP_PER_VALUE;
    }

    /// @notice The REP floor stored when anchor `index` landed (value
    ///         units): the cap of the next epoch's seeds.
    function anchorFloor(uint256 index) external view returns (uint256) {
        return _anchorFloor[index];
    }

    /// @notice The most anchor `index`'s claims' seeds can total: the floor
    ///         stored at the anchor before it (0 for the first), the floor
    ///         its close read.
    function seedCap(uint256 index) public view returns (uint256) {
        return index == 0 ? 0 : _anchorFloor[index - 1];
    }

    /// @notice Claim an earner's leaf of an anchored epoch, on its behalf.
    ///         Anyone may submit (the household's operator, typically); the
    ///         leaf fixes the earner, so nothing can go elsewhere. Once per
    ///         (earner, epoch). Credits the amounts and pays each token to
    ///         the earner in the same transaction; a token whose transfer
    ///         fails (paused, frozen, the earner blocked) stays owed, for
    ///         withdraw or withdrawTo later. `seed` (value units) is the
    ///         leaf's part of the REP floor, credited up to what is left of
    ///         the epoch's seedCap.
    function claim(
        address earner,
        bytes32 epochIdHash,
        address[] calldata tokens,
        uint256[] calldata amounts,
        uint256 seed,
        bytes32[] calldata proof
    ) external nonReentrant {
        string storage epochId = epochIdByHash[epochIdHash];
        if (bytes(epochId).length == 0) revert EpochNotAnchored(epochIdHash);
        if (claimedEpoch[earner][epochIdHash]) revert AlreadyClaimed(epochIdHash);
        if (tokens.length != amounts.length) revert LengthMismatch();
        for (uint256 i = 1; i < tokens.length; i++) {
            if (tokens[i] <= tokens[i - 1]) revert TokensNotSorted();
        }

        uint256 index = anchorIndexByEpochId[epochId] - 1;
        bytes32 root = anchors[index].claimRoot;
        if (root == bytes32(0)) revert ClaimRootMissing(epochIdHash);
        if (!_verifyProof(proof, root, claimLeaf(earner, tokens, amounts, seed))) {
            revert ClaimProofInvalid(epochIdHash);
        }

        claimedEpoch[earner][epochIdHash] = true;
        uint256 value = _credit(earner, epochIdHash, index, tokens, amounts);
        if (seed != 0) value += _seed(earner, epochIdHash, index, seed);
        if (value > 0) {
            earnedValue[earner] += value;
            totalEarnedValue += value;
        }
        emit Claimed(earner, epochIdHash, value, earnedValue[earner]);
        for (uint256 i = 0; i < tokens.length; i++) {
            if (amounts[i] != 0) _tryPay(earner, tokens[i]);
        }
    }

    /// @dev Pay what `earner` is owed in `token` if the transfer succeeds;
    ///      otherwise leave it owed. Books first, restored on failure.
    function _tryPay(address earner, address token) private {
        uint256 a = owed[earner][token];
        if (a == 0) return;
        owed[earner][token] = 0;
        held[token] -= a;
        if (IERC20(token).trySafeTransfer(earner, a)) {
            emit Withdrawn(earner, token, earner, a);
        } else {
            owed[earner][token] = a;
            held[token] += a;
            emit PayDeferred(earner, token, a);
        }
    }

    /// @notice The leaf a claim proves (exposed for the close and tests).
    function claimLeaf(
        address earner,
        address[] calldata tokens,
        uint256[] calldata amounts,
        uint256 seed
    ) public pure returns (bytes32) {
        return keccak256(bytes.concat(keccak256(
            abi.encode(CLAIM_TAG, earner, tokens, amounts, seed))));
    }

    /// @dev Credit up to `seed` of what is left of anchor `index`'s seedCap
    ///      and return what was credited.
    function _seed(address earner, bytes32 epochIdHash, uint256 index, uint256 seed)
        private returns (uint256 s)
    {
        uint256 claimed = seedClaimed[index];
        uint256 cap = seedCap(index);
        if (claimed >= cap) return 0;
        s = seed < cap - claimed ? seed : cap - claimed;
        seedClaimed[index] = claimed + s;
        uint256 after_ = seededValue[earner] + s;
        seededValue[earner] = after_;
        totalSeededValue += s;
        emit Seeded(earner, epochIdHash, s, after_);
    }

    /// @dev Credit the amounts (each capped by its pool) and return their
    ///      REP value at the anchor's stored parities.
    function _credit(
        address earner,
        bytes32 epochIdHash,
        uint256 index,
        address[] calldata tokens,
        uint256[] calldata amounts
    ) private returns (uint256 value) {
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 a = amounts[i];
            if (a == 0) continue;
            address t = tokens[i];
            uint256 after_ = claimedTotal[t] + a;
            uint256 cap = poolCap(t);
            if (after_ > cap) revert PoolExceeded(t, cap, after_);
            claimedTotal[t] = after_;
            owed[earner][t] += a;
            value += (a * _anchorParity[index][t]) / PARITY_SCALE;
            emit ClaimedToken(earner, epochIdHash, t, a);
        }
    }

    /// @notice Pay `earner` what it is owed in `token`. Anyone may trigger
    ///         it; the money only goes to the earner.
    function withdraw(address earner, address token) external nonReentrant {
        _withdraw(earner, token, earner);
    }

    /// @notice The earner withdraws what it is owed in `token` to another
    ///         address (for an address its token's issuer blocked).
    function withdrawTo(address token, address to) external nonReentrant {
        if (to == address(0)) revert ZeroAddress();
        _withdraw(msg.sender, token, to);
    }

    function _withdraw(address earner, address token, address to) private {
        uint256 a = owed[earner][token];
        if (a == 0) revert NothingOwed();
        owed[earner][token] = 0;
        held[token] -= a;
        IERC20(token).safeTransfer(to, a);
        emit Withdrawn(earner, token, to, a);
    }

    /// @notice Pay the treasury what it is owed in `token`. Anyone.
    function withdrawTreasury(address token) external nonReentrant {
        uint256 a = treasuryOwed[token];
        if (a == 0) revert NothingOwed();
        treasuryOwed[token] = 0;
        held[token] -= a;
        IERC20(token).safeTransfer(treasury, a);
        emit TreasuryWithdrawn(token, a);
    }

    /// @notice Tokens sent here outside a payment belong to the treasury:
    ///         move the surplus over the books into what it is owed. Anyone.
    ///         The pools take only fees.
    function sweep(address token) external nonReentrant {
        if (treasury == address(0)) revert ZeroAddress();
        uint256 bal = IERC20(token).balanceOf(address(this));
        uint256 h = held[token];
        if (bal <= h) revert NothingOwed();
        uint256 surplus = bal - h;
        held[token] = bal;
        treasuryOwed[token] += surplus;
        emit Swept(token, surplus);
    }

    /// @dev Sorted-pair merkle verification (OpenZeppelin-compatible).
    function _verifyProof(
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
}
