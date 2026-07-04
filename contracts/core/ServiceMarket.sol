// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {EIP712} from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/// @title ServiceMarket — decentralized monetizable APIs (services_market.md)
///
/// Three contracts, one file, one deployment surface for the indexer:
///
///   1. ServiceRegistry — provider publishes/updates/retires a Service.
///      Provider MUST be a registered agent on Substrate.sol (identity is
///      chain-verified via the ISubstrate interface). Events → indexer →
///      Firestore `services`. Chain = truth, blob = spec storage.
///
///   2. ServiceEscrow (v1, escrow-per-request) — fully-atomic settlement:
///      client deposits an ERC20 amount for a (serviceId, requestId),
///      provider marks delivered, client releases. Timeout paths on BOTH
///      sides so neither can grief indefinitely (see griefing analysis on
///      the state machine below).
///
///   3. PaymentChannel (v1.5, prepaid credits) — unidirectional channel:
///      client opens with a deposit, hands the provider EIP-712 vouchers
///      over (channelId, cumulativeAmount) off-chain per work item, provider
///      closes claiming the latest cumulative, remainder refunds to client
///      after a challenge window.
///
/// Service commerce is deliberately independent of ATN's constitutional
/// roles: ANY ERC20 is payable, no substrate standing, no mint. The trust
/// basis is behavioral (identity + atomic payment + receipts), per the spec.

/// @dev Minimal read of Substrate.sol's agent registry. We only need to
///      know "is this address an active registered agent?" — the same gate
///      registerTool / recordTrainingForEpoch use.
interface ISubstrate {
    function isRegistered(address agent) external view returns (bool);
}

// =============================================================================
// ServiceRegistry
// =============================================================================

contract ServiceRegistry {
    ISubstrate public immutable substrate;

    struct Service {
        address provider;      // the registered agent offering the service
        bytes32 specDigest;    // sha256 of the service_spec blob (NOT IPFS)
        address token;         // ERC20 the ask is denominated in
        uint256 askAmount;     // per-work-item price
        bool active;           // false once retired
        uint256 registeredAt;
        uint256 updatedAt;
    }

    /// @dev serviceId is assigned sequentially; index 0 is unused so that a
    ///      zero serviceId reliably means "no such service".
    uint256 public serviceCount;
    mapping(uint256 => Service) public services;

    event ServiceRegistered(
        uint256 indexed serviceId,
        address indexed provider,
        bytes32 indexed specDigest,
        address token,
        uint256 askAmount
    );
    event ServiceAskUpdated(
        uint256 indexed serviceId,
        address indexed provider,
        address token,
        uint256 askAmount
    );
    event ServiceRetired(uint256 indexed serviceId, address indexed provider);

    error NotRegisteredAgent();
    error SpecDigestRequired();
    error TokenRequired();
    error NoSuchService();
    error NotServiceProvider();
    error ServiceNotActive();

    constructor(address substrateAddr) {
        require(substrateAddr != address(0), "substrate=0");
        substrate = ISubstrate(substrateAddr);
    }

    modifier onlyAgent() {
        if (!substrate.isRegistered(msg.sender)) revert NotRegisteredAgent();
        _;
    }

    /// @notice Publish a new service. Provider = msg.sender, must be a
    ///         registered agent on Substrate.
    function registerService(
        bytes32 specDigest,
        address token,
        uint256 askAmount
    ) external onlyAgent returns (uint256 serviceId) {
        if (specDigest == bytes32(0)) revert SpecDigestRequired();
        if (token == address(0)) revert TokenRequired();
        serviceId = ++serviceCount;
        services[serviceId] = Service({
            provider: msg.sender,
            specDigest: specDigest,
            token: token,
            askAmount: askAmount,
            active: true,
            registeredAt: block.timestamp,
            updatedAt: block.timestamp
        });
        emit ServiceRegistered(serviceId, msg.sender, specDigest, token, askAmount);
    }

    /// @notice Change the ask (token and/or amount) of a live service.
    function updateServiceAsk(
        uint256 serviceId,
        address token,
        uint256 askAmount
    ) external {
        Service storage s = services[serviceId];
        if (s.provider == address(0)) revert NoSuchService();
        if (s.provider != msg.sender) revert NotServiceProvider();
        if (!s.active) revert ServiceNotActive();
        if (token == address(0)) revert TokenRequired();
        s.token = token;
        s.askAmount = askAmount;
        s.updatedAt = block.timestamp;
        emit ServiceAskUpdated(serviceId, msg.sender, token, askAmount);
    }

    /// @notice Retire a service (stops appearing in the storefront). Escrow
    ///         and channels already open are unaffected — settlement lives
    ///         in the other contracts, not here.
    function retireService(uint256 serviceId) external {
        Service storage s = services[serviceId];
        if (s.provider == address(0)) revert NoSuchService();
        if (s.provider != msg.sender) revert NotServiceProvider();
        if (!s.active) revert ServiceNotActive();
        s.active = false;
        s.updatedAt = block.timestamp;
        emit ServiceRetired(serviceId, msg.sender);
    }

    function getService(uint256 serviceId) external view returns (Service memory) {
        return services[serviceId];
    }
}

// =============================================================================
// ServiceEscrow (v1 — escrow-per-request)
// =============================================================================
//
// State machine per (client, provider, serviceId, requestId):
//
//   None --openRequest--> Open --markDelivered--> Delivered --release--> Released
//                          |                          |
//                          | (after releaseTimeout)   | (after claimTimeout, if
//                          |  client can reclaim)     |  client silent, provider
//                          v                          |  can claim)
//                       Reclaimed                     v
//                                                   Claimed
//
// GRIEFING ANALYSIS
// -----------------
//   * Provider never delivers (takes deposit hostage): funds are held in
//     escrow, not with the provider. Client reclaims after `releaseTimeout`
//     from openRequest. Provider gains nothing by stalling; worst case the
//     client's capital is locked for `releaseTimeout` — bounded, refundable.
//
//   * Client never releases after honest delivery (services rendered, payment
//     withheld): provider marked Delivered, so after `claimTimeout` from the
//     markDelivered time the provider can `claimDelivered` and take the
//     escrowed amount unilaterally. Client's only lever is to release EARLY
//     (which they'd do to reward good service / build the provider's reviews)
//     or to dispute off-chain within the window. No on-chain dispute court in
//     v1 — the spec says trust is behavioral (receipts/reviews), so the money
//     path just has to be non-grief; adjudication is social.
//
//   * The two timeouts are ordered releaseTimeout (client's reclaim clock,
//     runs from open) vs claimTimeout (provider's claim clock, runs from
//     delivery). Because delivery must happen before releaseTimeout expires
//     (otherwise the client reclaims), and claimTimeout only starts at
//     delivery, the windows don't overlap into a race: once Delivered, the
//     client can still release any time, but reclaim is no longer available
//     (state left Open). First finalizing action wins; all paths are terminal.
//
//   * Front-running / griefing the finalizer: release/reclaim/claim each
//     require the correct caller (client for release/reclaim, provider for
//     claim) so no third party can force a settlement.
//
// Timeouts are constructor args (network can tune per deployment). Amounts
// are pulled via transferFrom at openRequest so the escrow custodies real
// tokens, not a promise. SafeERC20 + ReentrancyGuard + CEI throughout.

contract ServiceEscrow is ReentrancyGuard {
    using SafeERC20 for IERC20;

    ServiceRegistry public immutable registry;

    /// @notice Seconds after openRequest before an undelivered request can be
    ///         reclaimed by the client.
    uint256 public immutable releaseTimeout;
    /// @notice Seconds after markDelivered before a delivered-but-unreleased
    ///         request can be claimed by the provider.
    uint256 public immutable claimTimeout;

    enum Status { None, Open, Delivered, Finalized }

    struct Request {
        address client;
        address provider;
        address token;
        uint256 amount;
        uint256 serviceId;
        uint256 openedAt;
        uint256 deliveredAt;
        Status status;
    }

    /// @dev keyed by keccak256(client, serviceId, requestId) so the same
    ///      requestId can be reused across services/clients without clashing.
    mapping(bytes32 => Request) public requests;

    event RequestOpened(
        bytes32 indexed key,
        address indexed client,
        address indexed provider,
        uint256 serviceId,
        bytes32 requestId,
        address token,
        uint256 amount
    );
    event RequestDelivered(bytes32 indexed key);
    event RequestReleased(bytes32 indexed key, address indexed provider, uint256 amount);
    event RequestReclaimed(bytes32 indexed key, address indexed client, uint256 amount);
    event RequestClaimed(bytes32 indexed key, address indexed provider, uint256 amount);

    error RequestExists();
    error RequestNotOpen();
    error RequestNotDelivered();
    error NotClient();
    error NotProvider();
    error ZeroAmount();
    error TooEarly();
    error ServiceInactive();
    error NoSuchService();

    constructor(
        address registryAddr,
        uint256 releaseTimeout_,
        uint256 claimTimeout_
    ) {
        require(registryAddr != address(0), "registry=0");
        require(releaseTimeout_ > 0 && claimTimeout_ > 0, "timeout=0");
        registry = ServiceRegistry(registryAddr);
        releaseTimeout = releaseTimeout_;
        claimTimeout = claimTimeout_;
    }

    function requestKey(
        address client,
        uint256 serviceId,
        bytes32 requestId
    ) public pure returns (bytes32) {
        return keccak256(abi.encode(client, serviceId, requestId));
    }

    /// @notice Client opens an escrowed request against a service, funding it
    ///         from their ERC20 balance (must approve this contract first).
    ///         The token/provider are read from the registry so the client
    ///         can't misroute funds; `amount` is client-chosen (may match or
    ///         exceed the ask — extra is a tip, released with the rest).
    function openRequest(
        uint256 serviceId,
        bytes32 requestId,
        uint256 amount
    ) external nonReentrant {
        if (amount == 0) revert ZeroAmount();
        ServiceRegistry.Service memory svc = registry.getService(serviceId);
        if (svc.provider == address(0)) revert NoSuchService();
        if (!svc.active) revert ServiceInactive();

        bytes32 key = requestKey(msg.sender, serviceId, requestId);
        if (requests[key].status != Status.None) revert RequestExists();

        requests[key] = Request({
            client: msg.sender,
            provider: svc.provider,
            token: svc.token,
            amount: amount,
            serviceId: serviceId,
            openedAt: block.timestamp,
            deliveredAt: 0,
            status: Status.Open
        });

        // Effects done above; interaction last (CEI). transferFrom pulls the
        // real tokens into escrow custody.
        IERC20(svc.token).safeTransferFrom(msg.sender, address(this), amount);

        emit RequestOpened(
            key, msg.sender, svc.provider, serviceId, requestId, svc.token, amount
        );
    }

    /// @notice Provider signals the work item is delivered. Starts the
    ///         claimTimeout clock and blocks the client's reclaim path.
    function markDelivered(
        address client,
        uint256 serviceId,
        bytes32 requestId
    ) external {
        bytes32 key = requestKey(client, serviceId, requestId);
        Request storage r = requests[key];
        if (r.status != Status.Open) revert RequestNotOpen();
        if (msg.sender != r.provider) revert NotProvider();
        r.status = Status.Delivered;
        r.deliveredAt = block.timestamp;
        emit RequestDelivered(key);
    }

    /// @notice Client releases escrow to the provider (accepts delivery, or
    ///         pays out early to reward good service). Works whether the
    ///         provider marked delivery or not — the client can always pay.
    function release(uint256 serviceId, bytes32 requestId) external nonReentrant {
        bytes32 key = requestKey(msg.sender, serviceId, requestId);
        Request storage r = requests[key];
        if (r.status != Status.Open && r.status != Status.Delivered) {
            revert RequestNotOpen();
        }
        if (msg.sender != r.client) revert NotClient();
        uint256 amount = r.amount;
        address provider = r.provider;
        address token = r.token;
        r.status = Status.Finalized;
        r.amount = 0;
        IERC20(token).safeTransfer(provider, amount);
        emit RequestReleased(key, provider, amount);
    }

    /// @notice Client reclaims escrow for an undelivered request after
    ///         releaseTimeout. Blocked once the provider marked delivery.
    function reclaim(uint256 serviceId, bytes32 requestId) external nonReentrant {
        bytes32 key = requestKey(msg.sender, serviceId, requestId);
        Request storage r = requests[key];
        if (r.status != Status.Open) revert RequestNotOpen();
        if (msg.sender != r.client) revert NotClient();
        if (block.timestamp < r.openedAt + releaseTimeout) revert TooEarly();
        uint256 amount = r.amount;
        address client = r.client;
        address token = r.token;
        r.status = Status.Finalized;
        r.amount = 0;
        IERC20(token).safeTransfer(client, amount);
        emit RequestReclaimed(key, client, amount);
    }

    /// @notice Provider claims escrow for a DELIVERED request the client
    ///         never released, after claimTimeout from delivery. This is the
    ///         provider's anti-grief lever against a silent client.
    function claimDelivered(
        address client,
        uint256 serviceId,
        bytes32 requestId
    ) external nonReentrant {
        bytes32 key = requestKey(client, serviceId, requestId);
        Request storage r = requests[key];
        if (r.status != Status.Delivered) revert RequestNotDelivered();
        if (msg.sender != r.provider) revert NotProvider();
        if (block.timestamp < r.deliveredAt + claimTimeout) revert TooEarly();
        uint256 amount = r.amount;
        address provider = r.provider;
        address token = r.token;
        r.status = Status.Finalized;
        r.amount = 0;
        IERC20(token).safeTransfer(provider, amount);
        emit RequestClaimed(key, provider, amount);
    }

    function getRequest(
        address client,
        uint256 serviceId,
        bytes32 requestId
    ) external view returns (Request memory) {
        return requests[requestKey(client, serviceId, requestId)];
    }
}

// =============================================================================
// PaymentChannel (v1.5 — prepaid credits, unidirectional)
// =============================================================================
//
// "Prepaid credits done trustlessly." Two on-chain txs for N off-chain work
// items:
//
//   1. openChannel(provider, token, deposit) — client escrows a deposit.
//   2. per work item, OFF-CHAIN: client signs an EIP-712 voucher over
//      (channelId, cumulativeAmount). cumulativeAmount is MONOTONE — each
//      voucher supersedes the last. "Not served = no voucher" preserves
//      per-item granularity without a tx per item.
//   3. closeChannel(channelId, cumulativeAmount, sig) — provider submits the
//      HIGHEST voucher it holds; contract pays min(cumulative, deposit) to
//      the provider and opens a challenge window.
//   4. after the challenge window, withdrawRemainder refunds the unclaimed
//      remainder to the client.
//
// Unidirectional only (client→provider). No bidirectional / mutual-close
// complexity — the spec explicitly says keep it minimal and standard.
//
// GRIEFING ANALYSIS
// -----------------
//   * Voucher REPLAY: closeChannel can only be called once per channel
//     (status flips to Closing). A resubmitted voucher hits ChannelNotOpen.
//   * STALE voucher (provider submits an older, lower cumulative than the
//     latest it holds): only hurts the provider — it claims less. The client
//     benefits (larger remainder). We still store the claimed cumulative so a
//     provider can't later "top up" with a higher voucher after close (single
//     close). Providers are expected to submit their highest voucher.
//   * Over-claim (cumulative > deposit): capped at deposit via min(). A
//     client cannot be drained past what it escrowed; a provider cannot forge
//     a voucher (EIP-712 sig recovers to the client, checked on close).
//   * Client vanishing after close: the challenge window is a timer, not a
//     client action. Once it elapses ANYONE can trigger withdrawRemainder
//     (refunds to the client). Provider is never blocked from its claimed
//     funds — those transferred at close, before the window.
//   * Client double-spend / not-served: since a voucher is only produced by
//     the CLIENT and only AFTER the client accepts a work item, a provider
//     that under-delivers simply gets no higher voucher — item-level
//     granularity, same as escrow, at 1/N the gas.
//
// The challenge window in v1.5 is a settle-delay for the remainder refund
// (giving an honest party time to act) rather than a fraud-proof court —
// unidirectional channels have no "higher voucher exists" fraud for the
// PAYER to prove (a higher voucher only helps the provider, who is the one
// closing). We keep the window as the standard refund-delay primitive and as
// the hook a future bidirectional upgrade would use.

contract PaymentChannel is ReentrancyGuard, EIP712 {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    /// @notice Seconds after close before the client's remainder can be
    ///         withdrawn.
    uint256 public immutable challengeWindow;

    enum Status { None, Open, Closing, Settled }

    struct Channel {
        address client;
        address provider;
        address token;
        uint256 deposit;
        uint256 claimed;       // cumulative paid to provider at close
        uint256 closesAt;      // timestamp remainder becomes withdrawable
        Status status;
    }

    uint256 public channelCount;
    mapping(uint256 => Channel) public channels;

    /// @dev EIP-712 typed voucher: the client signs this off-chain per item.
    bytes32 private constant VOUCHER_TYPEHASH =
        keccak256("Voucher(uint256 channelId,uint256 cumulativeAmount)");

    event ChannelOpened(
        uint256 indexed channelId,
        address indexed client,
        address indexed provider,
        address token,
        uint256 deposit
    );
    event ChannelClosed(
        uint256 indexed channelId,
        uint256 paidToProvider,
        uint256 remainderToClient,
        uint256 closesAt
    );
    event RemainderWithdrawn(uint256 indexed channelId, address indexed client, uint256 amount);

    error ProviderRequired();
    error TokenRequired();
    error ZeroDeposit();
    error ChannelNotOpen();
    error ChannelNotClosing();
    error NotProvider();
    error BadVoucherSignature();
    error WindowNotElapsed();

    constructor(uint256 challengeWindow_)
        EIP712("AutonetPaymentChannel", "1")
    {
        require(challengeWindow_ > 0, "window=0");
        challengeWindow = challengeWindow_;
    }

    /// @notice Client opens a unidirectional channel to a provider, escrowing
    ///         `deposit` of `token`. Must approve this contract first.
    function openChannel(
        address provider,
        address token,
        uint256 deposit
    ) external nonReentrant returns (uint256 channelId) {
        if (provider == address(0)) revert ProviderRequired();
        if (token == address(0)) revert TokenRequired();
        if (deposit == 0) revert ZeroDeposit();
        channelId = ++channelCount;
        channels[channelId] = Channel({
            client: msg.sender,
            provider: provider,
            token: token,
            deposit: deposit,
            claimed: 0,
            closesAt: 0,
            status: Status.Open
        });
        IERC20(token).safeTransferFrom(msg.sender, address(this), deposit);
        emit ChannelOpened(channelId, msg.sender, provider, token, deposit);
    }

    /// @notice EIP-712 digest a client signs for a voucher. Exposed for
    ///         off-chain signers / tests.
    function voucherHash(
        uint256 channelId,
        uint256 cumulativeAmount
    ) public view returns (bytes32) {
        return _hashTypedDataV4(
            keccak256(abi.encode(VOUCHER_TYPEHASH, channelId, cumulativeAmount))
        );
    }

    /// @notice Provider closes the channel with its highest voucher. Pays the
    ///         provider min(cumulativeAmount, deposit); the rest becomes the
    ///         client's withdrawable remainder after the challenge window.
    /// @param cumulativeAmount The voucher's cumulative total.
    /// @param signature Client's EIP-712 signature over (channelId, cumulative).
    function closeChannel(
        uint256 channelId,
        uint256 cumulativeAmount,
        bytes calldata signature
    ) external nonReentrant {
        Channel storage c = channels[channelId];
        if (c.status != Status.Open) revert ChannelNotOpen();
        if (msg.sender != c.provider) revert NotProvider();

        // Recover the client's signature over the typed voucher.
        bytes32 digest = voucherHash(channelId, cumulativeAmount);
        address signer = digest.recover(signature);
        if (signer != c.client) revert BadVoucherSignature();

        uint256 pay = cumulativeAmount > c.deposit ? c.deposit : cumulativeAmount;
        c.claimed = pay;
        c.status = Status.Closing;
        c.closesAt = block.timestamp + challengeWindow;

        uint256 remainder = c.deposit - pay;
        // Provider is paid immediately at close (it produced a valid voucher).
        if (pay > 0) {
            IERC20(c.token).safeTransfer(c.provider, pay);
        }
        emit ChannelClosed(channelId, pay, remainder, c.closesAt);
    }

    /// @notice After the challenge window, refund the client's remainder.
    ///         Permissionless trigger (it's a timer) but funds only ever go
    ///         to the client.
    function withdrawRemainder(uint256 channelId) external nonReentrant {
        Channel storage c = channels[channelId];
        if (c.status != Status.Closing) revert ChannelNotClosing();
        if (block.timestamp < c.closesAt) revert WindowNotElapsed();
        uint256 remainder = c.deposit - c.claimed;
        address client = c.client;
        address token = c.token;
        c.status = Status.Settled;
        if (remainder > 0) {
            IERC20(token).safeTransfer(client, remainder);
        }
        emit RemainderWithdrawn(channelId, client, remainder);
    }

    function getChannel(uint256 channelId) external view returns (Channel memory) {
        return channels[channelId];
    }
}
