// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {EIP712} from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/// @title ServiceMarket — decentralized monetizable APIs (services_market.md)
///
/// Two contracts, one file, one deployment surface for the indexer:
///
///   1. ServiceRegistry — provider publishes/updates/retires a Service.
///      Provider MUST be a registered agent on Substrate.sol (identity is
///      chain-verified via the ISubstrate interface). Events → indexer →
///      Firestore `services`. Chain = truth, blob = spec storage.
///
///   2. PaymentChannel — the ONLY settlement rail (prepaid, unidirectional):
///      client opens with a deposit (its credit with that provider, across
///      all of the provider's services) and may top it up; it hands the
///      provider EIP-712 vouchers over (channelId, cumulativeAmount)
///      off-chain per work item; the provider cashes in the latest
///      cumulative whenever it likes (settle) without ending the channel,
///      and either side ends it (closeChannel / startClose), the remainder
///      refunding to the client after a challenge window.
///
/// Channel-only settlement is a ratified decision (services_market.md,
/// 2026-07-04), not an interim state: the postpaid escrow was DELETED. An
/// unarbitrated escrow cannot know delivery truth, so some party must bear
/// the lie — a false "delivered" claim steals the deposit, or a silent
/// client steals the work. The channel dissolves the dilemma by making
/// exposure per-item and PREPAID: each request carries a client-signed
/// voucher covering exactly that item's ask, so the theft ceiling is ONE
/// voucher, sized by the client — worth less than the review history a
/// cheating provider burns. No arbitration, no fulfillment oracle.
///
/// Service commerce is ATN-only (ratified 2026-07-10): the earlier "any
/// ERC20 is payable" doctrine is retired. Channel settlement routes the
/// provider payout through Substrate.payForService, so the fee-recycled
/// emission doctrine holds on the canonical rail (closes audit gap G1: the
/// 2.5% service fee that finances the commons was bypassed by the raw
/// safeTransfer payout). Services still get NO substrate standing, mint, or
/// verdict-layer claims — the trust basis is behavioral (identity + atomic
/// payment + receipts), per the spec.

/// @dev Read of Substrate.sol's agent registry AND its ATN ERC20 surface.
///      Registry needs only the isRegistered gate; the channel additionally
///      pulls/refunds deposits and settles through payForService. Substrate's
///      ERC20 functions revert on failure (they never return false), so the
///      bool returns are checked only defensively.
interface ISubstrate {
    function isRegistered(address agent) external view returns (bool);
    function isLive(address agent) external view returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function payForService(address recipient, uint256 amount, bytes32 requestId)
        external
        returns (bool);
    function governor() external view returns (address);
}

// =============================================================================
// ServiceRegistry
// =============================================================================

contract ServiceRegistry {
    ISubstrate public immutable substrate;

    struct Service {
        address provider;      // the registered agent offering the service
        bytes32 specDigest;    // sha256 of the service_spec blob (NOT IPFS)
        uint256 askAmount;     // per-work-item price, ATN-denominated
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
        uint256 askAmount
    );
    event ServiceAskUpdated(
        uint256 indexed serviceId,
        address indexed provider,
        uint256 askAmount
    );
    event ServiceRetired(uint256 indexed serviceId, address indexed provider);

    error NotRegisteredAgent();
    error SpecDigestRequired();
    error NoSuchService();
    error NotServiceProvider();
    error ServiceNotActive();

    constructor(address substrateAddr) {
        require(substrateAddr != address(0), "substrate=0");
        substrate = ISubstrate(substrateAddr);
    }

    modifier onlyAgent() {
        // Live, not just registered: a revoked agent (or one under a revoked
        // ancestor) may not list or change services.
        if (!substrate.isLive(msg.sender)) revert NotRegisteredAgent();
        _;
    }

    /// @notice Publish a new service. Provider = msg.sender, must be a
    ///         registered agent on Substrate. Ask is ATN-denominated.
    function registerService(
        bytes32 specDigest,
        uint256 askAmount
    ) external onlyAgent returns (uint256 serviceId) {
        if (specDigest == bytes32(0)) revert SpecDigestRequired();
        serviceId = ++serviceCount;
        services[serviceId] = Service({
            provider: msg.sender,
            specDigest: specDigest,
            askAmount: askAmount,
            active: true,
            registeredAt: block.timestamp,
            updatedAt: block.timestamp
        });
        emit ServiceRegistered(serviceId, msg.sender, specDigest, askAmount);
    }

    /// @notice Change the ask amount of a live service.
    function updateServiceAsk(
        uint256 serviceId,
        uint256 askAmount
    ) external {
        Service storage s = services[serviceId];
        if (s.provider == address(0)) revert NoSuchService();
        if (s.provider != msg.sender) revert NotServiceProvider();
        if (!s.active) revert ServiceNotActive();
        s.askAmount = askAmount;
        s.updatedAt = block.timestamp;
        emit ServiceAskUpdated(serviceId, msg.sender, askAmount);
    }

    /// @notice Retire a service (stops appearing in the storefront). Channels
    ///         already open are unaffected — settlement lives in the
    ///         PaymentChannel contract, not here.
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
// PaymentChannel — prepaid credits, unidirectional (the ONLY settlement rail)
// =============================================================================
//
// "Prepaid credits done trustlessly." A buyer's channel with a provider is
// its credit balance with that provider, used across all of the provider's
// services, like API credits. On-chain txs are per deposit and per cash-in,
// never per work item:
//
//   1. openChannel(provider, deposit) — client escrows a deposit of ATN.
//      topUp(channelId, amount) adds to it while the channel is Open.
//   2. per work item, OFF-CHAIN: client signs an EIP-712 voucher over
//      (channelId, cumulativeAmount). cumulativeAmount is MONOTONE — each
//      voucher supersedes the last. "Not served = no voucher" preserves
//      per-item granularity without a tx per item.
//   3. settle(channelId, cumulativeAmount, sig) — provider cashes in the
//      HIGHEST voucher it holds, as often as it likes: the contract pays
//      min(cumulative, deposit) minus what it already paid, and the channel
//      stays as it was (Open stays Open), so the buyer's credit carries on.
//      `claimed` is the cumulative paid to the provider so far (gross).
//   4. closeChannel(channelId, cumulativeAmount, sig) — provider ends the
//      channel with its best voucher: pays whatever settle has not paid yet
//      and opens a challenge window. Or the client ends it (startClose):
//      the provider may still settle or close inside that window.
//   5. after the challenge window, withdrawRemainder refunds deposit -
//      claimed to the client.
//
// ATN-only (ratified 2026-07-10). The deposit and every top-up are pulled
// with substrate.transferFrom and every provider payout (settle, and close
// when it pays something) routes through
// substrate.payForService(provider, pay, channelId) — so the provider
// receives NET of the service fee, exactly like the direct payForService
// rail (closes audit gap G1), and the fee burns as soon as the provider
// cashes in rather than when the channel ends. Every payout emits
// ChannelPaid. Vouchers remain GROSS-denominated (the cumulative the client
// signs is the pre-fee ask); the fee is taken at each payout. The theft
// ceiling below is UNCHANGED by settling in steps and by top-ups — the fee
// is on the same side of every voucher, and each payout only moves claimed
// up to a cumulative the client signed, so the bound is still one
// client-sized voucher's increment (net of fee, strictly smaller). The
// remainder refund is NOT a service payment and pays no fee.
//
// Unidirectional only (client→provider). No bidirectional / mutual-close
// complexity — the spec explicitly says keep it minimal and standard.
//
// GRIEFING ANALYSIS
// -----------------
//   * Voucher REPLAY: settle pays only the part of a voucher above claimed,
//     so a resubmitted voucher pays nothing (NothingToSettle). closeChannel
//     can only be called once per channel (status flips to Closing); after
//     it, settle and closeChannel hit ChannelNotOpen.
//   * STALE voucher (provider submits an older, lower cumulative than the
//     latest it holds): only hurts the provider — it claims less now, and a
//     later settle with the higher voucher pays the difference. A close with
//     a voucher at or below claimed pays nothing more. The client benefits
//     (larger remainder) only if the provider closes on the stale one.
//   * Over-claim (cumulative > deposit): capped at deposit via min(). A
//     client cannot be drained past what it escrowed; a provider cannot forge
//     a voucher (EIP-712 sig recovers to the client, checked on settle and
//     close). A top-up raises the cap: a voucher the client signed above the
//     old deposit becomes collectable up to the new one, which is what the
//     client signed for (a voucher is the client's running total of work it
//     accepted, not a payment capped by the deposit of the day).
//   * Client vanishing after close: the challenge window is a timer, not a
//     client action. Once it elapses ANYONE can trigger withdrawRemainder
//     (refunds to the client). Provider is never blocked from its claimed
//     funds — those transferred at each settle and at close, before the
//     window.
//   * Client double-spend / not-served: since a voucher is only produced by
//     the CLIENT and only AFTER the client accepts a work item, a provider
//     that under-delivers simply gets no higher voucher — per-item
//     granularity at 1/N the gas. The theft ceiling is exactly one voucher:
//     a provider that pockets a voucher and never serves steals at most that
//     item's increment, so the client's loss is bounded by a single item,
//     however many times the provider has settled before.
//   * Top-ups only while Open: a client that started a close cannot add to
//     the deposit (it opens a new channel instead), so the remainder it
//     waits for never moves during the window.
//
// The challenge window is a settle-delay for the remainder refund
// (giving an honest party time to act) rather than a fraud-proof court —
// unidirectional channels have no "higher voucher exists" fraud for the
// PAYER to prove (a higher voucher only helps the provider, who is the one
// closing). We keep the window as the standard refund-delay primitive and as
// the hook a future bidirectional upgrade would use.

contract PaymentChannel is ReentrancyGuard, EIP712 {
    using ECDSA for bytes32;

    /// @notice The ATN token / settlement contract (Substrate.sol). Deposits
    ///         are pulled here and payouts route through its payForService.
    ISubstrate public immutable substrate;

    /// @notice Seconds after close before the client's remainder can be
    ///         withdrawn, for channels opened from now on. Governed: the
    ///         Substrate's governor (the DAO timelock) sets it. Each channel
    ///         keeps the window it opened with, so a change never moves an
    ///         open channel's deadline.
    uint256 public challengeWindow;

    /// ClientClosing (appended, so earlier values keep their numbers): the
    /// client started a close. The provider may still settle with its best
    /// voucher until closesAt; after that the client takes the whole deposit
    /// back, so a vanished provider never locks it.
    enum Status { None, Open, Closing, Settled, ClientClosing }

    struct Channel {
        address client;
        address provider;
        uint256 deposit;
        uint256 claimed;       // cumulative paid to the provider so far (gross)
        uint256 closesAt;      // timestamp remainder becomes withdrawable
        Status status;
        uint256 window;        // challengeWindow when the channel opened
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
        uint256 deposit
    );
    /// @notice A payout to the provider (settle, or a close that pays
    ///         something): `amount` gross paid now, `paidTotal` the
    ///         channel's cumulative paid so far (its `claimed`).
    event ChannelPaid(
        uint256 indexed channelId,
        address indexed client,
        address indexed provider,
        uint256 amount,
        uint256 paidTotal
    );
    event ChannelToppedUp(
        uint256 indexed channelId,
        address indexed client,
        uint256 amount,
        uint256 deposit
    );
    /// @dev paidToProvider is the cumulative paid over the channel's life
    ///      (its `claimed` at close), not only what the close itself paid.
    event ChannelClosed(
        uint256 indexed channelId,
        uint256 paidToProvider,
        uint256 remainderToClient,
        uint256 closesAt
    );
    event RemainderWithdrawn(uint256 indexed channelId, address indexed client, uint256 amount);
    event ChannelCloseStarted(uint256 indexed channelId, address indexed client, uint256 closesAt);
    event ChallengeWindowSet(uint256 previous, uint256 current);

    error ProviderRequired();
    error ZeroDeposit();
    error ChannelNotOpen();
    error ChannelNotClosing();
    error NotProvider();
    error NotClient();
    error BadVoucherSignature();
    error WindowNotElapsed();
    error DepositTransferFailed();
    error NotGovernor();
    error ZeroWindow();
    error NothingToSettle();

    constructor(address substrateAddr, uint256 challengeWindow_)
        EIP712("AutonetPaymentChannel", "1")
    {
        require(substrateAddr != address(0), "substrate=0");
        require(challengeWindow_ > 0, "window=0");
        substrate = ISubstrate(substrateAddr);
        challengeWindow = challengeWindow_;
    }

    /// @notice Set the challenge window for channels opened from now on.
    ///         Only the Substrate's governor (the DAO timelock).
    function setChallengeWindow(uint256 newWindow) external {
        if (msg.sender != substrate.governor()) revert NotGovernor();
        if (newWindow == 0) revert ZeroWindow();
        emit ChallengeWindowSet(challengeWindow, newWindow);
        challengeWindow = newWindow;
    }

    /// @notice Client opens a unidirectional channel to a provider, escrowing
    ///         `deposit` of ATN. Must approve this contract on Substrate first.
    function openChannel(
        address provider,
        uint256 deposit
    ) external nonReentrant returns (uint256 channelId) {
        if (provider == address(0)) revert ProviderRequired();
        if (deposit == 0) revert ZeroDeposit();
        channelId = ++channelCount;
        channels[channelId] = Channel({
            client: msg.sender,
            provider: provider,
            deposit: deposit,
            claimed: 0,
            closesAt: 0,
            status: Status.Open,
            window: challengeWindow
        });
        // Substrate reverts on insufficient allowance/balance; the bool is
        // defensive (it never returns false).
        if (!substrate.transferFrom(msg.sender, address(this), deposit)) {
            revert DepositTransferFailed();
        }
        emit ChannelOpened(channelId, msg.sender, provider, deposit);
    }

    /// @notice Client adds `amount` of ATN to an Open channel's deposit (its
    ///         credit with the provider). Must approve this contract on
    ///         Substrate first, as for openChannel.
    function topUp(uint256 channelId, uint256 amount) external nonReentrant {
        Channel storage c = channels[channelId];
        if (c.status != Status.Open) revert ChannelNotOpen();
        if (msg.sender != c.client) revert NotClient();
        if (amount == 0) revert ZeroDeposit();
        c.deposit += amount;
        if (!substrate.transferFrom(msg.sender, address(this), amount)) {
            revert DepositTransferFailed();
        }
        emit ChannelToppedUp(channelId, msg.sender, amount, c.deposit);
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

    /// @notice Client starts closing an open channel. The provider has one
    ///         challenge window to collect its best voucher (settle or
    ///         closeChannel); if it does not, the client withdraws everything
    ///         not yet paid after the window (withdrawRemainder).
    function startClose(uint256 channelId) external nonReentrant {
        Channel storage c = channels[channelId];
        if (c.status != Status.Open) revert ChannelNotOpen();
        if (msg.sender != c.client) revert NotClient();
        c.status = Status.ClientClosing;
        c.closesAt = block.timestamp + c.window;
        emit ChannelCloseStarted(channelId, msg.sender, c.closesAt);
    }

    /// @notice Provider cashes in without ending the channel, up to
    ///         `upTo` (gross, cumulative): what it has served. Vouchers pay
    ///         ahead, so the best voucher may cover more than was served; an
    ///         honest provider claims only what it served, and the rest stays
    ///         the client's (refunded at the close, or served later under the
    ///         same voucher). Pays min(cumulativeAmount, deposit, upTo) minus
    ///         what was already paid, NET of the service fee (routed through
    ///         payForService); the channel keeps its status (Open stays Open,
    ///         a client's close keeps its deadline).
    /// @param cumulativeAmount The voucher's cumulative total (gross).
    /// @param signature Client's EIP-712 signature over (channelId, cumulative).
    /// @param upTo The cumulative total to be paid after this call, at most.
    function settle(
        uint256 channelId,
        uint256 cumulativeAmount,
        bytes calldata signature,
        uint256 upTo
    ) external nonReentrant {
        Channel storage c = channels[channelId];
        uint256 capped = _checkProviderVoucher(c, channelId, cumulativeAmount, signature);
        if (upTo < capped) capped = upTo;
        if (capped <= c.claimed) revert NothingToSettle();
        _payProvider(c, channelId, capped);
    }

    /// @notice Provider closes the channel with its highest voucher. Pays the
    ///         provider whatever of min(cumulativeAmount, deposit) settle has
    ///         not paid yet, NET of the service fee (routed through
    ///         payForService); deposit - claimed becomes the client's
    ///         withdrawable remainder after the challenge window.
    /// @param cumulativeAmount The voucher's cumulative total (gross).
    /// @param signature Client's EIP-712 signature over (channelId, cumulative).
    function closeChannel(
        uint256 channelId,
        uint256 cumulativeAmount,
        bytes calldata signature
    ) external nonReentrant {
        Channel storage c = channels[channelId];
        uint256 capped = _checkProviderVoucher(c, channelId, cumulativeAmount, signature);
        c.status = Status.Closing;
        c.closesAt = block.timestamp + c.window;
        // A voucher at or below what settle already paid pays nothing more.
        if (capped > c.claimed) {
            _payProvider(c, channelId, capped);
        }
        emit ChannelClosed(channelId, c.claimed, c.deposit - c.claimed, c.closesAt);
    }

    /// @dev The provider may collect while the channel is Open, or inside the
    ///      window of a close the client started. Checks the caller and the
    ///      client's voucher signature; returns min(cumulative, deposit).
    function _checkProviderVoucher(
        Channel storage c,
        uint256 channelId,
        uint256 cumulativeAmount,
        bytes calldata signature
    ) private view returns (uint256 capped) {
        if (c.status != Status.Open && !(c.status == Status.ClientClosing
                && block.timestamp < c.closesAt)) revert ChannelNotOpen();
        if (msg.sender != c.provider) revert NotProvider();

        // Recover the client's signature over the typed voucher.
        bytes32 digest = voucherHash(channelId, cumulativeAmount);
        address signer = digest.recover(signature);
        if (signer != c.client) revert BadVoucherSignature();

        capped = cumulativeAmount > c.deposit ? c.deposit : cumulativeAmount;
    }

    /// @dev Pay the provider up to `capped` (> claimed): claimed moves first,
    ///      then the payout routes through payForService so the fee-recycled
    ///      emission takes its cut on the canonical rail and the provider
    ///      receives the net. The channelId is the requestId so the payout is
    ///      matchable off-chain.
    function _payProvider(Channel storage c, uint256 channelId, uint256 capped) private {
        uint256 pay = capped - c.claimed;
        c.claimed = capped;
        substrate.payForService(c.provider, pay, bytes32(channelId));
        emit ChannelPaid(channelId, c.client, c.provider, pay, capped);
    }

    /// @notice After the challenge window, refund the client's remainder.
    ///         Permissionless trigger (it's a timer) but funds only ever go
    ///         to the client. A refund is not a service payment — no fee.
    function withdrawRemainder(uint256 channelId) external nonReentrant {
        Channel storage c = channels[channelId];
        // A provider close (remainder) or a client close the provider never
        // closed (the deposit less whatever settle paid).
        if (c.status != Status.Closing && c.status != Status.ClientClosing) {
            revert ChannelNotClosing();
        }
        if (block.timestamp < c.closesAt) revert WindowNotElapsed();
        uint256 remainder = c.deposit - c.claimed;
        address client = c.client;
        c.status = Status.Settled;
        if (remainder > 0) {
            substrate.transfer(client, remainder);
        }
        emit RemainderWithdrawn(channelId, client, remainder);
    }

    function getChannel(uint256 channelId) external view returns (Channel memory) {
        return channels[channelId];
    }
}
