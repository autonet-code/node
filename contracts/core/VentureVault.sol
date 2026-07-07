// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @title VentureVault — the agent-as-venture funding instrument (third contract
///        of the autonet economy).
///
/// One instance PER VENTURE (deploy-per-venture, like Project clones). ATN — the
/// Substrate.sol token — is the ONLY currency. The AGENT is the investable unit,
/// but the venture is a HUMAN + AGENT operation: the moat verifiers probe is
/// usually the human's data, hardware, credentials, or labor, not the model
/// (every competitor rents the same models). Backers therefore fund the
/// venture's whole operating burn — the tranche drip is unrestricted ATN to the
/// agent address, and the agent's owner is paid from it like any other cost;
/// nothing earmarks it for inference. The venture converts proven utility into
/// service revenue, and backers hold a pull-based claim on the vault's NET
/// revenue (never the artifact — handing over pinned code would leak the moat).
///
/// NO JUDGE ANYWHERE (venture_vault_design.md, ratified 2026-07-05). The
/// pre-substrate Project's arbiter / appeal / overrule machine is DELETED
/// wholesale — an unarbitrated escrow cannot know delivery truth, so we never
/// seat one. Instead:
///
///   * arbitration judged the PAST (needs shared truth) → gone;
///   * tranche voting allocates the FUTURE (each backer's private judgment
///     suffices) → the halt vote.
///
/// This is the third instance of the ratified theft-ceiling pattern
/// (voucher→item in ServiceMarket's channel, channel→client, tranche→investor
/// here): a backer's downside is bounded to the tranches not yet dripped.
///
/// LIFECYCLE (one forward-only path)
/// ---------------------------------
///   Fundraising → (raiseMin met by deadline?)
///     no  → Refunding    (everyone withdraws their full contribution)
///     yes → Trials       (verifiers probe the moat; N distinct-owner passes)
///             greenlight → Active   (tranche drip begins; verifier fee paid)
///             deadline w/o greenlight → Refunding
///   Active → revenue lands here; claimRevenue splits termsBps→backers,
///            remainder→agent. Backers may HALT future tranches (≥1/3 weight);
///            the optional jurisdiction timelock may VETO → Vetoed (permanent
///            halt + refunds of UNSPENT raise; never touches distributed
///            revenue).
///
/// PULL PAYMENTS THROUGHOUT. The vault never pushes ATN to a party that might
/// revert; every payout is a claim the beneficiary pulls (agent tranche/agent
/// revenue share, backer refund, backer revenue share, verifier fee). Reentrancy
/// -guarded; no admin beyond the optional timelock veto.

/// @dev Minimal read of Substrate.sol: agent identity + owner binding. Distinct
///      -owner verifier enforcement leans on getAgentOwner (the cryptographically
///      verified owner wallet), NOT on a self-declared field.
interface ISubstrateVault {
    function isRegistered(address agent) external view returns (bool);
    function getAgentOwner(address agent) external view returns (address);
}

contract VentureVault is ReentrancyGuard {
    using SafeERC20 for IERC20;

    // =========================================================================
    // IMMUTABLE CONFIG (set at construction — the ratified/prospectus terms)
    // =========================================================================

    /// @notice The ATN token (Substrate.sol). The ONLY currency this vault
    ///         touches — raise, tranches, revenue, refunds, verifier fee.
    IERC20 public immutable atn;
    /// @notice Substrate, for agent identity + owner-binding reads.
    ISubstrateVault public immutable substrate;

    /// @notice The venture agent — receives tranche drip and the agent share of
    ///         revenue. Must be a registered agent with a bound owner.
    address public immutable agent;
    /// @notice The venture agent's owner wallet at construction (getAgentOwner).
    ///         Verifiers must be distinct from this owner (no self-verify).
    address public immutable agentOwner;

    /// @notice sha256 of the prospectus manifest blob (mission hash + terms).
    bytes32 public immutable prospectusDigest;

    /// @notice Backer share of NET revenue, in basis points (0..10_000). The
    ///         remainder goes to the agent. Pro-rata WITHIN this share by
    ///         contribution weight.
    uint16 public immutable termsBps;

    /// @notice Below this, the raise fails at the deadline → full refunds.
    uint256 public immutable raiseMin;
    /// @notice Contribution is capped here; contribute reverts past it.
    uint256 public immutable raiseMax;

    /// @notice Number of equal tranches the raised total is dripped over.
    uint256 public immutable plannedTranches;
    /// @notice Seconds per tranche period (default = the epoch duration passed
    ///          at construction). One tranche becomes claimable each period.
    uint256 public immutable tranchePeriod;

    /// @notice Distinct-owner verifier passes required to greenlight (N-of-M).
    uint256 public immutable greenlightQuorum;

    /// @notice Contribution deadline AND the trials deadline. Before greenlight,
    ///         reaching this with raiseMin met but quorum unmet → refunds; not
    ///         reaching raiseMin → refunds.
    uint256 public immutable trialDeadline;

    /// @notice Optional jurisdiction timelock that may veto (permanent halt +
    ///          refunds of unspent raise). address(0) = no veto power exists.
    address public immutable timelock;

    /// @notice Verifier fee: 1% of the raise, split equally among distinct-owner
    ///          verifiers who passed by greenlight time. Basis points.
    uint16 public constant VERIFIER_FEE_BPS = 100; // 1%
    uint16 public constant BPS_DENOM = 10_000;

    /// @notice Backer weight (of the raised total) that can halt / un-halt future
    ///          tranches: strictly greater than 1/3. Encoded as a threshold the
    ///          accumulated vote weight must EXCEED (× raised).
    /// @dev The check is `haltWeight * 3 > raised` — i.e. weight > raised/3.
    ///      "≥ 1/3 of raised weight can halt" with the boundary handled below.

    // =========================================================================
    // LIFECYCLE STATE
    // =========================================================================

    enum Phase { Fundraising, Trials, Active, Refunding, Vetoed }
    Phase public phase;

    /// @notice Total ATN contributed by backers (final = the raise, once trials
    ///          open). During Fundraising it grows as backers contribute.
    uint256 public raised;

    /// @notice Per-backer contribution (pro-rata weight for revenue + refunds).
    mapping(address => uint256) public contribution;

    /// @notice Timestamp trials were entered / drip clock started (greenlight).
    uint256 public greenlitAt;

    /// @notice Tranches already claimed by the agent.
    uint256 public tranchesClaimed;

    /// @notice ATN moved OUT of the raise pool to the agent as tranche drip. This
    ///          is the "spent" raise — refunds (veto path) only touch the unspent
    ///          remainder.
    uint256 public trancheDistributed;

    // ---- halt vote ----------------------------------------------------------

    /// @notice Whether future tranche drip is halted (set by backer vote).
    bool public halted;
    /// @notice Per-backer current halt-vote flag (weight = their contribution).
    mapping(address => bool) public haltVote;
    /// @notice Sum of contribution weight currently voting to halt.
    uint256 public haltWeight;

    // ---- verifiers ----------------------------------------------------------

    struct Trial {
        bool attested;   // this verifier agent has attested
        bool passed;     // its verdict
        bool feeClaimed; // pulled its slice of the verifier fee
    }
    /// @notice verifier agent → its trial record.
    mapping(address => Trial) public trials;
    /// @notice Distinct owner wallets that have PASSED (dedupe by owner).
    mapping(address => bool) private passedOwner;
    /// @notice Distinct-owner passing verifiers, in attest order. Length at
    ///          greenlight time is the fee-split denominator.
    address[] public passers;
    /// @notice Frozen at greenlight: verifier fee total (1% of raise) and the
    ///          per-verifier slice.
    uint256 public verifierFeePool;
    uint256 public verifierFeeSlice;

    // ---- revenue accumulator (pull-based, CoinOffering / RepToken style) -----

    /// @notice Fixed-point scale for the per-weight revenue accumulator.
    uint256 private constant ACC_PRECISION = 1e18;
    /// @notice Cumulative backer-share revenue per unit of contribution weight,
    ///          scaled by ACC_PRECISION. Increases on each claimRevenue.
    uint256 public accRevenuePerWeight;
    /// @notice Per-backer accumulator checkpoint at last claim.
    mapping(address => uint256) public backerRevenueDebt;
    /// @notice Undistributed agent share of revenue (pull by the agent).
    uint256 public agentRevenueOwed;
    /// @notice Total ATN received as revenue and split so far (audit).
    uint256 public totalRevenueSplit;

    // =========================================================================
    // EVENTS (indexer mirrors)
    // =========================================================================

    event Contributed(address indexed backer, uint256 amount, uint256 raisedTotal);
    event RaiseFailed(uint256 raised, uint256 raiseMin);
    event TrialsOpened(uint256 raised, uint256 deadline);
    event TrialAttested(address indexed verifier, address indexed owner, bool passed, bytes32 reportDigest);
    event Greenlit(uint256 raised, uint256 verifierCount, uint256 verifierFeePool, uint256 greenlitAt);
    event TrancheClaimed(address indexed agent, uint256 index, uint256 amount);
    event RevenueReceived(address indexed from, uint256 amount);
    event RevenueSplit(uint256 amount, uint256 toBackers, uint256 toAgent);
    event BackerRevenueClaimed(address indexed backer, uint256 amount);
    event AgentRevenueClaimed(address indexed agent, uint256 amount);
    event VerifierFeeClaimed(address indexed verifier, uint256 amount);
    event HaltVoteCast(address indexed backer, bool halt, uint256 haltWeight);
    event Halted(uint256 haltWeight, uint256 raised);
    event Unhalted(uint256 haltWeight, uint256 raised);
    event Vetoed(uint256 unspentRefundable);
    event Refunded(address indexed backer, uint256 amount);

    // =========================================================================
    // ERRORS
    // =========================================================================

    error ZeroAddress();
    error AgentNotRegistered();
    error AgentOwnerUnset();
    error BadTerms();
    error BadRaiseBounds();
    error BadTranches();
    error BadPeriod();
    error BadQuorum();
    error DeadlineInPast();
    error WrongPhase();
    error DeadlinePassed();
    error DeadlineNotReached();
    error ZeroAmount();
    error RaiseMaxExceeded(uint256 remaining);
    error NotAgent();
    error NotTimelock();
    error NoTimelock();
    error NotABacker();
    error VerifierNotRegistered();
    error VerifierOwnerUnset();
    error VerifierIsVentureOwner();
    error VerifierOwnerNotDistinct();
    error AlreadyAttested();
    error QuorumNotMet(uint256 have, uint256 need);
    error NothingToClaim();
    error NoTrancheDue();
    error DidNotPass();
    error FeeAlreadyClaimed();
    error NotAPasser();

    // =========================================================================
    // CONSTRUCTOR
    // =========================================================================

    constructor(
        address substrateAddr,
        address agentAddr,
        bytes32 prospectusDigest_,
        uint16 termsBps_,
        uint256 raiseMin_,
        uint256 raiseMax_,
        uint256 plannedTranches_,
        uint256 tranchePeriod_,
        uint256 greenlightQuorum_,
        uint256 trialDeadline_,
        address timelock_
    ) {
        if (substrateAddr == address(0) || agentAddr == address(0)) revert ZeroAddress();
        if (prospectusDigest_ == bytes32(0)) revert ZeroAddress();

        ISubstrateVault sub = ISubstrateVault(substrateAddr);
        if (!sub.isRegistered(agentAddr)) revert AgentNotRegistered();
        address o = sub.getAgentOwner(agentAddr);
        if (o == address(0)) revert AgentOwnerUnset();

        if (termsBps_ > BPS_DENOM) revert BadTerms();
        if (raiseMin_ == 0 || raiseMax_ < raiseMin_) revert BadRaiseBounds();
        if (plannedTranches_ == 0) revert BadTranches();
        if (tranchePeriod_ == 0) revert BadPeriod();
        if (greenlightQuorum_ == 0) revert BadQuorum();
        if (trialDeadline_ <= block.timestamp) revert DeadlineInPast();

        substrate = sub;
        atn = IERC20(substrateAddr); // ATN is Substrate.sol's ERC20 surface
        agent = agentAddr;
        agentOwner = o;
        prospectusDigest = prospectusDigest_;
        termsBps = termsBps_;
        raiseMin = raiseMin_;
        raiseMax = raiseMax_;
        plannedTranches = plannedTranches_;
        tranchePeriod = tranchePeriod_;
        greenlightQuorum = greenlightQuorum_;
        trialDeadline = trialDeadline_;
        timelock = timelock_;

        phase = Phase.Fundraising;
    }

    // =========================================================================
    // FUNDRAISING
    // =========================================================================

    /// @notice Contribute ATN to the raise. Caller must approve this vault first.
    ///         Contributions are pooled; weight (for revenue + refunds) is the
    ///         contribution total. Capped at raiseMax; reverts once full or past
    ///         the deadline.
    function contribute(uint256 amount) external nonReentrant {
        if (phase != Phase.Fundraising) revert WrongPhase();
        if (block.timestamp > trialDeadline) revert DeadlinePassed();
        if (amount == 0) revert ZeroAmount();
        uint256 remaining = raiseMax - raised;
        if (amount > remaining) revert RaiseMaxExceeded(remaining);

        contribution[msg.sender] += amount;
        raised += amount;

        atn.safeTransferFrom(msg.sender, address(this), amount);
        emit Contributed(msg.sender, amount, raised);
    }

    /// @notice Open verifier trials. Callable once raiseMin is met (anyone may
    ///         trigger — it's a state advance, not a privileged act). If the
    ///         deadline passes without raiseMin, the raise has failed → use
    ///         failRaise / refund instead.
    function openTrials() external nonReentrant {
        if (phase != Phase.Fundraising) revert WrongPhase();
        if (raised < raiseMin) revert QuorumNotMet(raised, raiseMin);
        phase = Phase.Trials;
        emit TrialsOpened(raised, trialDeadline);
    }

    /// @notice Mark the raise failed (raiseMin not met by the deadline) → opens
    ///         full refunds. Permissionless timer trigger.
    function failRaise() external nonReentrant {
        if (phase != Phase.Fundraising) revert WrongPhase();
        if (block.timestamp <= trialDeadline) revert DeadlineNotReached();
        if (raised >= raiseMin) revert WrongPhase(); // raiseMin met → trials, not failure
        phase = Phase.Refunding;
        emit RaiseFailed(raised, raiseMin);
    }

    // =========================================================================
    // VERIFIER TRIALS
    // =========================================================================

    /// @notice A registered agent attests its black-box trial verdict on the
    ///         venture. The verifier's OWNER must be distinct from every other
    ///         PASSING verifier's owner AND from the venture agent's owner (no
    ///         self-verify, no sock-puppets). N distinct-owner passes greenlight.
    /// @param pass         The verdict.
    /// @param reportDigest sha256 of the off-chain trial report blob.
    function attestTrial(bool pass, bytes32 reportDigest) external nonReentrant {
        if (phase != Phase.Trials) revert WrongPhase();
        if (block.timestamp > trialDeadline) revert DeadlinePassed();
        if (trials[msg.sender].attested) revert AlreadyAttested();

        if (!substrate.isRegistered(msg.sender)) revert VerifierNotRegistered();
        address vowner = substrate.getAgentOwner(msg.sender);
        if (vowner == address(0)) revert VerifierOwnerUnset();
        if (vowner == agentOwner) revert VerifierIsVentureOwner();

        trials[msg.sender] = Trial({attested: true, passed: pass, feeClaimed: false});

        if (pass) {
            // Distinct-owner dedupe: a second agent under an already-counted
            // owner cannot add a second pass toward quorum or the fee split.
            if (passedOwner[vowner]) revert VerifierOwnerNotDistinct();
            passedOwner[vowner] = true;
            passers.push(msg.sender);
        }

        emit TrialAttested(msg.sender, vowner, pass, reportDigest);
    }

    /// @notice Finalize trials: if quorum met, greenlight (start drip, freeze the
    ///          verifier fee). Callable by anyone once the passes exist. If the
    ///          deadline passes without quorum, use failTrials for refunds.
    function greenlight() external nonReentrant {
        if (phase != Phase.Trials) revert WrongPhase();
        uint256 count = passers.length;
        if (count < greenlightQuorum) revert QuorumNotMet(count, greenlightQuorum);

        phase = Phase.Active;
        greenlitAt = block.timestamp;

        // Verifier fee: 1% of the raise, split equally among the distinct-owner
        // passers. Held out of the tranche pool (drip is over raise − fee).
        verifierFeePool = (raised * VERIFIER_FEE_BPS) / BPS_DENOM;
        verifierFeeSlice = verifierFeePool / count;
        // Any rounding dust from the equal split stays in the vault (treated as
        // agent-claimable via the tranche pool remainder), negligible by design.

        emit Greenlit(raised, count, verifierFeePool, greenlitAt);
    }

    /// @notice Trials failed to reach quorum by the deadline → full refunds.
    ///          Permissionless timer trigger.
    function failTrials() external nonReentrant {
        if (phase != Phase.Trials) revert WrongPhase();
        if (block.timestamp <= trialDeadline) revert DeadlineNotReached();
        if (passers.length >= greenlightQuorum) revert WrongPhase(); // could greenlight
        phase = Phase.Refunding;
        emit RaiseFailed(raised, raiseMin);
    }

    /// @notice A distinct-owner passing verifier pulls its equal slice of the
    ///          1% fee. Available once greenlit. Only the recorded passers (the
    ///          distinct-owner agents captured in `passers` at attest time) split
    ///          the fee — a verifier whose owner slot was already taken never
    ///          entered `passers`, so its `passed` flag is set but it is not a
    ///          fee recipient.
    function claimVerifierFee() external nonReentrant {
        if (phase != Phase.Active) revert WrongPhase();
        Trial storage t = trials[msg.sender];
        if (!t.passed) revert DidNotPass();
        if (!_isPasser(msg.sender)) revert NotAPasser();
        if (t.feeClaimed) revert FeeAlreadyClaimed();
        t.feeClaimed = true;
        verifierFeeClaimedCount += 1;
        if (verifierFeeSlice > 0) {
            atn.safeTransfer(msg.sender, verifierFeeSlice);
        }
        emit VerifierFeeClaimed(msg.sender, verifierFeeSlice);
    }

    function _isPasser(address a) private view returns (bool) {
        uint256 n = passers.length;
        for (uint256 i = 0; i < n; i++) {
            if (passers[i] == a) return true;
        }
        return false;
    }

    // =========================================================================
    // TRANCHES (drip to the agent)
    // =========================================================================

    /// @notice Total ATN available to drip over all tranches: the raise minus the
    ///          verifier fee held out at greenlight.
    function tranchePool() public view returns (uint256) {
        return raised - verifierFeePool;
    }

    /// @notice How many tranches have VESTED (become claimable) as of now, given
    ///          the drip clock (greenlitAt) and halt state. Capped at
    ///          plannedTranches. When halted, no NEW tranches vest beyond those
    ///          already vested when the halt took effect — but since halt "takes
    ///          effect next period", we compute vested purely by elapsed periods
    ///          and gate NEW claims on `halted` at claim time.
    function tranchesVested() public view returns (uint256) {
        if (phase != Phase.Active || greenlitAt == 0) return 0;
        uint256 elapsed = block.timestamp - greenlitAt;
        // Tranche 1 is claimable immediately at greenlight (period 0), then one
        // per elapsed tranchePeriod thereafter.
        uint256 vested = 1 + (elapsed / tranchePeriod);
        if (vested > plannedTranches) vested = plannedTranches;
        return vested;
    }

    /// @notice Agent pulls all vested-but-unclaimed tranches. Halt freezes drip
    ///          at the current period boundary: while halted, only tranches that
    ///          vested in a PRIOR period (i.e. up to the period in which the halt
    ///          landed) remain claimable — new periods do not vest. We implement
    ///          "takes effect next period" by allowing the tranche for the CURRENT
    ///          period to be claimed, but no further, while halted.
    function claimTranche() external nonReentrant {
        if (msg.sender != agent) revert NotAgent();
        if (phase != Phase.Active) revert WrongPhase();

        uint256 vested = tranchesVested();
        if (halted) {
            // Halt takes effect next period: the tranche for the period in which
            // the halt was cast is still claimable, nothing beyond. `haltPeriod`
            // records that boundary.
            uint256 allowed = haltPeriodVested;
            if (vested > allowed) vested = allowed;
        }
        if (vested <= tranchesClaimed) revert NoTrancheDue();

        uint256 pool = tranchePool();
        // Equal per-tranche size; the final tranche absorbs integer remainder so
        // the full pool drips over exactly plannedTranches claims.
        uint256 perTranche = pool / plannedTranches;

        uint256 amount;
        for (uint256 i = tranchesClaimed; i < vested; i++) {
            uint256 slice = perTranche;
            if (i == plannedTranches - 1) {
                // Last tranche: remainder so total == pool exactly.
                slice = pool - (perTranche * (plannedTranches - 1));
            }
            amount += slice;
            emit TrancheClaimed(agent, i, slice);
        }
        tranchesClaimed = vested;
        trancheDistributed += amount;

        if (amount > 0) {
            atn.safeTransfer(agent, amount);
        }
    }

    // =========================================================================
    // REVENUE (pull-based split: termsBps → backers pro-rata, remainder → agent)
    // =========================================================================

    /// @notice Snapshot of tranche vesting at the moment the halt took effect,
    ///          so a halt freezes drip at the current period boundary.
    uint256 public haltPeriodVested;

    /// @notice Split ATN revenue sitting in the vault into the backer accumulator
    ///          (termsBps) and the agent's owed balance (remainder). Anyone may
    ///          call; it processes any ATN that arrived since the last split
    ///          (payForInference lands ATN here by transfer, unlabeled to the
    ///          split — so we measure the vault's unallocated balance).
    /// @dev    Unallocated = current ATN balance − (unspent raise + verifier fee
    ///          pool still owed + revenue already owed to backers/agent). We track
    ///          the split-eligible amount as balance minus everything the vault
    ///          still owes on non-revenue accounts.
    function claimRevenue() external nonReentrant {
        if (phase != Phase.Active && phase != Phase.Vetoed) revert WrongPhase();
        uint256 amount = _unsplitRevenue();
        if (amount == 0) revert NothingToClaim();

        uint256 toBackers = (amount * termsBps) / BPS_DENOM;
        uint256 toAgent = amount - toBackers;

        if (toBackers > 0 && raised > 0) {
            accRevenuePerWeight += (toBackers * ACC_PRECISION) / raised;
            // Track the exact aggregate credited to backers so the vault's
            // non-revenue liabilities are computed from exact integers (never
            // reconstructed from the floored accumulator — that drift could let
            // a later split eat into tranche principal).
            backerRevenueOwed += toBackers;
        } else {
            // No backer weight or zero backer share → all to agent.
            toAgent = amount;
            toBackers = 0;
        }
        agentRevenueOwed += toAgent;
        totalRevenueSplit += amount;

        emit RevenueSplit(amount, toBackers, toAgent);
    }

    /// @notice ATN that has arrived as revenue but not yet been split. Equals the
    ///          vault balance minus everything the vault holds that is NOT
    ///          undistributed revenue (all tracked as exact integers).
    function _unsplitRevenue() internal view returns (uint256) {
        uint256 bal = atn.balanceOf(address(this));
        uint256 owed = _nonRevenueHeld();
        if (bal <= owed) return 0;
        return bal - owed;
    }

    /// @notice Everything the vault currently holds that is NOT undistributed
    ///          revenue: undripped tranche principal, unclaimed verifier fee, and
    ///          split-but-unpulled revenue (backer + agent). All exact integers,
    ///          decremented as each obligation is paid out.
    function _nonRevenueHeld() internal view returns (uint256) {
        uint256 undrippedPrincipal = tranchePool() - trancheDistributed;
        uint256 feeOwed = verifierFeePool - (verifierFeeSlice * verifierFeeClaimedCount);
        return undrippedPrincipal + feeOwed + agentRevenueOwed + backerRevenueOwed;
    }

    /// @notice Exact aggregate backer revenue credited but not yet pulled.
    uint256 public backerRevenueOwed;
    /// @notice Count of verifier-fee slices already pulled (for the fee-owed calc).
    uint256 public verifierFeeClaimedCount;

    /// @notice A backer pulls its accrued share of split revenue (pro-rata by
    ///          contribution weight). Pull-based accumulator — safe against any
    ///          number of revenue drops and backers.
    function claimBackerRevenue() external nonReentrant {
        uint256 c = contribution[msg.sender];
        if (c == 0) revert NotABacker();
        uint256 acc = accRevenuePerWeight;
        uint256 owed = (c * (acc - backerRevenueDebt[msg.sender])) / ACC_PRECISION;
        backerRevenueDebt[msg.sender] = acc;
        if (owed == 0) revert NothingToClaim();
        // Reduce the exact aggregate by what this backer pulls. Flooring dust
        // (never assigned to any backer) simply remains in the vault and is
        // swept into a future revenue split — it never blocks a pull.
        backerRevenueOwed -= owed;
        atn.safeTransfer(msg.sender, owed);
        emit BackerRevenueClaimed(msg.sender, owed);
    }

    /// @notice The agent pulls its accumulated revenue remainder.
    function claimAgentRevenue() external nonReentrant {
        if (msg.sender != agent) revert NotAgent();
        uint256 owed = agentRevenueOwed;
        if (owed == 0) revert NothingToClaim();
        agentRevenueOwed = 0;
        atn.safeTransfer(agent, owed);
        emit AgentRevenueClaimed(agent, owed);
    }

    // =========================================================================
    // HALT VOTE (backers, ≥1/3 of raised weight; symmetric un-halt)
    // =========================================================================

    /// @notice Cast / clear this backer's halt vote. Weight = contribution. When
    ///          the accumulated halt weight EXCEEDS 1/3 of the raise it halts
    ///          future tranches (effective next period). Dropping back to ≤1/3
    ///          un-halts symmetrically.
    /// @param wantHalt true to vote halt, false to withdraw the vote.
    function setHaltVote(bool wantHalt) external nonReentrant {
        if (phase != Phase.Active) revert WrongPhase();
        uint256 c = contribution[msg.sender];
        if (c == 0) revert NotABacker();

        bool cur = haltVote[msg.sender];
        if (cur != wantHalt) {
            haltVote[msg.sender] = wantHalt;
            if (wantHalt) {
                haltWeight += c;
            } else {
                haltWeight -= c;
            }
        }
        emit HaltVoteCast(msg.sender, wantHalt, haltWeight);

        // ≥ 1/3 of raised weight can halt. Boundary: exactly 1/3 counts as
        // "representing ≥ 1/3", so the threshold is haltWeight*3 >= raised.
        bool shouldHalt = haltWeight * 3 >= raised;
        if (shouldHalt && !halted) {
            halted = true;
            // Halt takes effect NEXT period: freeze the drip at the tranche
            // vested for the CURRENT period.
            haltPeriodVested = tranchesVested();
            emit Halted(haltWeight, raised);
        } else if (!shouldHalt && halted) {
            halted = false;
            emit Unhalted(haltWeight, raised);
        }
    }

    // =========================================================================
    // DAO VETO (optional timelock — permanent halt + refunds of UNSPENT raise)
    // =========================================================================

    /// @notice The jurisdiction timelock permanently halts the venture and opens
    ///          refunds of the UNSPENT raise (tranche principal not yet dripped +
    ///          the verifier fee pool if never greenlit). NEVER touches revenue
    ///          already distributed or owed. Forward-facing (allowed by the
    ///          no-judge doctrine); not a ruling on the past.
    function daoVeto() external nonReentrant {
        if (timelock == address(0)) revert NoTimelock();
        if (msg.sender != timelock) revert NotTimelock();
        if (phase == Phase.Refunding || phase == Phase.Vetoed) revert WrongPhase();

        phase = Phase.Vetoed;
        halted = true;

        // Refundable = the unspent raise. If greenlit, that's the undripped
        // tranche principal (verifier fee already earmarked/claimable stays with
        // verifiers). If pre-greenlight (veto during Fundraising/Trials), the
        // whole raise is unspent and refundable pro-rata.
        emit Vetoed(_refundableRaise());
    }

    function _refundableRaise() internal view returns (uint256) {
        if (greenlitAt == 0) {
            // Never greenlit: whole raise is unspent (no verifier fee earmarked).
            return raised;
        }
        return tranchePool() - trancheDistributed;
    }

    // =========================================================================
    // REFUNDS (Refunding on raise/trial failure; Vetoed on daoVeto)
    // =========================================================================

    /// @notice Backer withdraws its refund.
    ///          - Raise/trial failure (Refunding): full contribution back.
    ///          - Veto (Vetoed): pro-rata share of the UNSPENT raise only; the
    ///            spent (dripped) portion is gone, distributed revenue untouched.
    function refund() external nonReentrant {
        if (phase != Phase.Refunding && phase != Phase.Vetoed) revert WrongPhase();
        uint256 c = contribution[msg.sender];
        if (c == 0) revert NotABacker();

        uint256 amount;
        if (phase == Phase.Refunding) {
            // Full refund: raise never left the vault as principal.
            amount = c;
        } else {
            // Vetoed: pro-rata of the unspent raise.
            uint256 refundable = _refundableRaise();
            amount = (c * refundable) / raised;
        }

        contribution[msg.sender] = 0;
        // Clear any halt vote weight this backer held so state stays consistent.
        if (haltVote[msg.sender]) {
            haltVote[msg.sender] = false;
            haltWeight -= c;
        }

        if (amount > 0) {
            atn.safeTransfer(msg.sender, amount);
        }
        emit Refunded(msg.sender, amount);
    }

    // =========================================================================
    // VIEWS
    // =========================================================================

    function passerCount() external view returns (uint256) {
        return passers.length;
    }

    /// @notice Pending backer revenue for `backer` (not yet pulled).
    function pendingBackerRevenue(address backer) external view returns (uint256) {
        uint256 c = contribution[backer];
        if (c == 0) return 0;
        return (c * (accRevenuePerWeight - backerRevenueDebt[backer])) / ACC_PRECISION;
    }

    /// @notice Tranche principal not yet dripped to the agent.
    function undrippedTranchePrincipal() external view returns (uint256) {
        return tranchePool() - trancheDistributed;
    }
}

// =============================================================================
// VentureVaultFactory — deploy-per-venture, emits VentureCreated for the indexer
// =============================================================================

/// @dev Minimal factory: one plain deployment per venture (NOT a clone proxy —
///      the vault has immutable per-venture config, and deploy-per-venture keeps
///      each venture's storage fully isolated). Emits VentureCreated so the
///      indexer mirrors new vaults into the off-chain directory, exactly as
///      ServiceRegistered / AgentRegistered are mirrored.
contract VentureVaultFactory {
    address public immutable substrate;

    event VentureCreated(
        address indexed vault,
        address indexed agent,
        address indexed creator,
        bytes32 prospectusDigest,
        uint256 raiseMin,
        uint256 raiseMax
    );

    error ZeroAddress();

    constructor(address substrateAddr) {
        if (substrateAddr == address(0)) revert ZeroAddress();
        substrate = substrateAddr;
    }

    /// @notice Deploy a new VentureVault. All venture terms are passed through to
    ///         the vault constructor (which validates them + the agent binding).
    function createVenture(
        address agentAddr,
        bytes32 prospectusDigest,
        uint16 termsBps,
        uint256 raiseMin,
        uint256 raiseMax,
        uint256 plannedTranches,
        uint256 tranchePeriod,
        uint256 greenlightQuorum,
        uint256 trialDeadline,
        address timelock
    ) external returns (address vault) {
        VentureVault v = new VentureVault(
            substrate,
            agentAddr,
            prospectusDigest,
            termsBps,
            raiseMin,
            raiseMax,
            plannedTranches,
            tranchePeriod,
            greenlightQuorum,
            trialDeadline,
            timelock
        );
        vault = address(v);
        emit VentureCreated(
            vault,
            agentAddr,
            msg.sender,
            prospectusDigest,
            raiseMin,
            raiseMax
        );
    }
}
