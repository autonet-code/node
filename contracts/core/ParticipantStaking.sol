// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../tokens/ATNToken.sol";
import "../utils/AutonetLib.sol";

/**
 * @title ParticipantStaking
 * @dev Manages staking for all participant roles in the Autonet ecosystem.
 *      Handles deposits, withdrawals, lockup periods, and slashing.
 */
contract ParticipantStaking {
    using AutonetLib for AutonetLib.ParticipantRole;

    ATNToken public immutable atnToken;
    address public governance;

    mapping(address => AutonetLib.StakeInfo) public stakes;
    mapping(AutonetLib.ParticipantRole => uint256) public minStakeAmount;
    mapping(AutonetLib.ParticipantRole => uint256) public stakeLockupDuration;

    // Authorized slashers (ResultsRewards, DisputeManager, etc.)
    mapping(address => bool) public authorizedSlashers;

    event Staked(address indexed participant, AutonetLib.ParticipantRole indexed role, uint256 amount);
    event Unstaked(address indexed participant, AutonetLib.ParticipantRole indexed role, uint256 amount);
    event UnstakeRequested(address indexed participant, uint256 unlockTime);
    event Slashed(address indexed participant, AutonetLib.ParticipantRole indexed role, uint256 amount, string reason);
    event MinStakeUpdated(AutonetLib.ParticipantRole indexed role, uint256 newMinStake);
    event SlasherAuthorized(address indexed slasher, bool authorized);

    modifier onlyGovernance() {
        require(msg.sender == governance, "Not governance");
        _;
    }

    modifier onlyAuthorizedSlasher() {
        require(authorizedSlashers[msg.sender] || msg.sender == governance, "Not authorized to slash");
        _;
    }

    constructor(address _atnToken, address _governance) {
        atnToken = ATNToken(_atnToken);
        governance = _governance;

        // Default minimum stakes (in wei)
        uint8 decimals = atnToken.decimals();
        minStakeAmount[AutonetLib.ParticipantRole.PROPOSER] = 100 * (10**decimals);
        minStakeAmount[AutonetLib.ParticipantRole.SOLVER] = 50 * (10**decimals);
        minStakeAmount[AutonetLib.ParticipantRole.COORDINATOR] = 500 * (10**decimals);
        minStakeAmount[AutonetLib.ParticipantRole.AGGREGATOR] = 1000 * (10**decimals);
        minStakeAmount[AutonetLib.ParticipantRole.VALIDATOR] = 10000 * (10**decimals);
        // Inference providers stake as guarantee for BME payment integrity
        minStakeAmount[AutonetLib.ParticipantRole.INFERENCE_PROVIDER] = 200 * (10**decimals);

        // Default lockup durations
        stakeLockupDuration[AutonetLib.ParticipantRole.PROPOSER] = 7 days;
        stakeLockupDuration[AutonetLib.ParticipantRole.SOLVER] = 3 days;
        stakeLockupDuration[AutonetLib.ParticipantRole.COORDINATOR] = 14 days;
        stakeLockupDuration[AutonetLib.ParticipantRole.AGGREGATOR] = 14 days;
        stakeLockupDuration[AutonetLib.ParticipantRole.VALIDATOR] = 21 days;
        // Inference providers: 7-day lockup (shorter than training roles — faster market)
        stakeLockupDuration[AutonetLib.ParticipantRole.INFERENCE_PROVIDER] = 7 days;
    }

    function setGovernance(address _newGovernance) external onlyGovernance {
        governance = _newGovernance;
    }

    function setMinStake(AutonetLib.ParticipantRole _role, uint256 _amount) external onlyGovernance {
        minStakeAmount[_role] = _amount;
        emit MinStakeUpdated(_role, _amount);
    }

    function setLockupDuration(AutonetLib.ParticipantRole _role, uint256 _duration) external onlyGovernance {
        stakeLockupDuration[_role] = _duration;
    }

    function setAuthorizedSlasher(address _slasher, bool _authorized) external onlyGovernance {
        authorizedSlashers[_slasher] = _authorized;
        emit SlasherAuthorized(_slasher, _authorized);
    }

    function stake(AutonetLib.ParticipantRole _role, uint256 _amount) external {
        require(_role != AutonetLib.ParticipantRole.NONE, "Invalid role");
        require(_amount >= minStakeAmount[_role], "Below minimum stake");
        require(!stakes[msg.sender].active, "Already staked");

        stakes[msg.sender] = AutonetLib.StakeInfo({
            amount: _amount,
            role: _role,
            lockupUntil: 0,
            active: true
        });

        require(atnToken.transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        emit Staked(msg.sender, _role, _amount);
    }

    function addToStake(uint256 _amount) external {
        AutonetLib.StakeInfo storage userStake = stakes[msg.sender];
        require(userStake.active, "Not staked");
        require(userStake.lockupUntil == 0, "Unstake in progress");

        userStake.amount += _amount;
        require(atnToken.transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        emit Staked(msg.sender, userStake.role, _amount);
    }

    function requestUnstake() external {
        AutonetLib.StakeInfo storage userStake = stakes[msg.sender];
        require(userStake.active, "Not staked");
        require(userStake.lockupUntil == 0, "Unstake already requested");

        userStake.lockupUntil = block.timestamp + stakeLockupDuration[userStake.role];
        emit UnstakeRequested(msg.sender, userStake.lockupUntil);
    }

    function unstake() external {
        AutonetLib.StakeInfo storage userStake = stakes[msg.sender];
        require(userStake.active, "Not staked");
        require(userStake.lockupUntil > 0, "Unstake not requested");
        require(block.timestamp >= userStake.lockupUntil, "Still locked");

        uint256 amount = userStake.amount;
        AutonetLib.ParticipantRole role = userStake.role;

        userStake.active = false;
        userStake.amount = 0;
        userStake.role = AutonetLib.ParticipantRole.NONE;
        userStake.lockupUntil = 0;

        require(atnToken.transfer(msg.sender, amount), "Transfer failed");
        emit Unstaked(msg.sender, role, amount);
    }

    function slash(address _participant, uint256 _amount, string calldata _reason) external onlyAuthorizedSlasher {
        AutonetLib.StakeInfo storage userStake = stakes[_participant];
        require(userStake.active, "Not staked");
        require(_amount <= userStake.amount, "Slash exceeds stake");

        userStake.amount -= _amount;
        emit Slashed(_participant, userStake.role, _amount, _reason);

        // Deactivate if below minimum
        if (userStake.amount < minStakeAmount[userStake.role]) {
            userStake.active = false;
        }
    }

    function getStake(address _participant) external view returns (AutonetLib.StakeInfo memory) {
        return stakes[_participant];
    }

    function isActiveParticipant(address _participant, AutonetLib.ParticipantRole _role) external view returns (bool) {
        AutonetLib.StakeInfo memory s = stakes[_participant];
        return s.active && s.role == _role && s.amount >= minStakeAmount[_role];
    }
}
