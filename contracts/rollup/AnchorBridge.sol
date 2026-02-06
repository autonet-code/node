// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title AnchorBridge
 * @dev Stores sidechain checkpoint roots and manages L1/L2 token locks.
 *      Part of the rollup infrastructure for Autonet.
 */
contract AnchorBridge is Ownable {
    IERC20 public immutable token;

    bytes32 public latestRoot;
    uint256 public latestEpoch;

    // Validator set for checkpoint submissions
    mapping(address => bool) public validators;
    uint256 public validatorCount;
    uint256 public requiredSignatures;

    // Token bridge state
    mapping(bytes32 => bool) public processedDeposits;
    mapping(bytes32 => bool) public processedWithdrawals;

    // Checkpoint approval tracking
    mapping(uint256 => mapping(address => bool)) public checkpointApprovals;
    mapping(uint256 => uint256) public checkpointApprovalCount;
    mapping(uint256 => bytes32) public pendingCheckpointRoots;

    event CheckpointSubmitted(uint256 indexed epoch, bytes32 root, address indexed submitter);
    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);
    event Deposited(address indexed user, uint256 amount, bytes32 depositId);
    event Withdrawn(address indexed user, uint256 amount, bytes32 withdrawalId);

    constructor(address _token) Ownable(msg.sender) {
        token = IERC20(_token);
        requiredSignatures = 1; // Start with 1 for PoC
    }

    function addValidator(address _validator) external onlyOwner {
        require(!validators[_validator], "Already validator");
        validators[_validator] = true;
        validatorCount++;
        emit ValidatorAdded(_validator);
    }

    function removeValidator(address _validator) external onlyOwner {
        require(validators[_validator], "Not validator");
        validators[_validator] = false;
        validatorCount--;
        emit ValidatorRemoved(_validator);
    }

    function setRequiredSignatures(uint256 _required) external onlyOwner {
        require(_required <= validatorCount, "Too many required");
        requiredSignatures = _required;
    }

    /**
     * @dev Submit a checkpoint from the L2 chain.
     *      For PoC, simplified to require caller to be a validator.
     */
    function submitCheckpoint(uint256 epoch, bytes32 root) external {
        require(validators[msg.sender], "Not validator");
        require(epoch >= latestEpoch, "Epoch too old");
        require(!checkpointApprovals[epoch][msg.sender], "Already approved");

        // If first approval for this epoch, store the root
        if (checkpointApprovalCount[epoch] == 0) {
            pendingCheckpointRoots[epoch] = root;
        } else {
            require(pendingCheckpointRoots[epoch] == root, "Root mismatch");
        }

        checkpointApprovals[epoch][msg.sender] = true;
        checkpointApprovalCount[epoch]++;

        if (checkpointApprovalCount[epoch] >= requiredSignatures) {
            latestRoot = root;
            latestEpoch = epoch;
            emit CheckpointSubmitted(epoch, root, msg.sender);
        }
    }

    /**
     * @dev Deposit tokens to L2.
     */
    function deposit(uint256 amount) external {
        require(amount > 0, "Zero amount");
        require(token.transferFrom(msg.sender, address(this), amount), "Transfer failed");

        bytes32 depositId = keccak256(abi.encodePacked(msg.sender, amount, block.number, block.timestamp));
        processedDeposits[depositId] = true;

        emit Deposited(msg.sender, amount, depositId);
    }

    /**
     * @dev Withdraw tokens from L2 (with proof verification).
     *      For PoC, simplified proof check.
     */
    function withdraw(uint256 amount, bytes32 withdrawalId, bytes32[] calldata proof) external {
        require(!processedWithdrawals[withdrawalId], "Already processed");

        // Simplified proof verification for PoC
        // In production, verify Merkle proof against latestRoot
        bytes32 leaf = keccak256(abi.encodePacked(msg.sender, amount, withdrawalId));
        require(_verifyProof(proof, latestRoot, leaf), "Invalid proof");

        processedWithdrawals[withdrawalId] = true;
        require(token.transfer(msg.sender, amount), "Transfer failed");

        emit Withdrawn(msg.sender, amount, withdrawalId);
    }

    function _verifyProof(bytes32[] calldata proof, bytes32 root, bytes32 leaf) internal pure returns (bool) {
        bytes32 computedHash = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];
            if (computedHash <= proofElement) {
                computedHash = keccak256(abi.encodePacked(computedHash, proofElement));
            } else {
                computedHash = keccak256(abi.encodePacked(proofElement, computedHash));
            }
        }
        return computedHash == root;
    }
}
