// contracts/Registry.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import {IJurisdictionData} from "../interfaces/IJurisdictionData.sol";

contract Registry is IERC721Receiver, ReentrancyGuard {

    mapping (string => string) private reg;
    string[] private keys;
    address public owner;
    address public wrapper;
    address public jurisdictionAddress;
    mapping(bytes32 => uint256) public earmarkedFunds;

    modifier _treasuryOps(){
         require(msg.sender == owner , "Only the DAO can make transfers");
        _;
    }

    modifier _regedit() {
        require(msg.sender == owner || msg.sender==wrapper, "Only the DAO can edit registry");
        _;
    }

    event ReceivedETH(address indexed from, uint256 amount);
    event ReceivedERC721(address indexed from, address indexed token, uint256 tokenId);
    event TransferredETH(address indexed to, uint256 amount);
    event TransferredERC20(address indexed token, address indexed to, uint256 amount);
    event TransferredERC721(address indexed token, address indexed to, uint256 tokenId);
    
    event JurisdictionAddressSet(address indexed jurisdiction);
    event FundsEarmarked(bytes32 indexed purpose, uint256 amount);
    event EarmarkedFundsWithdrawn(bytes32 indexed purpose, uint256 amount);
    event EarmarkedFundsDisbursed(address indexed recipient, bytes32 indexed purpose, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);


     receive() external payable {
        emit ReceivedETH(msg.sender, msg.value);
    }
     function onERC721Received(
        address operator,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external override returns (bytes4) {
        operator; data;
        emit ReceivedERC721(from, msg.sender, tokenId);
        return this.onERC721Received.selector;
    }

    function transferETH(address payable to, uint256 amount) _treasuryOps external nonReentrant {
        require(address(this).balance >= amount, "Insufficient balance");
        to.transfer(amount);
        emit TransferredETH(to, amount);
    }

    function transferERC20(
        address token,
        address to,
        uint256 amount
    ) external _treasuryOps {
        bool success = IERC20(token).transfer(to, amount);
        require(success, "ERC20 transfer failed");
        emit TransferredERC20(token, to, amount);
    }

    function transferERC721(
        address token,
        address to,
        uint256 tokenId
    ) external _treasuryOps {
        IERC721(token).safeTransferFrom(address(this), to, tokenId);
        emit TransferredERC721(token, to, tokenId);
    }
    
    constructor(address _owner, address _wrapper) {
        require(_owner != address(0), "Owner address cannot be zero");
        owner = _owner;
        wrapper=_wrapper;
    }

    event RegistryUpdated(string  key, string  value);

    // NEW FUNCTION: Transfer ownership to the Timelock after setup
    function transferOwnership(address newOwner) external {
        require(msg.sender == owner, "Only the current owner can transfer ownership");
        require(newOwner != address(0), "New owner cannot be the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function editRegistry(string memory key, string memory value) public _regedit {
        if (bytes(reg[key]).length == 0) {
            keys.push(key);
        }
        reg[key] = value;
        emit RegistryUpdated(key, value);
    }
    
    function batchEditRegistry(string[] memory newKeys, string[] memory values) public _regedit {
        for (uint256 i = 0; i < newKeys.length; i++) {
            string memory key = newKeys[i];
            string memory value = values[i];
            if (bytes(reg[key]).length == 0) {
                keys.push(key);
            }
            reg[key] = value;
        }
    }

    function getRegistryValue(string memory key) public view returns (string memory) {
        return reg[key];
    }

    function getAllKeys() public view returns (string[] memory) {
        return keys;
    }

    function getAllValues() public view returns (string[] memory) {
        string[] memory values = new string[](keys.length);
        for (uint i = 0; i < keys.length; i++) {
            values[i] = reg[keys[i]];
        }
        return values;
    }

    function setJurisdictionAddress(address _jurisdictionAddress) external _regedit {
        require(_jurisdictionAddress != address(0), "Jurisdiction address cannot be zero");
        jurisdictionAddress = _jurisdictionAddress;
        emit JurisdictionAddressSet(_jurisdictionAddress);
    }

    function earmarkFunds(bytes32 purpose, uint256 amount, address tokenAddress) external _treasuryOps {
        uint256 currentBalance = IERC20(tokenAddress).balanceOf(address(this));
        require(currentBalance >= earmarkedFunds[purpose] + amount, "Cannot earmark more than available balance");
        earmarkedFunds[purpose] += amount;
        emit FundsEarmarked(purpose, amount);
    }

    function withdrawEarmarkedFunds(bytes32 purpose, uint256 amount) external _treasuryOps {
        require(earmarkedFunds[purpose] >= amount, "Registry: Cannot withdraw more than earmarked");
        earmarkedFunds[purpose] -= amount;
        emit EarmarkedFundsWithdrawn(purpose, amount);
    }

    function disburseEarmarked(address recipient, uint256 amount, bytes32 purpose, address tokenAddress) external nonReentrant {
        require(msg.sender == jurisdictionAddress, "Registry: Caller is not the Jurisdiction");
        require(earmarkedFunds[purpose] >= amount, "Registry: Insufficient earmarked funds");
        
        uint256 currentBalance = IERC20(tokenAddress).balanceOf(address(this));
        require(currentBalance >= amount, "Registry: Insufficient token balance for disbursement");

        earmarkedFunds[purpose] -= amount;
        bool success = IERC20(tokenAddress).transfer(recipient, amount);
        require(success, "ERC20 transfer failed during disbursement");
        emit EarmarkedFundsDisbursed(recipient, purpose, amount);
    }

    /**
     * @notice Allows DAO governance to reclaim funds from a concluded benefits epoch after a grace period.
     * @dev The grace period starts from the beginning of the *next* epoch.
     * @param epochId The ID of the epoch to reclaim from.
     * @param paymentToken The address of the token used in that epoch.
     * @param isDelegateReward A boolean to specify which type of epoch it was.
     */
    function reclaimEarmarkedFunds(uint256 epochId, address paymentToken, bool isDelegateReward) external _treasuryOps {
        // 1. Get the configured grace period from this registry.
        string memory gracePeriodStr = getRegistryValue("benefits.claim.gracePeriod");
        uint256 gracePeriod = Strings.parseUint(gracePeriodStr);
        require(gracePeriod > 0, "Registry: Grace period not set");

        // 2. Determine the timestamp when the grace period started.
        // This is the start time of the *next* epoch (epochId + 1).
        uint48 gracePeriodStartTime;
        if (isDelegateReward) {
            gracePeriodStartTime = IJurisdictionData(jurisdictionAddress).getDelegateRewardEpochStart(epochId + 1);
        } else {
            gracePeriodStartTime = IJurisdictionData(jurisdictionAddress).getPassiveIncomeEpochStart(epochId + 1);
        }

        // 3. Perform the critical time check.
        require(gracePeriodStartTime > 0, "Registry: The subsequent epoch has not started yet");
        require(block.timestamp > gracePeriodStartTime + gracePeriod, "Registry: Claim grace period has not passed");

        // 4. Construct the purpose hash and reclaim the funds.
        bytes32 purpose;
        if (isDelegateReward) {
            purpose = keccak256(abi.encodePacked("DELEGATE_REWARD", epochId, paymentToken));
        } else {
            purpose = keccak256(abi.encodePacked("PASSIVE_INCOME", epochId, paymentToken));
        }
        
        uint256 remainingAmount = earmarkedFunds[purpose];
        require(remainingAmount > 0, "Registry: No funds to reclaim");
        
        earmarkedFunds[purpose] = 0;

        emit EarmarkedFundsWithdrawn(purpose, remainingAmount);
    }
   
}
// Registry.sol