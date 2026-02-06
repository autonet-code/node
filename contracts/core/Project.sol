// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../tokens/ATNToken.sol";
import "../tokens/ProjectToken.sol";
import "../utils/AutonetLib.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title Project
 * @dev Manages AI development projects: funding, PT issuance, training budgets,
 *      and inference services with PT-based discounts.
 */
contract Project is Ownable {
    ATNToken public immutable atnToken;
    address public governance;
    mapping(address => bool) public authorizedDisbursers;

    uint256 public nextProjectId = 1;

    struct ProjectData {
        uint256 id;
        string name;
        string descriptionCid;
        address founder;
        AutonetLib.ProjectStatus status;
        uint256 creationTime;

        // Tokens
        ProjectToken projectToken;
        uint256 totalPTsIssued;

        // Funding
        uint256 fundingGoalATN;
        uint256 currentFundingATN;
        uint256 taskRewardBudgetATN;

        // AI Service
        string matureModelWeightsCid;
        uint256 atnPricePerInference;
        address inferenceFeeRecipient;
    }

    mapping(uint256 => ProjectData) public projects;
    mapping(uint256 => AutonetLib.DiscountTier[]) public discountTiers;
    mapping(uint256 => mapping(address => uint256)) public withdrawnRevenue;
    mapping(address => uint256[]) public founderProjects;

    uint256 public nextInferenceRequestId;

    event ProjectCreated(uint256 indexed projectId, address indexed founder, string name, address projectToken);
    event ProjectFunded(uint256 indexed projectId, address indexed funder, uint256 atnAmount, uint256 ptsIssued);
    event ProjectStatusChanged(uint256 indexed projectId, AutonetLib.ProjectStatus newStatus);
    event TaskBudgetAllocated(uint256 indexed projectId, uint256 amount);
    event MatureModelUpdated(uint256 indexed projectId, string weightsCid);
    event InferencePriceSet(uint256 indexed projectId, uint256 price);
    event InferenceRequested(uint256 indexed projectId, address indexed user, uint256 requestId, string inputCid, uint256 fee);
    event RevenueWithdrawn(uint256 indexed projectId, address indexed shareholder, uint256 amount);
    event DisburserAuthorized(address indexed disburser, bool authorized);

    modifier projectExists(uint256 _projectId) {
        require(projects[_projectId].founder != address(0), "Project does not exist");
        _;
    }

    modifier onlyFounderOrGov(uint256 _projectId) {
        require(msg.sender == projects[_projectId].founder || msg.sender == governance, "Not founder or governance");
        _;
    }

    modifier onlyGov() {
        require(msg.sender == governance, "Not governance");
        _;
    }

    constructor(address _atnToken, address _governance) Ownable(msg.sender) {
        atnToken = ATNToken(_atnToken);
        governance = _governance;
    }

    function setGovernance(address _newGovernance) external onlyOwner {
        governance = _newGovernance;
    }

    function setAuthorizedDisburser(address _disburser, bool _authorized) external onlyGov {
        authorizedDisbursers[_disburser] = _authorized;
        emit DisburserAuthorized(_disburser, _authorized);
    }

    function createProject(
        string memory _name,
        string memory _descriptionCid,
        uint256 _fundingGoalATN,
        uint256 _initialBudgetATN,
        uint256 _founderPTAmount,
        string memory _ptName,
        string memory _ptSymbol
    ) external returns (uint256 projectId) {
        projectId = nextProjectId++;
        require(_initialBudgetATN <= _fundingGoalATN, "Budget exceeds goal");

        ProjectToken newPT = new ProjectToken(_ptName, _ptSymbol, address(this));

        ProjectData storage p = projects[projectId];
        p.id = projectId;
        p.name = _name;
        p.descriptionCid = _descriptionCid;
        p.founder = msg.sender;
        p.status = AutonetLib.ProjectStatus.FUNDING;
        p.creationTime = block.timestamp;
        p.projectToken = newPT;
        p.fundingGoalATN = _fundingGoalATN;
        p.taskRewardBudgetATN = _initialBudgetATN;
        p.inferenceFeeRecipient = address(this);

        if (_founderPTAmount > 0) {
            newPT.mint(msg.sender, _founderPTAmount);
            p.totalPTsIssued = _founderPTAmount;
        }

        founderProjects[msg.sender].push(projectId);
        emit ProjectCreated(projectId, msg.sender, _name, address(newPT));
    }

    function fundProject(uint256 _projectId, uint256 _atnAmount, uint256 _expectedPTs)
        external projectExists(_projectId)
    {
        ProjectData storage p = projects[_projectId];
        require(p.status == AutonetLib.ProjectStatus.FUNDING, "Not in funding phase");

        require(atnToken.transferFrom(msg.sender, address(this), _atnAmount), "Transfer failed");
        p.currentFundingATN += _atnAmount;

        p.projectToken.mint(msg.sender, _expectedPTs);
        p.totalPTsIssued += _expectedPTs;

        emit ProjectFunded(_projectId, msg.sender, _atnAmount, _expectedPTs);

        if (p.currentFundingATN >= p.fundingGoalATN) {
            p.status = AutonetLib.ProjectStatus.ACTIVE_TRAINING;
            emit ProjectStatusChanged(_projectId, AutonetLib.ProjectStatus.ACTIVE_TRAINING);
        }
    }

    function allocateTaskBudget(uint256 _projectId, uint256 _amount)
        external onlyFounderOrGov(_projectId) projectExists(_projectId)
    {
        ProjectData storage p = projects[_projectId];
        require(p.currentFundingATN >= p.taskRewardBudgetATN + _amount, "Insufficient funds");
        p.taskRewardBudgetATN += _amount;
        emit TaskBudgetAllocated(_projectId, _amount);
    }

    function updateStatus(uint256 _projectId, AutonetLib.ProjectStatus _newStatus)
        external onlyGov projectExists(_projectId)
    {
        projects[_projectId].status = _newStatus;
        emit ProjectStatusChanged(_projectId, _newStatus);
    }

    function setMatureModel(uint256 _projectId, string memory _weightsCid, uint256 _price)
        external onlyFounderOrGov(_projectId) projectExists(_projectId)
    {
        ProjectData storage p = projects[_projectId];
        p.matureModelWeightsCid = _weightsCid;
        p.atnPricePerInference = _price;
        p.status = AutonetLib.ProjectStatus.DEPLOYED;

        emit MatureModelUpdated(_projectId, _weightsCid);
        emit InferencePriceSet(_projectId, _price);
        emit ProjectStatusChanged(_projectId, AutonetLib.ProjectStatus.DEPLOYED);
    }

    function setDiscountTiers(uint256 _projectId, AutonetLib.DiscountTier[] memory _tiers)
        external onlyFounderOrGov(_projectId) projectExists(_projectId)
    {
        delete discountTiers[_projectId];
        for (uint i = 0; i < _tiers.length; i++) {
            discountTiers[_projectId].push(_tiers[i]);
        }
    }

    function getEffectivePrice(uint256 _projectId, address _user) public view returns (uint256) {
        ProjectData storage p = projects[_projectId];
        uint256 userPT = p.projectToken.balanceOf(_user);
        uint256 discount = 0;

        AutonetLib.DiscountTier[] storage tiers = discountTiers[_projectId];
        for (uint i = 0; i < tiers.length; i++) {
            if (userPT >= tiers[i].ptThreshold && tiers[i].discountPermille > discount) {
                discount = tiers[i].discountPermille;
            }
        }

        require(discount <= 1000, "Invalid discount");
        return (p.atnPricePerInference * (1000 - discount)) / 1000;
    }

    function requestInference(uint256 _projectId, string memory _inputCid)
        external projectExists(_projectId) returns (uint256 requestId)
    {
        ProjectData storage p = projects[_projectId];
        require(p.status == AutonetLib.ProjectStatus.DEPLOYED, "Not deployed");

        uint256 fee = getEffectivePrice(_projectId, msg.sender);
        require(atnToken.transferFrom(msg.sender, p.inferenceFeeRecipient, fee), "Fee transfer failed");

        requestId = nextInferenceRequestId++;
        emit InferenceRequested(_projectId, msg.sender, requestId, _inputCid, fee);
    }

    function disburseFromBudget(uint256 _projectId, address _recipient, uint256 _amount)
        external projectExists(_projectId) returns (bool)
    {
        require(authorizedDisbursers[msg.sender] || msg.sender == governance, "Not authorized to disburse");
        ProjectData storage p = projects[_projectId];
        require(p.taskRewardBudgetATN >= _amount, "Insufficient budget");

        p.taskRewardBudgetATN -= _amount;
        require(atnToken.transfer(_recipient, _amount), "Transfer failed");
        return true;
    }

    function withdrawRevenue(uint256 _projectId) external projectExists(_projectId) {
        ProjectData storage p = projects[_projectId];
        require(p.status == AutonetLib.ProjectStatus.DEPLOYED, "Not generating revenue");
        require(p.totalPTsIssued > 0, "No PTs issued");

        uint256 userPT = p.projectToken.balanceOf(msg.sender);
        require(userPT > 0, "Not a PT holder");

        uint256 contractBalance = atnToken.balanceOf(address(this));
        uint256 distributable = contractBalance > p.currentFundingATN ?
            contractBalance - p.currentFundingATN : 0;

        uint256 shareRatio = (userPT * 1e18) / p.totalPTsIssued;
        uint256 totalEntitlement = (distributable * shareRatio) / 1e18;
        uint256 alreadyWithdrawn = withdrawnRevenue[_projectId][msg.sender];

        require(totalEntitlement > alreadyWithdrawn, "Nothing to withdraw");
        uint256 amount = totalEntitlement - alreadyWithdrawn;

        withdrawnRevenue[_projectId][msg.sender] += amount;
        require(atnToken.transfer(msg.sender, amount), "Transfer failed");

        emit RevenueWithdrawn(_projectId, msg.sender, amount);
    }

    function getProjectToken(uint256 _projectId) external view returns (address) {
        return address(projects[_projectId].projectToken);
    }
}
