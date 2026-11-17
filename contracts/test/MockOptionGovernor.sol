// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @dev Test-only stand-in for HomebaseDAO with the options counting module
///      (trustless-contracts), as AnchorElection sees it. The test sets
///      each proposal's state, snapshot and winner directly. It is its own
///      timelock: `exec` forwards a call as the timelock would. NOT deployed
///      to any real network.
contract MockOptionGovernor {
    mapping(uint256 => uint8) public stateOf;
    mapping(uint256 => uint256) public snapshotOf;
    mapping(uint256 => uint256) public winnerOf;
    mapping(uint256 => bool) public decidedOf;
    uint256 public votingDelay;
    mapping(uint256 => address) public proposerOf;
    mapping(uint256 => uint256) public etaOf;
    uint256 public proposeCount;

    function setVotingDelay(uint256 d) external { votingDelay = d; }

    function setProposal(uint256 pid, uint8 st, uint256 snapshot) external {
        stateOf[pid] = st;
        snapshotOf[pid] = snapshot;
    }

    function setWinner(uint256 pid, bool decided, uint256 index) external {
        decidedOf[pid] = decided;
        winnerOf[pid] = index;
    }

    function state(uint256 pid) external view returns (uint8) { return stateOf[pid]; }

    /// @dev As HomebaseDAO for an election call: the proposal becomes
    ///      Pending (state 0) with its snapshot one voting delay ahead. The
    ///      test moves it on with setProposal.
    function propose(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description
    ) external returns (uint256 pid) {
        pid = uint256(keccak256(abi.encode(
            targets, values, calldatas, keccak256(bytes(description)))));
        require(proposerOf[pid] == address(0), "proposed");
        proposerOf[pid] = msg.sender;
        proposeCount += 1;
        stateOf[pid] = 0;
        snapshotOf[pid] = block.timestamp + votingDelay;
    }

    function setEta(uint256 pid, uint256 eta) external { etaOf[pid] = eta; }

    function proposalEta(uint256 pid) external view returns (uint256) { return etaOf[pid]; }

    function proposalSnapshot(uint256 pid) external view returns (uint256) {
        return snapshotOf[pid];
    }

    function hashProposal(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) external pure returns (uint256) {
        return uint256(keccak256(abi.encode(targets, values, calldatas, descriptionHash)));
    }

    function winningOption(uint256 pid) external view returns (bool, uint256) {
        return (decidedOf[pid], winnerOf[pid]);
    }

    function timelock() external view returns (address) { return address(this); }

    function exec(address target, bytes calldata data) external {
        (bool ok, bytes memory ret) = target.call(data);
        if (!ok) {
            assembly { revert(add(ret, 32), mload(ret)) }
        }
    }
}
