// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @dev Test-only stand-in for the DAO's RepToken (C:/code/dao/
///      trustless-contracts/contracts/RepToken.sol) read surface used by
///      nodes/common/voice_state.py: timestamp-clocked ERC20Votes-style
///      historical reads getPastVotes / getPastTotalSupply (mode=timestamp).
///      NOT deployed to any real network. It stores one checkpoint per
///      account keyed by the timestamp it was written at; a read at
///      `timepoint` returns the latest value with checkpoint ts <= timepoint
///      (OZ upper-lookup semantics), so pinning to a prior anchor timestamp
///      is honored exactly.
contract MockRepToken {
    struct Point { uint256 ts; uint256 value; }

    mapping(address => Point[]) private _votes;
    Point[] private _supply;

    function decimals() external pure returns (uint8) { return 18; }
    function CLOCK_MODE() external pure returns (string memory) { return "mode=timestamp"; }
    function clock() external view returns (uint48) { return uint48(block.timestamp); }

    /// @dev Write a vote checkpoint for `account` at the CURRENT block
    ///      timestamp, and bump the total-supply checkpoint by the delta.
    function setVotes(address account, uint256 value) external {
        uint256 ts = block.timestamp;
        uint256 prev = _latest(_votes[account]);
        _votes[account].push(Point(ts, value));
        uint256 supplyPrev = _latest(_supply);
        // value replaces prev for this account: supply += (value - prev)
        _supply.push(Point(ts, supplyPrev + value - prev));
    }

    function _latest(Point[] storage pts) private view returns (uint256) {
        if (pts.length == 0) return 0;
        return pts[pts.length - 1].value;
    }

    function _at(Point[] storage pts, uint256 timepoint) private view returns (uint256) {
        // Upper-lookup: latest point with ts <= timepoint.
        uint256 v = 0;
        for (uint256 i = 0; i < pts.length; i++) {
            if (pts[i].ts <= timepoint) v = pts[i].value; else break;
        }
        return v;
    }

    function getPastVotes(address account, uint256 timepoint) external view returns (uint256) {
        return _at(_votes[account], timepoint);
    }

    function getPastTotalSupply(uint256 timepoint) external view returns (uint256) {
        return _at(_supply, timepoint);
    }
}
