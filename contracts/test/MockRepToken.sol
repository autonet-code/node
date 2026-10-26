// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @dev Test-only stand-in for the DAO's RepToken (C:/code/dao/
///      trustless-contracts/contracts/RepToken.sol) read surface used by
///      nodes/common/voice_state.py: timestamp-clocked ERC20Votes-style
///      historical reads getPastBalance / getPastVotes / getPastTotalSupply
///      (mode=timestamp).
///      NOT deployed to any real network. It stores one checkpoint per
///      account keyed by the timestamp it was written at; a read at
///      `timepoint` returns the latest value with checkpoint ts <= timepoint
///      (OZ upper-lookup semantics), so pinning to a prior anchor timestamp
///      is honored exactly.
contract MockRepToken {
    struct Point { uint256 ts; uint256 value; }

    mapping(address => Point[]) private _votes;
    mapping(address => Point[]) private _balances;
    Point[] private _supply;

    function decimals() external pure returns (uint8) { return 18; }
    function CLOCK_MODE() external pure returns (string memory) { return "mode=timestamp"; }
    function clock() external view returns (uint48) { return uint48(block.timestamp); }

    /// @dev Write a balance checkpoint for `account` at the CURRENT block
    ///      timestamp (self-delegated, so its votes too), and bump the
    ///      total-supply checkpoint by the delta.
    function setVotes(address account, uint256 value) external {
        uint256 ts = block.timestamp;
        uint256 prev = _latest(_balances[account]);
        _balances[account].push(Point(ts, value));
        _votes[account].push(Point(ts, value));
        uint256 supplyPrev = _latest(_supply);
        // value replaces prev for this account: supply += (value - prev)
        _supply.push(Point(ts, supplyPrev + value - prev));
    }

    /// @dev Move voting power only (a delegation): balances and supply
    ///      stay as they are.
    function setDelegatedVotes(address account, uint256 value) external {
        _votes[account].push(Point(block.timestamp, value));
    }

    /// @dev Bump total supply WITHOUT any account's votes (models REP held
    ///      by a holder that never delegated: counts in supply, votes nowhere).
    function addUndelegatedSupply(uint256 amount) external {
        _supply.push(Point(block.timestamp, _latest(_supply) + amount));
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

    function getPastBalance(address account, uint256 timepoint) external view returns (uint256) {
        return _at(_balances[account], timepoint);
    }

    function getPastVotes(address account, uint256 timepoint) external view returns (uint256) {
        return _at(_votes[account], timepoint);
    }

    function getPastTotalSupply(uint256 timepoint) external view returns (uint256) {
        return _at(_supply, timepoint);
    }
}
