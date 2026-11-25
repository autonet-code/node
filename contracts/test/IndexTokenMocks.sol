// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {Strings} from "@openzeppelin/contracts/utils/Strings.sol";

/// @dev Test only. The kinds of token a value index can hold, for the
///      payment, pool and channel tests: plain ones of 6 and 18 decimals,
///      one whose transfers return nothing (USDT on Ethereum), one that takes
///      a cut of every transfer, and one its issuer can pause. None is
///      deployed to any real network.
contract MockIndexToken is ERC20 {
    uint8 private immutable _decimals;

    constructor(string memory name_, string memory symbol_, uint8 decimals_)
        ERC20(name_, symbol_)
    {
        _decimals = decimals_;
    }

    function decimals() public view override returns (uint8) {
        return _decimals;
    }

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}

/// @dev Takes `cutBps` of every transfer (burned), as a token with a
///      transfer fee does once its owner turns the fee on.
contract CutToken is MockIndexToken {
    uint256 public cutBps;

    constructor(uint256 cutBps_) MockIndexToken("Cut", "CUT", 6) {
        cutBps = cutBps_;
    }

    function _update(address from, address to, uint256 value) internal override {
        if (from != address(0) && to != address(0) && cutBps > 0) {
            uint256 cut = (value * cutBps) / 10_000;
            super._update(from, address(0), cut);
            value -= cut;
        }
        super._update(from, to, value);
    }
}

/// @dev Its issuer can pause every transfer, as USDC's and USDT's can.
contract PausableToken is MockIndexToken {
    bool public paused;

    error TokenPaused();

    constructor() MockIndexToken("Pausable", "PAUSE", 6) {}

    function setPaused(bool p) external {
        paused = p;
    }

    function _update(address from, address to, uint256 value) internal override {
        if (paused && from != address(0)) revert TokenPaused();
        super._update(from, to, value);
    }
}

/// @dev Transfers and approvals return nothing, as USDT's do on Ethereum.
///      Callers must go through SafeERC20; a plain IERC20 call reverts on the
///      empty return data.
contract NoReturnToken {
    string public constant name = "NoReturn";
    string public constant symbol = "NORET";
    uint8 public constant decimals = 6;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
        emit Transfer(address(0), to, amount);
    }

    function transfer(address to, uint256 amount) external {
        _move(msg.sender, to, amount);
    }

    function transferFrom(address from, address to, uint256 amount) external {
        uint256 a = allowance[from][msg.sender];
        require(a >= amount, "allowance");
        if (a != type(uint256).max) allowance[from][msg.sender] = a - amount;
        _move(from, to, amount);
    }

    function approve(address spender, uint256 amount) external {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
    }

    function _move(address from, address to, uint256 amount) private {
        require(balanceOf[from] >= amount, "balance");
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
    }
}

/// @dev Test only: the DAO Registry's read surface, with an open setter.
///      `setParity` writes the key the value index uses,
///      jurisdiction.parity.<token, lowercase hex>.
contract MockRegistry {
    mapping(string => string) private _values;

    function getRegistryValue(string memory key) external view returns (string memory) {
        return _values[key];
    }

    function setValue(string memory key, string memory value) external {
        _values[key] = value;
    }

    function setParity(address token, uint256 parity) external {
        _values[string.concat("jurisdiction.parity.", Strings.toHexString(token))] =
            parity == 0 ? "" : Strings.toString(parity);
    }
}
