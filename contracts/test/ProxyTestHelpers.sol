// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {Clones} from "@openzeppelin/contracts/proxy/Clones.sol";
import {Substrate} from "../core/Substrate.sol";

/// @dev Test only. Importing ERC1967Proxy gives the tests its artifact: a
///      jurisdiction's Substrate is that proxy over one implementation.
contract SubstrateProxy is ERC1967Proxy {
    constructor(address implementation, bytes memory data) ERC1967Proxy(implementation, data) {}
}

/// @dev Test only: makes EIP-1167 clones the way the jurisdiction setup does
///      (clone, then initialize in the same transaction).
contract TestCloner {
    event Cloned(address clone);

    function cloneAndCall(address implementation, bytes calldata data) external returns (address clone) {
        clone = Clones.clone(implementation);
        (bool ok, bytes memory ret) = clone.call(data);
        if (!ok) {
            assembly { revert(add(ret, 32), mload(ret)) }
        }
        emit Cloned(clone);
    }

    function cloneOnly(address implementation) external returns (address clone) {
        clone = Clones.clone(implementation);
        emit Cloned(clone);
    }
}

/// @dev Test only: a later Substrate version, to upgrade a proxy to.
contract SubstrateV2Mock is Substrate {
    constructor()
        Substrate(address(1), address(1), address(0), address(1), 250, 5000)
    {}

    function version2() external pure returns (uint256) {
        return 2;
    }
}
