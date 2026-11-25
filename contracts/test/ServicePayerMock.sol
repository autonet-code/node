// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface ISubstratePay {
    function payForService(address token, address recipient, uint256 gross, bytes32 requestId) external;
}

interface IApproveOnly {
    function approve(address spender, uint256 amount) external returns (bool);
}

/// @dev Test-only: a CONTRACT payer (stand-in for a DAO contract that pays
///      a service from contract context). Pins that payForService has no
///      EOA-only or agent-only payer check.
contract ServicePayerMock {
    function pay(address substrate, address token, address recipient, uint256 gross, bytes32 requestId)
        external
    {
        IApproveOnly(token).approve(substrate, gross);
        ISubstratePay(substrate).payForService(token, recipient, gross, requestId);
    }
}
