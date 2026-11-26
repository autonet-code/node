// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @dev Test only: emits a log with the shape of Substrate's ServicePayment
///      without paying anything, so the daemon's payment check can be shown
///      to read only the jurisdiction's Substrate's own events.
contract ForgedPaymentMock {
    event ServicePayment(
        address indexed payer,
        address indexed recipient,
        address indexed token,
        uint256 amount,
        bytes32 requestId
    );

    function forge(address recipient, address token, uint256 amount, bytes32 requestId)
        external
    {
        emit ServicePayment(msg.sender, recipient, token, amount, requestId);
    }
}
