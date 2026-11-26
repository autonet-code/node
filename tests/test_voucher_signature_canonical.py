"""A voucher is credited only in the form PaymentChannel can cash: v 27/28
and low s (OpenZeppelin ECDSA.recover). Malleable forms that eth_account
also recovers are turned into that form, or refused."""

import os

from eth_account import Account

from atn.on_chain import _SECP256K1_N, canonical_signature


def _signed():
    acct = Account.create()
    digest = os.urandom(32)
    sig = Account.unsafe_sign_hash(digest, acct.key) if hasattr(
        Account, "unsafe_sign_hash") else Account.signHash(digest, acct.key)
    return acct.address, digest, bytes(sig.signature)


def _recover(digest, sig):
    return Account._recover_hash(digest, signature=sig)


def test_canonical_signature_is_kept():
    _addr, _digest, sig = _signed()
    assert sig[64] in (27, 28)
    assert canonical_signature(sig) == sig


def test_v_zero_one_becomes_27_28():
    addr, digest, sig = _signed()
    low_v = sig[:64] + bytes([sig[64] - 27])
    out = canonical_signature(low_v)
    assert out == sig
    assert _recover(digest, out) == addr


def test_high_s_is_flipped_to_low_s():
    addr, digest, sig = _signed()
    s_ = int.from_bytes(sig[32:64], "big")
    high = sig[:32] + (_SECP256K1_N - s_).to_bytes(32, "big") + bytes([55 - sig[64]])
    assert _recover(digest, high) == addr  # eth_account accepts it
    out = canonical_signature(high)
    assert out == sig
    assert int.from_bytes(out[32:64], "big") <= _SECP256K1_N // 2


def test_malformed_forms_are_refused():
    _addr, _digest, sig = _signed()
    assert canonical_signature(sig[:64]) is None
    assert canonical_signature(sig[:64] + bytes([37])) is None  # EIP-155 style
    assert canonical_signature(bytes(32) + sig[32:]) is None   # r = 0
