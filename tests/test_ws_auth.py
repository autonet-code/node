"""Unit tests for atn/ws_auth.py — the WS wallet-auth primitives.

No live socket: the auth logic is isolated so it tests in pure Python."""
from __future__ import annotations

import time

import pytest

from atn import ws_auth
from atn.ws_auth import ClientSession


# ---------------------------------------------------------------------------
# is_loopback — fail-closed truth table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("remote,expected", [
    (("127.0.0.1", 7700), True),
    (("::1", 7700, 0, 0), True),
    (("::ffff:127.0.0.1", 7700), True),
    (("127.0.0.5", 7700), True),          # 127.0.0.0/8 is all loopback
    ("localhost", True),
    (("10.0.0.5", 7700), False),
    (("192.168.1.20", 7700), False),
    (("203.0.113.5", 7700), False),
    (None, False),                         # unknown origin -> NOT local
    (("not-an-ip", 7700), False),          # unparseable -> fail closed
    ((), False),                           # empty tuple -> fail closed
])
def test_is_loopback(remote, expected):
    assert ws_auth.is_loopback(remote) is expected


# ---------------------------------------------------------------------------
# nonce + challenge
# ---------------------------------------------------------------------------

def test_new_nonce_unique():
    nonces = {ws_auth.new_nonce() for _ in range(200)}
    assert len(nonces) == 200          # no collisions
    assert all(len(n) == 32 for n in nonces)   # token_hex(16) -> 32 hex chars


_CH = dict(daemon_id="0xDaemon", chain_id=128123, owner_wallet="0xOwner", conn_id="c1")


def test_challenge_text_deterministic_for_fixed_inputs():
    a = ws_auth.build_challenge_text("abc", issued_at=1000, **_CH)
    b = ws_auth.build_challenge_text("abc", issued_at=1000, **_CH)
    assert a == b
    assert "nonce: abc" in a
    assert "daemon: 0xDaemon" in a and "chain: 128123" in a
    assert "owner: 0xOwner" in a and "connection: c1" in a


def test_challenge_text_domain_separation():
    base = ws_auth.build_challenge_text("abc", issued_at=1000, **_CH)
    # Any change to nonce / daemon / chain / owner / conn changes the text, so
    # a signature can't be replayed across daemons, chains, owners or sockets.
    assert base != ws_auth.build_challenge_text("xyz", issued_at=1000, **_CH)
    assert base != ws_auth.build_challenge_text("abc", issued_at=1000, **{**_CH, "daemon_id": "0xOther"})
    assert base != ws_auth.build_challenge_text("abc", issued_at=1000, **{**_CH, "chain_id": 1})
    assert base != ws_auth.build_challenge_text("abc", issued_at=1000, **{**_CH, "owner_wallet": "0xEve"})
    assert base != ws_auth.build_challenge_text("abc", issued_at=1000, **{**_CH, "conn_id": "c2"})


# ---------------------------------------------------------------------------
# recover_signer — real signature round-trip
# ---------------------------------------------------------------------------

def test_recover_signer_roundtrip():
    from eth_account import Account
    from eth_account.messages import encode_defunct

    acct = Account.create()
    challenge = ws_auth.build_challenge_text("nonce123", issued_at=1000, **_CH)
    signed = acct.sign_message(encode_defunct(text=challenge))
    recovered = ws_auth.recover_signer(challenge, signed.signature.hex())
    assert recovered is not None
    assert recovered.lower() == acct.address.lower()


def test_recover_signer_wrong_text_yields_different_address():
    from eth_account import Account
    from eth_account.messages import encode_defunct

    acct = Account.create()
    signed = acct.sign_message(encode_defunct(text="one thing"))
    # Recovering against a DIFFERENT message yields some other address (or the
    # call still succeeds but won't match acct) — the point is it != signer.
    recovered = ws_auth.recover_signer("a different challenge", signed.signature.hex())
    assert recovered is None or recovered.lower() != acct.address.lower()


def test_recover_signer_garbage_returns_none():
    assert ws_auth.recover_signer("challenge", "") is None
    assert ws_auth.recover_signer("challenge", "0xnotasignature") is None


# ---------------------------------------------------------------------------
# ClientSession lifecycle
# ---------------------------------------------------------------------------

def test_session_defaults():
    s = ClientSession()
    assert s.authed is False
    assert s.owner is False
    assert s.root_agent_id == "orchestrator"
    assert s.scope_ids is None       # full fleet by default


def test_challenge_expiry():
    s = ClientSession()
    s.nonce = "n"
    s.nonce_issued_at = time.time()
    assert s.challenge_expired() is False
    s.nonce_issued_at = time.time() - (ws_auth.CHALLENGE_TTL_SECS + 1)
    assert s.challenge_expired() is True


def test_clear_nonce():
    s = ClientSession()
    s.nonce = "n"
    s.clear_nonce()
    assert s.nonce is None


# ---------------------------------------------------------------------------
# assert_local_key_access — loopback-only, ignores auth/owner
# ---------------------------------------------------------------------------

def test_key_access_local_listener_only():
    # local=True means the socket arrived on the privileged loopback listener.
    local = ClientSession(local=True, authed=True, owner=True)
    remote_owner = ClientSession(local=False, authed=True, owner=True)
    assert ws_auth.assert_local_key_access(local) is True
    # A proven REMOTE owner still cannot export keys — gate is on the listener.
    assert ws_auth.assert_local_key_access(remote_owner) is False


def test_key_access_ignores_is_loopback_when_not_local_listener():
    # Even if the TCP peer looks loopback (proxy on localhost), a session that
    # did NOT arrive on the privileged local listener cannot export.
    proxied = ClientSession(local=False, is_loopback=True, authed=True, owner=True)
    assert ws_auth.assert_local_key_access(proxied) is False
