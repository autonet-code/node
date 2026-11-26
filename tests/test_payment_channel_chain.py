"""PaymentChannel settle and topUp through the daemon's client, on the
compiled contracts (eth_tester), in index tokens of 6 and 18 decimals: the
provider cashes in without ending the channel, up to what it served
(``up_to``), a voucher with nothing new is not sent, the buyer tops up (an
approve of the channel's token first), a close pays only the rest, every
payout is a ChannelPaid log that ``channel_payments`` reads back with the
channel's token, a voucher signed over another token does not verify, and
the provider's sweep settles on chain up to what was served, at a
threshold that is a value.

Needs ``npx hardhat compile`` (artifacts/) and eth_tester."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from atn import config as cfg
from atn import on_chain as oc
from atn.channel_settlement import SettlementLog, settle_channels
from nodes.common import tx_nonce
from tests.substrate_chain_helpers import account_key, deploy_substrate, index_tokens

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "contracts" / "core"
U = 1_000_000          # one whole token of the 6-decimal index token
WINDOW = 3600


def _artifact(sol: str, name: str):
    path = ARTIFACTS / sol / f"{name}.json"
    if not path.exists():
        pytest.skip(f"missing artifact: {path} (run npx hardhat compile)")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _net(gross: int) -> int:
    """What the provider receives of a payout: the fee rounds up."""
    return gross - (gross * 250 + 9_999) // 10_000


@pytest.fixture
def chain():
    tx_nonce.reset()
    sub_abi, sub_bin = _artifact("Substrate.sol", "Substrate")
    ch_abi, ch_bin = _artifact("ServiceMarket.sol", "PaymentChannel")
    w3 = Web3(EthereumTesterProvider())
    acct = w3.eth.accounts
    deployer, client, provider = acct[0], acct[1], acct[2]
    sub_addr = deploy_substrate(w3, sub_abi, sub_bin, deployer)
    substrate = w3.eth.contract(address=sub_addr, abi=sub_abi)
    tx = w3.eth.contract(abi=ch_abi, bytecode=ch_bin).constructor(
        sub_addr, WINDOW).transact({"from": deployer, "gas": 6_000_000})
    ch_addr = w3.eth.wait_for_transaction_receipt(tx).contractAddress
    usdc, dai = index_tokens(substrate)          # 6 and 18 decimals
    for t, amount in ((usdc, 100 * U), (dai, 100 * 10 ** 18)):
        tx = t.functions.mint(client, amount).transact({"from": deployer, "gas": 200_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
    smc = oc.ServiceMarketClient(cfg.RPBConfig(
        rpc_url="http://eth-tester", payment_channel_address=ch_addr,
        substrate_address=sub_addr, chain_id=int(w3.eth.chain_id)))
    smc._get_web3 = lambda: w3  # type: ignore[assignment]
    yield {"w3": w3, "smc": smc, "substrate": substrate, "usdc": usdc, "dai": dai,
           "client": client, "provider": provider,
           "client_key": account_key(w3, client),
           "provider_key": account_key(w3, provider)}
    tx_nonce.reset()


def _run(coro):
    return asyncio.run(coro)


def test_settle_top_up_close_and_the_payment_log(chain):
    smc, usdc = chain["smc"], chain["usdc"]
    ck, pk = chain["client_key"], chain["provider_key"]
    tok = usdc.address.lower()
    opened = _run(smc.open_channel(ck, chain["provider"], usdc.address, 5 * U))
    assert opened["success"] and opened["approve_tx_hash"], opened
    cid = opened["channel_id"]
    ch = _run(smc.get_channel(cid))
    assert ch["token"] == tok and int(ch["deposit"]) == 5 * U
    assert usdc.functions.balanceOf(chain["client"]).call() == 95 * U

    # Settle up to what was served: a voucher paying ahead to 3 with 2
    # served pays 2, and the channel stays open.
    sig = smc.sign_voucher(ck, cid, tok, 3 * U)
    assert _run(smc.verify_voucher(cid, 3 * U, sig))["valid"]
    res = _run(smc.settle_channel(pk, cid, 3 * U, sig, up_to=2 * U))
    assert res["success"] and res["paid"] == 2 * U and res["paid_total"] == 2 * U
    ch = _run(smc.get_channel(cid))
    assert ch["status"] == 1 and int(ch["claimed"]) == 2 * U
    assert usdc.functions.balanceOf(chain["provider"]).call() == _net(2 * U)

    # The same voucher again: refused by the estimate (NothingToSettle; the
    # eth_tester provider prints the selector's bytes, not its name), never
    # sent.
    nonce = chain["w3"].eth.get_transaction_count(chain["provider"])
    again = _run(smc.settle_channel(pk, cid, 3 * U, sig, up_to=2 * U))
    assert again["success"] is False and again.get("reverted")
    selector = bytes(Web3.keccak(text="NothingToSettle()")[:4])
    assert "NothingToSettle" in again["error"] or repr(selector) in again["error"]
    assert chain["w3"].eth.get_transaction_count(chain["provider"]) == nonce

    # Top up (the allowance was spent by the open: an approve goes first).
    up = _run(smc.top_up_channel(ck, cid, 3 * U))
    assert up["success"] and up["deposit"] == 8 * U and up["approve_tx_hash"]
    assert int(_run(smc.get_channel(cid))["deposit"]) == 8 * U
    # Only the client tops up.
    assert _run(smc.top_up_channel(pk, cid, U))["success"] is False

    res = _run(smc.settle_channel(pk, cid, 6 * U, smc.sign_voucher(ck, cid, tok, 6 * U)))
    assert res["paid"] == 4 * U and res["paid_total"] == 6 * U

    # Close pays only the rest; remainder = deposit - claimed.
    closed = _run(smc.close_channel(pk, cid, 7 * U, smc.sign_voucher(ck, cid, tok, 7 * U)))
    assert closed["success"], closed
    ch = _run(smc.get_channel(cid))
    assert ch["status"] == 2 and int(ch["claimed"]) == 7 * U
    assert usdc.functions.balanceOf(chain["provider"]).call() == \
        _net(2 * U) + _net(4 * U) + _net(U)
    # No top-up after a close.
    assert _run(smc.top_up_channel(ck, cid, U))["success"] is False
    # The provider's close collected all it can: the remainder (deposit 8 -
    # claimed 7) is the client's at once, with no window to wait.
    bal = usdc.functions.balanceOf(chain["client"]).call()
    wd = _run(smc.withdraw_remainder(ck, cid))
    assert wd["success"], wd
    assert usdc.functions.balanceOf(chain["client"]).call() == bal + U
    assert int(_run(smc.get_channel(cid))["status"]) == 3

    pays = _run(smc.channel_payments(0))
    assert [(p["channel_id"], p["amount"], p["paid_total"]) for p in pays] == [
        (cid, 2 * U, 2 * U), (cid, 4 * U, 6 * U), (cid, U, 7 * U)]
    assert all(p["client"] == chain["client"] and p["provider"] == chain["provider"]
               and p["token"] == tok for p in pays)
    # Filtered to a provider: the same three; another provider has none.
    assert _run(smc.channel_payments(0, provider=chain["provider"])) == pays
    assert _run(smc.channel_payments(0, provider=chain["client"])) == []
    assert pays == sorted(pays, key=lambda p: p["block"])
    # Pinned to a block before the close: the close's payout is not in it.
    assert len(_run(smc.channel_payments(0, pays[-1]["block"] - 1))) == 2


class _Store:
    def __init__(self):
        self.v = {}

    def channel_vouchers(self):
        return {k: dict(x) for k, x in self.v.items()}

    def forget_channel_voucher(self, cid):
        self.v.pop(str(cid), None)

    def has_pending_voucher(self, cid):
        return False


def test_the_providers_sweep_settles_on_chain_and_sends_nothing_twice(chain):
    smc = chain["smc"]
    ck, pk = chain["client_key"], chain["provider_key"]
    usdc = chain["usdc"]
    tok = usdc.address.lower()
    cid = _run(smc.open_channel(ck, chain["provider"], usdc.address, 5 * U))["channel_id"]
    store, ledger = _Store(), SettlementLog()
    key_for = lambda p: pk if p.lower() == chain["provider"].lower() else None  # noqa: E731
    # A voucher paying ahead to 2 U, half an U served: below the
    # threshold.
    store.v[str(cid)] = {"cumulative": 2 * U, "consumed": U // 2,
                         "signature": smc.sign_voucher(ck, cid, tok, 2 * U)}
    assert _run(settle_channels(store, smc, key_for, ledger=ledger)) == []
    # 1.5 U served: settled up to it, not to the voucher.
    store.v[str(cid)]["consumed"] = 3 * U // 2
    out = _run(settle_channels(store, smc, key_for, ledger=ledger))
    assert out[0]["action"] == "settled" and out[0]["ok"], out
    assert out[0]["reason"] == "threshold" and out[0]["amount"] == 3 * U // 2
    ch = _run(smc.get_channel(cid))
    assert ch["status"] == 1 and int(ch["claimed"]) == 3 * U // 2
    # Nothing new: nothing sent, even past the period.
    assert _run(settle_channels(store, smc, key_for, ledger=ledger,
                                settle_period_s=1, now=10 ** 12)) == []
    assert [p["amount"] for p in _run(smc.channel_payments(0))] == [3 * U // 2]


def test_a_channel_in_an_18_decimal_token(chain):
    """The same rail in DAI (18 decimals): amounts are base units of the
    channel's token, and the sweep's threshold is a value, so 1 DAI (1e18
    base units, worth 1e6 value units) settles at once."""
    smc, dai = chain["smc"], chain["dai"]
    ck, pk = chain["client_key"], chain["provider_key"]
    one = 10 ** 18
    tok = dai.address.lower()
    cid = _run(smc.open_channel(ck, chain["provider"], dai.address, 5 * one))["channel_id"]
    assert _run(smc.get_channel(cid))["token"] == tok
    assert _run(smc.value_of(dai.address, one)) == U
    store, ledger = _Store(), SettlementLog()
    key_for = lambda p: pk if p.lower() == chain["provider"].lower() else None  # noqa: E731
    store.v[str(cid)] = {"cumulative": 2 * one, "consumed": one,
                         "signature": smc.sign_voucher(ck, cid, tok, 2 * one)}
    out = _run(settle_channels(store, smc, key_for, ledger=ledger))
    assert out[0]["reason"] == "threshold" and out[0]["token"] == tok and out[0]["ok"]
    assert dai.functions.balanceOf(chain["provider"]).call() == _net(one)


def test_a_voucher_signed_over_another_token_does_not_verify(chain):
    """A voucher names its channel's token: one signed for the same channel
    and amount but another token recovers to someone else, so the provider
    refuses it and the contract would too."""
    smc, usdc, dai = chain["smc"], chain["usdc"], chain["dai"]
    ck, pk = chain["client_key"], chain["provider_key"]
    cid = _run(smc.open_channel(ck, chain["provider"], usdc.address, 5 * U))["channel_id"]
    wrong = smc.sign_voucher(ck, cid, dai.address.lower(), 2 * U)
    res = _run(smc.verify_voucher(cid, 2 * U, wrong))
    assert res["valid"] is False and "not the channel client" in res["reason"]
    assert res["token"] == usdc.address.lower()
    settled = _run(smc.settle_channel(pk, cid, 2 * U, wrong))
    assert settled["success"] is False and settled.get("reverted")
    right = smc.sign_voucher(ck, cid, usdc.address.lower(), 2 * U)
    assert _run(smc.verify_voucher(cid, 2 * U, right))["valid"]


def test_a_channel_opens_only_in_an_index_token(chain):
    """A token outside the value index: the contract refuses the open, so
    the client does not send it."""
    from tests.substrate_chain_helpers import artifact, _deploy
    w3, smc = chain["w3"], chain["smc"]
    stray = _deploy(w3, artifact("test/IndexTokenMocks.sol/MockIndexToken"),
                    w3.eth.accounts[0], "S", "S", 6)
    w3.eth.wait_for_transaction_receipt(stray.functions.mint(
        chain["client"], 10 * U).transact({"from": w3.eth.accounts[0], "gas": 200_000}))
    assert _run(smc.accepts_token(stray.address)) is False
    out = _run(smc.open_channel(chain["client_key"], chain["provider"], stray.address, U))
    assert out["success"] is False and out.get("reverted")
