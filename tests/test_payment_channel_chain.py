"""PaymentChannel settle and topUp through the daemon's client, on the
compiled contracts (eth_tester): the provider cashes in without ending the
channel, up to what it served (``up_to``), a voucher with nothing new is
not sent, the buyer tops up, a close pays only the rest, every payout is a
ChannelPaid log that ``channel_payments`` reads back, and the provider's
sweep settles on chain up to what was served.

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
from tests.substrate_chain_helpers import SLOT_VAULT, account_key, deploy_substrate

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "contracts" / "core"
ATN = 1_000_000
WINDOW = 3600


def _artifact(sol: str, name: str):
    path = ARTIFACTS / sol / f"{name}.json"
    if not path.exists():
        pytest.skip(f"missing artifact: {path} (run npx hardhat compile)")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _net(gross: int) -> int:
    return gross - gross * 250 // 10_000


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
    tx = substrate.functions.mintFromVault(client, 100 * ATN).transact(
        {"from": acct[SLOT_VAULT], "gas": 300_000})
    assert w3.eth.wait_for_transaction_receipt(tx).status == 1
    smc = oc.ServiceMarketClient(cfg.RPBConfig(
        rpc_url="http://eth-tester", payment_channel_address=ch_addr,
        substrate_address=sub_addr, chain_id=int(w3.eth.chain_id)))
    smc._get_web3 = lambda: w3  # type: ignore[assignment]
    yield {"w3": w3, "smc": smc, "substrate": substrate,
           "client": client, "provider": provider,
           "client_key": account_key(w3, client),
           "provider_key": account_key(w3, provider)}
    tx_nonce.reset()


def _run(coro):
    return asyncio.run(coro)


def test_settle_top_up_close_and_the_payment_log(chain):
    smc, sub = chain["smc"], chain["substrate"]
    ck, pk = chain["client_key"], chain["provider_key"]
    opened = _run(smc.open_channel(ck, chain["provider"], 5 * ATN))
    assert opened["success"], opened
    cid = opened["channel_id"]

    # Settle up to what was served: a voucher paying ahead to 3 ATN with
    # 2 ATN served pays 2 ATN, and the channel stays open.
    sig = smc.sign_voucher(ck, cid, 3 * ATN)
    res = _run(smc.settle_channel(pk, cid, 3 * ATN, sig, up_to=2 * ATN))
    assert res["success"] and res["paid"] == 2 * ATN and res["paid_total"] == 2 * ATN
    ch = _run(smc.get_channel(cid))
    assert ch["status"] == 1 and int(ch["claimed"]) == 2 * ATN
    assert sub.functions.balanceOf(chain["provider"]).call() == _net(2 * ATN)

    # The same voucher again: refused by the estimate (NothingToSettle; the
    # eth_tester provider prints the selector's bytes, not its name), never
    # sent.
    nonce = chain["w3"].eth.get_transaction_count(chain["provider"])
    again = _run(smc.settle_channel(pk, cid, 3 * ATN, sig, up_to=2 * ATN))
    assert again["success"] is False and again.get("reverted")
    selector = bytes(Web3.keccak(text="NothingToSettle()")[:4])
    assert "NothingToSettle" in again["error"] or repr(selector) in again["error"]
    assert chain["w3"].eth.get_transaction_count(chain["provider"]) == nonce

    # Top up (the allowance was spent by the open: an approve goes first).
    up = _run(smc.top_up_channel(ck, cid, 3 * ATN))
    assert up["success"] and up["deposit"] == 8 * ATN and up["approve_tx_hash"]
    assert int(_run(smc.get_channel(cid))["deposit"]) == 8 * ATN
    # Only the client tops up.
    assert _run(smc.top_up_channel(pk, cid, ATN))["success"] is False

    res = _run(smc.settle_channel(pk, cid, 6 * ATN, smc.sign_voucher(ck, cid, 6 * ATN)))
    assert res["paid"] == 4 * ATN and res["paid_total"] == 6 * ATN

    # Close pays only the rest; remainder = deposit - claimed.
    closed = _run(smc.close_channel(pk, cid, 7 * ATN, smc.sign_voucher(ck, cid, 7 * ATN)))
    assert closed["success"], closed
    ch = _run(smc.get_channel(cid))
    assert ch["status"] == 2 and int(ch["claimed"]) == 7 * ATN
    assert sub.functions.balanceOf(chain["provider"]).call() == \
        _net(2 * ATN) + _net(4 * ATN) + _net(ATN)
    # No top-up after a close.
    assert _run(smc.top_up_channel(ck, cid, ATN))["success"] is False

    pays = _run(smc.channel_payments(0))
    assert [(p["channel_id"], p["amount"], p["paid_total"]) for p in pays] == [
        (cid, 2 * ATN, 2 * ATN), (cid, 4 * ATN, 6 * ATN), (cid, ATN, 7 * ATN)]
    assert all(p["client"] == chain["client"] and p["provider"] == chain["provider"]
               for p in pays)
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
    cid = _run(smc.open_channel(ck, chain["provider"], 5 * ATN))["channel_id"]
    store, ledger = _Store(), SettlementLog()
    key_for = lambda p: pk if p.lower() == chain["provider"].lower() else None  # noqa: E731
    # A voucher paying ahead to 2 ATN, half an ATN served: below the
    # threshold.
    store.v[str(cid)] = {"cumulative": 2 * ATN, "consumed": ATN // 2,
                         "signature": smc.sign_voucher(ck, cid, 2 * ATN)}
    assert _run(settle_channels(store, smc, key_for, ledger=ledger)) == []
    # 1.5 ATN served: settled up to it, not to the voucher.
    store.v[str(cid)]["consumed"] = 3 * ATN // 2
    out = _run(settle_channels(store, smc, key_for, ledger=ledger))
    assert out[0]["action"] == "settled" and out[0]["ok"], out
    assert out[0]["reason"] == "threshold" and out[0]["amount"] == 3 * ATN // 2
    ch = _run(smc.get_channel(cid))
    assert ch["status"] == 1 and int(ch["claimed"]) == 3 * ATN // 2
    # Nothing new: nothing sent, even past the period.
    assert _run(settle_channels(store, smc, key_for, ledger=ledger,
                                settle_period_s=1, now=10 ** 12)) == []
    assert [p["amount"] for p in _run(smc.channel_payments(0))] == [3 * ATN // 2]
