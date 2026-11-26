"""Direct service payments in the value index's tokens, through the
daemon's client (``OnChainService.pay_for_service``) and the provider's
check of one (``verify_service_payment``), on the compiled Substrate
(eth_tester) with a 6-decimal and an 18-decimal index token.

Validates:

  1. The payer approves Substrate on the token, then pays: the recipient
     gets the gross less the fee (rounded up), Substrate the fee, split
     into the tool share and the treasury's; ServicePayment and ServiceFee
     carry the token and the gross.
  2. The provider's check accepts a payment in the price list's token with
     enough of it, and refuses one in another token whatever its number
     (a DAI payment against a USDC price), too little, another recipient,
     another payer, and a log of the same shape from another contract.
  3. A token outside the index: nothing is sent.
  4. Gas: the floors cover a first and a later payment and an approve.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Tuple

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from atn import config as cfg
from atn import on_chain as oc
from nodes.common import tx_nonce
from tests.substrate_chain_helpers import (
    _deploy,
    account_key,
    artifact,
    deploy_substrate,
    index_tokens,
)

SUBSTRATE_JSON = (Path(__file__).resolve().parents[1] / "artifacts" / "contracts"
                  / "core" / "Substrate.sol" / "Substrate.json")
U6, U18 = 10 ** 6, 10 ** 18


def _load_substrate() -> Tuple[list, str]:
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _fee_parts(gross: int) -> Tuple[int, int, int]:
    """(fee, tool share, to treasury): the contract's math, fee rounded up
    (serviceFeeBps 250, feeTreasuryBps 5000)."""
    fee = (gross * 250 + 9_999) // 10_000
    to_treasury = fee * 5_000 // 10_000
    return fee, fee - to_treasury, to_treasury


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def chain():
    tx_nonce.reset()
    abi, bytecode = _load_substrate()
    w3 = Web3(EthereumTesterProvider())
    acct = w3.eth.accounts
    deployer, payer, provider = acct[0], acct[1], acct[2]
    addr = deploy_substrate(w3, abi, bytecode, deployer)
    sub = w3.eth.contract(address=addr, abi=abi)
    usdc, dai = index_tokens(sub)
    for t, amount in ((usdc, 1_000 * U6), (dai, 1_000 * U18)):
        tx = t.functions.mint(payer, amount).transact({"from": deployer, "gas": 200_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
    svc = oc.OnChainService(cfg.RPBConfig(
        rpc_url="http://eth-tester", substrate_address=addr,
        chain_id=int(w3.eth.chain_id)))
    svc._get_web3 = lambda: w3  # type: ignore[assignment]
    yield {"w3": w3, "svc": svc, "sub": sub, "usdc": usdc, "dai": dai,
           "payer": payer, "provider": provider, "deployer": deployer,
           "payer_key": account_key(w3, payer)}
    tx_nonce.reset()


def _pay(chain, token, gross, rid="0x" + "11" * 32, recipient=None):
    return _run(chain["svc"].pay_for_service(
        chain["payer_key"], token.address, recipient or chain["provider"], gross, rid))


def _verify(chain, res, *, token, min_amount, rid="0x" + "11" * 32, recipient=None,
            payer=None):
    return _run(chain["svc"].verify_service_payment(
        rid, recipient or chain["provider"], min_amount, token.address.lower(),
        payer=payer if payer is not None else chain["payer"], tx_hash=res["tx_hash"]))


def test_a_payment_approves_then_pays_net_of_the_fee(chain):
    w3, sub, usdc = chain["w3"], chain["sub"], chain["usdc"]
    gross = 7 * U6 + 3
    fee, tool, treasury = _fee_parts(gross)
    res = _pay(chain, usdc, gross)
    assert res["success"] and res["approve_tx_hash"], res
    assert usdc.functions.balanceOf(chain["payer"]).call() == 1_000 * U6 - gross
    assert usdc.functions.balanceOf(chain["provider"]).call() == gross - fee
    assert usdc.functions.balanceOf(sub.address).call() == fee
    assert sub.functions.toolShareTotal(usdc.address).call() == tool
    assert sub.functions.treasuryOwed(usdc.address).call() == treasury
    receipt = w3.eth.get_transaction_receipt(res["tx_hash"])
    pay = sub.events.ServicePayment().process_receipt(receipt)[0]["args"]
    assert (pay["payer"], pay["recipient"], pay["token"], pay["amount"]) == (
        chain["payer"], chain["provider"], usdc.address, gross)
    feev = sub.events.ServiceFee().process_receipt(receipt)[0]["args"]
    assert (feev["token"], feev["amount"], feev["toolShare"], feev["toTreasury"]) == (
        usdc.address, gross, tool, treasury)
    # The allowance was used up: the next payment approves again.
    again = _pay(chain, usdc, U6, rid="0x" + "12" * 32)
    assert again["success"] and again["approve_tx_hash"]


def test_an_18_decimal_token_pays_the_same_way(chain):
    dai = chain["dai"]
    gross = 3 * U18
    fee, _, _ = _fee_parts(gross)
    res = _pay(chain, dai, gross)
    assert res["success"], res
    assert dai.functions.balanceOf(chain["provider"]).call() == gross - fee
    ok = _verify(chain, res, token=dai, min_amount=gross)
    assert ok["verified"] and ok["token"] == dai.address.lower(), ok


def test_the_check_accepts_the_price_lists_token_and_amount(chain):
    usdc = chain["usdc"]
    res = _pay(chain, usdc, 5 * U6)
    ok = _verify(chain, res, token=usdc, min_amount=5 * U6)
    assert ok["verified"] and ok["reason"] == "ok", ok
    assert (ok["amount"], ok["token"], ok["payer"]) == (
        5 * U6, usdc.address.lower(), chain["payer"])
    assert isinstance(ok["paid_at"], int)


def test_a_payment_in_another_token_is_refused_whatever_its_number(chain):
    """A DAI payment against a USDC price: 5e18 DAI base units is far more
    than 5e6 as a number, and it is still refused, as is a USDC payment
    against a DAI price."""
    usdc, dai = chain["usdc"], chain["dai"]
    in_dai = _pay(chain, dai, 5 * U18)
    out = _verify(chain, in_dai, token=usdc, min_amount=5 * U6)
    assert out["verified"] is False and "paid in token" in out["reason"], out
    in_usdc = _pay(chain, usdc, 5 * U6, rid="0x" + "22" * 32)
    out = _verify(chain, in_usdc, token=dai, min_amount=5, rid="0x" + "22" * 32)
    assert out["verified"] is False and "paid in token" in out["reason"], out


def test_the_check_refuses_too_little_another_recipient_or_payer(chain):
    usdc = chain["usdc"]
    res = _pay(chain, usdc, 5 * U6)
    low = _verify(chain, res, token=usdc, min_amount=5 * U6 + 1)
    assert low["verified"] is False and "min_amount" in low["reason"]
    other = _verify(chain, res, token=usdc, min_amount=1,
                    recipient=chain["w3"].eth.accounts[3])
    assert other["verified"] is False and "matched" in other["reason"]
    wrong_payer = _verify(chain, res, token=usdc, min_amount=1,
                          payer=chain["w3"].eth.accounts[4])
    assert wrong_payer["verified"] is False and wrong_payer["reason"] == "payer mismatch"
    wrong_rid = _verify(chain, res, token=usdc, min_amount=1, rid="0x" + "99" * 32)
    assert wrong_rid["verified"] is False
    assert _run(chain["svc"].verify_service_payment(
        "0x" + "11" * 32, chain["provider"], 1, usdc.address))["reason"].startswith(
        "tx_hash required")


def test_a_forged_log_from_another_contract_is_ignored(chain):
    """Any contract can emit a log shaped like ServicePayment; only the
    jurisdiction's Substrate's own count."""
    w3, usdc = chain["w3"], chain["usdc"]
    forger = _deploy(w3, artifact("test/ForgedPaymentMock.sol/ForgedPaymentMock"),
                     chain["deployer"])
    rid = "0x" + "33" * 32
    tx = forger.functions.forge(chain["provider"], usdc.address, 10 ** 30,
                                bytes.fromhex("33" * 32)).transact(
        {"from": chain["payer"], "gas": 200_000})
    out = _run(chain["svc"].verify_service_payment(
        rid, chain["provider"], 1, usdc.address.lower(), payer=chain["payer"],
        tx_hash=tx))
    assert out["verified"] is False and "no ServicePayment event" in out["reason"]


def test_a_token_outside_the_index_sends_nothing(chain):
    w3 = chain["w3"]
    stray = _deploy(w3, artifact("test/IndexTokenMocks.sol/MockIndexToken"),
                    chain["deployer"], "S", "S", 6)
    w3.eth.wait_for_transaction_receipt(stray.functions.mint(
        chain["payer"], 10 * U6).transact({"from": chain["deployer"], "gas": 200_000}))
    nonce = w3.eth.get_transaction_count(chain["payer"])
    res = _pay(chain, stray, U6)
    assert res["success"] is False and "not in the value index" in res["error"]
    assert w3.eth.get_transaction_count(chain["payer"]) == nonce
    assert _run(chain["svc"].accepts_token(stray.address)) is False
    assert _run(chain["svc"].accepts_token(chain["usdc"].address)) is True


def test_zero_or_a_zero_recipient_is_not_sent(chain):
    usdc = chain["usdc"]
    zero = _pay(chain, usdc, 0)
    assert zero["success"] is False
    res = _pay(chain, usdc, U6, recipient="0x" + "00" * 20)
    assert res["success"] is False and res.get("reverted")


def test_the_gas_floors_cover_a_payment(chain):
    """Measured: an approve, a first payment in a token (it lists the token
    and opens its checkpoints) and a later one. Each stays under the floor
    pay_for_service sends with, so an estimate that undercounts never runs
    a payment out of gas."""
    w3, usdc = chain["w3"], chain["usdc"]
    first = _pay(chain, usdc, 5 * U6)
    later = _pay(chain, usdc, 5 * U6, rid="0x" + "44" * 32)
    used = [w3.eth.get_transaction_receipt(h)["gasUsed"]
            for h in (first["approve_tx_hash"], first["tx_hash"], later["tx_hash"])]
    approve, pay_first, pay_later = used
    assert approve < oc.APPROVE_GAS_FLOOR
    assert pay_later < pay_first < oc.PAY_GAS_FLOOR, used


def test_the_readouts_are_per_index_token(chain):
    """What the daemon shows of money: the index's tokens, an address's
    balance of each, a provider's earnings per token from the payment logs,
    the pools per token and the value claimed (none here)."""
    svc, usdc, dai = chain["svc"], chain["usdc"], chain["dai"]
    assert _pay(chain, usdc, 5 * U6)["success"]
    assert _pay(chain, dai, 2 * U18, rid="0x" + "55" * 32)["success"]

    index = _run(svc.index_tokens())
    assert [(t["token"], t["decimals"], t["parity"]) for t in index] == sorted(
        [(usdc.address.lower(), 6, str(10 ** 30)), (dai.address.lower(), 18, str(10 ** 18))])
    assert {t["symbol"] for t in index} == {"T6", "T18"}

    bal = _run(svc.get_agent_balances(chain["payer"]))
    by = {r["token"]: r for r in bal["tokens"]}
    assert by[usdc.address.lower()]["balance"] == str(1_000 * U6 - 5 * U6)
    assert by[dai.address.lower()]["balance"] == str(1_000 * U18 - 2 * U18)
    assert by[dai.address.lower()]["decimals"] == 18 and by[dai.address.lower()]["accepted"]
    assert bal["earned_value"] == "0" and int(bal["gas_balance"]) > 0

    rec = _run(svc.get_provider_record(chain["provider"]))
    assert {(e["token"], e["amount"]) for e in rec["service_earnings"]} == {
        (usdc.address.lower(), str(5 * U6)), (dai.address.lower(), str(2 * U18))}
    assert rec["earned_value"] == "0" and rec["registered_at"] == 0
    assert _run(svc.get_provider_record(chain["payer"]))["service_earnings"] == []

    state = _run(svc.get_substrate_state())
    pools = {p["token"]: p for p in state["pools"]}
    assert pools[usdc.address.lower()]["tool_share_total"] == str(_fee_parts(5 * U6)[1])
    assert pools[dai.address.lower()]["claimed_total"] == "0"
    assert state["total_earned_value"] == "0"

    claim = _run(svc.get_claim_status(chain["payer"]))
    assert claim == {"earner": chain["payer"], "earned_value": "0", "seeded_value": "0",
                     "owed": [], "claimed": []}

    fleet = _run(svc.get_fleet_voice(chain["payer"]))
    assert {r["token"]: r["balance"] for r in fleet["fleet_tokens"]} == {
        r["token"]: r["balance"] for r in bal["tokens"]}
    assert fleet["agents"] == [] and fleet["total_earned_value"] == "0"
