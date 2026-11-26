"""The owner's money readouts over WS, on the compiled Substrate
(eth_tester) with a 6-decimal and an 18-decimal index token: every
balance is per index token in base units (with the token's symbol and
decimals), earnings are the value claimed from the pools, and the sweep to
the owner moves each token held. No ATN, no supply."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("eth_tester")

from eth_account import Account
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

import atn.on_chain as oc
from atn.ws_server import ClientSession, WebSocketBridge
from nodes.common import tx_nonce
from tests.substrate_chain_helpers import account_key, deploy_substrate, index_tokens
from tests.test_service_payment_gate import _make_runtime

SUBSTRATE_JSON = (Path(__file__).resolve().parents[2] / "artifacts" / "contracts"
                  / "core" / "Substrate.sol" / "Substrate.json")
U6, U18 = 10 ** 6, 10 ** 18


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    tx_nonce.reset()
    art = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    sub = w3.eth.contract(address=deploy_substrate(w3, art["abi"], art["bytecode"],
                                                    deployer), abi=art["abi"])
    usdc, dai = index_tokens(sub)
    rt = _make_runtime(tmp_path, chain=True)
    rt._config.autonet.substrate_address = sub.address
    rt._config.autonet.network_joined = True
    monkeypatch.setattr(oc.OnChainService, "_get_web3", lambda self: w3)
    operator = Account.from_key(rt.registry.daemon_operator_key()).address
    for t, amount in ((usdc, 7 * U6), (dai, 3 * U18)):
        w3.eth.wait_for_transaction_receipt(t.functions.mint(operator, amount).transact(
            {"from": deployer, "gas": 200_000}))
    yield SimpleNamespace(w3=w3, rt=rt, sub=sub, usdc=usdc, dai=dai, operator=operator,
                          bridge=WebSocketBridge(rt), deployer=deployer)
    tx_nonce.reset()


def _ws(d, msg):
    owner = ClientSession(local=True, authed=True, owner=True)
    return asyncio.run(d.bridge._handle_message({"msg_id": "m", **msg}, owner))


def _by_token(rows):
    return {r["token"]: r for r in rows}


def test_operator_status_shows_each_index_token_and_gas(daemon):
    out = _ws(daemon, {"type": "operator_status"})
    assert out["ok"], out
    res = out["result"]
    assert "atn_balance" not in res and res["gas_balance_wei"] >= 0
    rows = _by_token(res["token_balances"])
    usdc, dai = daemon.usdc.address.lower(), daemon.dai.address.lower()
    assert (rows[usdc]["balance"], rows[usdc]["decimals"]) == (str(7 * U6), 6)
    assert (rows[dai]["balance"], rows[dai]["decimals"]) == (str(3 * U18), 18)
    assert {rows[usdc]["symbol"], rows[dai]["symbol"]} == {"T6", "T18"}


def test_fleet_voice_and_the_owners_claims(daemon):
    out = _ws(daemon, {"type": "fleet_voice", "owner": daemon.operator})
    assert out["ok"] and out["result"]["available"], out
    res = out["result"]
    assert _by_token(res["owner_tokens"])[daemon.usdc.address.lower()]["balance"] == \
        str(7 * U6)
    assert _by_token(res["fleet_tokens"])[daemon.dai.address.lower()]["balance"] == \
        str(3 * U18)
    assert (res["owner_earned_value"], res["fleet_earned_value"],
            res["total_earned_value"]) == ("0", "0", "0")
    assert not {"supply", "owner_balance", "fleet_total", "mint_supply"} & set(res)

    daemon.rt._config.autonet.owner_wallet = daemon.operator
    claim = _ws(daemon, {"type": "owner_claim_status"})["result"]
    assert claim["earned_value"] == "0" and claim["owed"] == []
    assert claim["seeded_value"] == "0"
    assert claim["claimed"] == []
    assert "claimed_mint_total" not in claim

    training = _ws(daemon, {"type": "rpb_agent_training", "address": daemon.operator})
    assert training["ok"] and training["result"]["earned_value"] == "0"
    assert _by_token(training["result"]["tokens"])[daemon.usdc.address.lower()][
        "balance"] == str(7 * U6)

    state = _ws(daemon, {"type": "rpb_state"})["result"]
    assert state["total_earned_value"] == "0" and state["pools"] == []
    assert "atn_total_supply" not in state


def test_list_index_tokens_reads_the_index(daemon):
    from atn import agent_tools as at
    at._INDEX_CACHE.clear()
    out = asyncio.run(at._list_index_tokens(daemon.rt, {}))
    assert {t["token"] for t in out["tokens"]} == {
        daemon.usdc.address.lower(), daemon.dai.address.lower()}
    assert {t["decimals"] for t in out["tokens"]} == {6, 18}


def test_the_sweep_moves_each_token_to_the_owner(daemon, monkeypatch):
    """A native agent holding both tokens: one transfer per token to the
    owner wallet (it is unbound), the operator topping up its gas first."""
    w3 = daemon.w3
    agent = w3.eth.accounts[3]
    key = account_key(w3, agent)
    owner = w3.eth.accounts[4]
    for t, amount in ((daemon.usdc, 5 * U6), (daemon.dai, 2 * U18)):
        w3.eth.wait_for_transaction_receipt(t.functions.mint(agent, amount).transact(
            {"from": daemon.deployer, "gas": 200_000}))
    # The operator pays the agent's gas: fund it first.
    w3.eth.wait_for_transaction_receipt(w3.eth.send_transaction(
        {"from": daemon.deployer, "to": daemon.operator, "value": 10 ** 18}))
    daemon.rt._config.autonet.owner_wallet = owner
    defn = SimpleNamespace(id="native-1", identity=SimpleNamespace(address=agent))
    monkeypatch.setattr(daemon.rt, "list_agents", lambda: [(defn, None)])
    monkeypatch.setattr(daemon.rt.registry, "get_agent_key", lambda aid: key)
    monkeypatch.setattr(daemon.rt.registry, "is_operator_identity", lambda ident: False)
    out = _ws(daemon, {"type": "sweep_to_owner"})
    assert out["ok"], out
    (row,) = out["result"]["swept"]
    assert row["owner"] == owner and not row.get("error"), row
    assert {(i["token"], i["amount"]) for i in row["tokens"]} == {
        (daemon.usdc.address.lower(), str(5 * U6)), (daemon.dai.address.lower(), str(2 * U18))}
    assert all(i.get("tx") and not i.get("error") for i in row["tokens"])
    assert daemon.usdc.functions.balanceOf(owner).call() == 5 * U6
    assert daemon.dai.functions.balanceOf(owner).call() == 2 * U18
    # Nothing left: the next sweep moves nothing.
    again = _ws(daemon, {"type": "sweep_to_owner"})["result"]["swept"]
    assert again[0]["skipped"] == "no index tokens"


def test_withdraw_owed_is_refused_when_nothing_is_owed(daemon):
    out = _ws(daemon, {"type": "withdraw_owed", "token": daemon.usdc.address})
    assert not out["ok"] and "owner wallet" in out["error"]
    daemon.rt._config.autonet.owner_wallet = daemon.w3.eth.accounts[4]
    assert not _ws(daemon, {"type": "withdraw_owed", "token": "usdc"})["ok"]
    out = _ws(daemon, {"type": "withdraw_owed", "token": daemon.usdc.address})
    assert not out["ok"] and "nothing is owed" in out["error"]


def test_agent_api_blocks_withdraw_owed():
    from atn.agent_api import _BLOCKED
    assert "withdraw_owed" in _BLOCKED
