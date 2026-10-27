"""Epoch rules "2" end to end (Decision 2026-10-26): two daemon stacks
(event gossip over one hub, FederatedCloseDriver in pipelined mode,
ChainSubmissionDriver on the election path) on the real contracts: the
DAO's HomebaseDAO with options counting, AnchorElection, Substrate, a
timestamp REP token and a timelock.

- Batches gossip between the daemons and epochs close on the agreed
  schedule: cutoff = previous cutoff + length (one proposal lifecycle,
  the floor while no target share is set), empty windows sliding.
- Every input is pinned to the previous cutoff and every epoch is anchored
  by election, closes running ahead of the anchors.
- One daemon is then given a batch the other never sees: the two close
  that epoch differently, the election picks one, the other rewinds onto
  the anchored state and both keep anchoring.
"""

from __future__ import annotations

from typing import List

import pytest

eth_tester = pytest.importorskip("eth_tester")

from unittest.mock import MagicMock

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.chain_submission_driver import (
    AgentChainIdentity,
    ChainSubmissionConfig,
    ChainSubmissionDriver,
)
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.event_gossip import (
    EventBatch, EventGossip, InMemoryHub, InMemoryTransport, Keypair,
    SignedBatch, _sign,
)
from nodes.common.federated_close_driver import FederatedCloseDriver
from nodes.common.state_sync import fetch_carry_over_from_chain
from nodes.common.voice_state import _anchor_cutoff, read_voice_state
from tests.substrate_chain_helpers import (
    account_key,
    deploy_substrate,
    register_bound,
)
from tests.test_election_anchor import (
    GOV, HOMEBASE, REP, ROOT, _artifact, _deploy, _next_address, _travel, _tx,
)

RPB = "rpb-e2e"


def _now(w3) -> int:
    return int(w3.eth.get_block("latest")["timestamp"])


class Daemon:
    def __init__(self, name, w3, hub, contracts, agents, resolver, tmp,
                 operator=None, owner_wallet=""):
        sub, gov, el, rep, sub_abi = contracts
        self.name = name
        self.kp = Keypair.generate()
        self.gossip = EventGossip(rpb_address=RPB, keypair=self.kp,
                                  transport=InMemoryTransport(hub, name),
                                  world_service=MagicMock())
        self.gossip.clock = lambda: float(_now(w3))
        self.close = FederatedCloseDriver(
            gossip=self.gossip, embedding_dim=8,
            tool_registrations_path=tmp / f"{name}_regs.json",
            tool_vetting_path=tmp / f"{name}_vet.json",
            tool_positions_path=tmp / f"{name}_pos.json")
        self.close.clock = lambda: float(_now(w3))
        self.close.voice_source = lambda pin=None: read_voice_state(
            sub.address, web3=w3, rep_token_address=rep.address,
            governor_address=gov.address, pin=pin)

        def chain_source():
            n = int(sub.functions.anchorCount().call())
            anchors = []
            for i in range(max(0, n - 16), n):
                a = sub.functions.getAnchor(i).call()
                anchors.append({"number": i + 1, "epoch_id": str(a[0]),
                                "root": bytes(a[1]).hex(),
                                "cutoff": _anchor_cutoff(a)})
            return {"anchor_count": n, "anchors": anchors}

        self.close.chain_source = chain_source
        self.close.carry_source = lambda: fetch_carry_over_from_chain(sub, resolver)
        idents = [AgentChainIdentity(a, account_key(w3, a)) for a in agents]
        # The daemon operator key: anchors, proposes, votes and claims.
        # Without an explicit one, the first agent's key stands in.
        op_key = account_key(w3, operator) if operator else idents[0].private_key
        self.chain = ChainSubmissionDriver(
            config=ChainSubmissionConfig(
                substrate_address=sub.address, chain_id=w3.eth.chain_id,
                daemon_private_key=op_key,
                governor_address=gov.address,
                anchor_election_address=el.address,
                owner_wallet=owner_wallet),
            agent_chain_resolver=lambda: idents,
            blob_resolver=resolver,
            anchorer=EpochAnchorer(
                config=EpochAnchorerConfig(epoch_anchor_address=sub.address,
                                           chain_id=w3.eth.chain_id),
                web3=w3, contract_abi=sub_abi),
            tick_interval=3600, submitter_web3=w3)
        self.closed: List[str] = []

    def publish(self, tag: str):
        self.gossip.publish_events([{
            "kind": "observation_added", "seq": len(self.closed) + 1,
            "author_agent": self.name, "obs_id": f"obs_{tag}",
            "coords": [0.5, 0.1, 0.0, 0.0, 0.0, 0.0], "label": tag}])

    def step(self):
        fed = self.close.run({})
        if fed is not None:
            self.closed.append(fed.epoch_id)
            self.chain.handle_federated_close(fed)
        self.chain.tick()


def _build_net(tmp_path, households: bool = False):
    """Two daemons on one chain. ``households`` (Decision 2026-10-27): REP
    sits only on two owner wallets, which delegate their votes to their
    daemon's operator key; agents hold no REP. Otherwise every agent holds
    and self-delegates REP (the pre-decision shape)."""
    hb = _artifact(HOMEBASE)
    names = [i["name"] for i in next(
        f for f in hb["abi"] if f.get("type") == "constructor")["inputs"]]
    if "_launcher" not in names:
        pytest.skip("HomebaseDAO artifact predates the options launcher")
    rep_art = _artifact(GOV / "TestRepVotes.json")
    if not any(f.get("name") == "getPastBalance" for f in rep_art["abi"]):
        pytest.skip("TestRepVotes artifact predates getPastBalance")
    tl_art = _artifact(GOV / "TestTimelock.json")
    sub_art = _artifact(ROOT / "core" / "Substrate.sol" / "Substrate.json")
    el_art = _artifact(ROOT / "core" / "AnchorElection.sol" / "AnchorElection.json")

    w3 = Web3(EthereumTesterProvider())
    acc = w3.eth.accounts
    deployer, agents = acc[0], acc[1:5]
    operators, owners = [acc[1], acc[2]], [acc[5], acc[6]]
    rep = _deploy(w3, rep_art)
    if households:
        for owner, op in zip(owners, operators):
            _tx(w3, rep.functions.mint(owner, REP), deployer)
            _tx(w3, rep.functions.delegate(op), owner)
    else:
        for a in agents:
            _tx(w3, rep.functions.mint(a, REP), deployer)
            _tx(w3, rep.functions.delegate(a), a)
    timelock = _deploy(w3, tl_art, 0, [], ["0x" + "00" * 20], deployer)
    gov = _deploy(w3, hb, rep.address, timelock.address, "Test DAO",
                  1, 3, 0, 4, deployer)
    election_addr = _next_address(w3, deployer, ahead=1)
    sub_addr = deploy_substrate(w3, sub_art["abi"], sub_art["bytecode"],
                                deployer, governor=timelock.address,
                                anchorer=election_addr)
    sub = w3.eth.contract(address=sub_addr, abi=sub_art["abi"])
    el = _deploy(w3, el_art, sub_addr, gov.address, rep.address)
    _tx(w3, gov.functions.setOptionSource(el.address, True), deployer)
    for role in ("PROPOSER_ROLE", "CANCELLER_ROLE"):
        rid = getattr(timelock.functions, role)().call()
        _tx(w3, timelock.functions.grantRole(rid, gov.address), deployer)
    hub = InMemoryHub()
    resolver = InMemoryBlobResolver()
    contracts = (sub, gov, el, rep, sub_art["abi"])
    if households:
        # Operators acc[1], acc[2] and one agent each (acc[3], acc[4]), all
        # bound to their daemon's owner wallet.
        for i, (who, owner) in enumerate([(acc[1], owners[0]), (acc[3], owners[0]),
                                          (acc[2], owners[1]), (acc[4], owners[1])]):
            assert register_bound(w3, sub, who, f"peer-{i}".encode(),
                                  owner=owner).status == 1
        _travel(w3, 2)
        a = Daemon("A", w3, hub, contracts, [acc[3]], resolver, tmp_path,
                   operator=operators[0], owner_wallet=owners[0])
        b = Daemon("B", w3, hub, contracts, [acc[4]], resolver, tmp_path,
                   operator=operators[1], owner_wallet=owners[1])
    else:
        for i, x in enumerate(agents):
            assert register_bound(w3, sub, x, f"peer-{i}".encode()).status == 1
        _travel(w3, 2)
        a = Daemon("A", w3, hub, contracts, agents[:2], resolver, tmp_path)
        b = Daemon("B", w3, hub, contracts, agents[2:], resolver, tmp_path)
    return {"w3": w3, "sub": sub, "el": el, "a": a, "b": b, "hub": hub,
            "operators": operators, "owners": owners}


@pytest.fixture
def net(tmp_path):
    return _build_net(tmp_path)


@pytest.fixture
def household_net(tmp_path):
    return _build_net(tmp_path, households=True)


def _run(net, until_anchors: int, max_steps: int = 400, publish_every: int = 4):
    w3, sub, a, b = net["w3"], net["sub"], net["a"], net["b"]
    for i in range(max_steps):
        if i % publish_every == 0:
            (a if (i // publish_every) % 2 == 0 else b).publish(f"o{i}")
        a.step()
        b.step()
        if int(sub.functions.anchorCount().call()) >= until_anchors:
            return i
        _travel(w3, 15)     # daemons tick about every 15 s
    raise AssertionError(
        f"only {sub.functions.anchorCount().call()} anchors after {max_steps} steps")


def test_epochs_close_on_schedule_and_anchor_by_election(net):
    sub, a, b = net["sub"], net["a"], net["b"]
    _run(net, until_anchors=3)
    anchors = [sub.functions.getAnchor(i).call() for i in range(3)]
    ids = [x[0] for x in anchors]
    assert [i.split("-")[1] for i in ids] == ["1", "2", "3"]
    # Both daemons closed the anchored epochs identically.
    assert ids == a.closed[:3] == b.closed[:3]
    # Cutoffs increase by at least one lifecycle (60 + 180 + 0 s); the
    # chain links each anchor to the one before.
    cutoffs = [_anchor_cutoff(x) for x in anchors]
    assert all(c2 - c1 >= 240 for c1, c2 in zip(cutoffs, cutoffs[1:]))
    for prev, cur in zip(anchors, anchors[1:]):
        assert bytes(cur[2]) == bytes(prev[1])


def test_a_divergent_epoch_is_rewound_and_the_chain_continues(net):
    w3, sub, a, b, hub = net["w3"], net["sub"], net["a"], net["b"], net["hub"]
    _run(net, until_anchors=1)
    # A batch only daemon A ever sees, signed by a third sender.
    kp = Keypair.generate()
    batch = EventBatch(rpb_address=RPB, sender_pubkey=kp.public_key,
                       batch_seq=1, events=[{
                           "kind": "observation_added", "seq": 1,
                           "author_agent": "lonely", "obs_id": "obs_lonely",
                           "coords": [0.4, 0.2, 0.0, 0.0, 0.0, 0.0],
                           "label": "lonely"}],
                       prev_batch_hash=b"", timestamp=float(_now(w3)))
    wire = SignedBatch(batch=batch, signature=_sign(
        kp.private_key, batch.canonical_payload())).to_wire()
    a.gossip._on_message(a.gossip.topic, wire)
    _run(net, until_anchors=4)
    anchors = [sub.functions.getAnchor(i).call()[0] for i in range(4)]
    # Somewhere the two closed an epoch differently; afterwards both
    # follow the anchored chain again.
    assert a.closed != b.closed
    # The daemon whose view lost closed one epoch number twice (rewound).
    def reclosed(d):
        nums = [int(e.split("-")[1]) for e in d.closed]
        return len(nums) != len(set(nums))
    assert reclosed(a) or reclosed(b)
    # One more tick reconciles both with the chain: nothing local is at or
    # below the anchored head.
    a.step()
    b.step()
    head = int(sub.functions.anchorCount().call())
    for d in (a, b):
        assert all(e["number"] > head for e in d.close.local_epochs)
    assert len(set(anchors)) == 4
    # Every anchor chains onto the one before.
    full = [sub.functions.getAnchor(i).call() for i in range(head)]
    for prev, cur in zip(full, full[1:]):
        assert bytes(cur[2]) == bytes(prev[1])


def test_operators_carry_elections_with_delegated_owner_rep(household_net):
    """Decision 2026-10-27: REP sits on owner wallets only; each owner
    delegated its votes to its daemon's operator key. The operators alone
    are drawn, register the options and vote, so epochs still anchor."""
    net = household_net
    w3, sub, el = net["w3"], net["sub"], net["el"]
    _run(net, until_anchors=2)
    proposers = {str(log["args"]["proposer"]).lower()
                 for log in el.events.OptionRegistered.get_logs(from_block=0)}
    assert proposers, "no option was registered"
    assert proposers <= {o.lower() for o in net["operators"]}
    ids = [sub.functions.getAnchor(i).call()[0] for i in range(2)]
    assert ids == net["a"].closed[:2] == net["b"].closed[:2]
