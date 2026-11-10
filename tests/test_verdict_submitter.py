"""The daemon shows ToolBond the malware verdicts of anchored closes
(Decisions 2026-11-09 and 2026-11-10): every concluded leaf goes to
ToolBond.conclude with its proof, which settles a malware verdict at once.
There is no appeal window and nothing left to settle afterwards."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from nodes.common.authoritative_encoding import encode_agent_mint_blob
from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.mint_merkle import mint_merkle_root, verify_concluded_proof
from nodes.common.verdict_submitter import VerdictSubmitter

CLEAN, BAD = "aa" * 32, "bb" * 32
MINT = {"0x" + "11" * 20: 2.0}
CONCLUDED = {CLEAN: 1000, BAD: 7000}


class _Call:
    def __init__(self, fn, *args):
        self.fn, self.args = fn, args

    def call(self):
        return self.fn(*self.args)


class FakeBond:
    def __init__(self):
        self.verdict: Dict[bytes, list] = {}
        self.sent: List[tuple] = []
        outer = self

        class F:
            def verdicts(self, d):
                return _Call(lambda: tuple(outer.verdict.get(d, [0, 0, False, 0, 0])))

            def conclude(self, d, index, bps, proof):
                return ("conclude", d, index, bps, proof)
        self.functions = F()


class FakeSubstrate:
    def __init__(self, cid, root):
        anchors = [("e1",), ("e2",), ("e3",)]

        class F:
            def anchorCount(self):
                return _Call(lambda: len(anchors))

            def getAnchor(self, i):
                return _Call(lambda: anchors[i])

            def getAnchorByEpochId(self, eid):
                return _Call(lambda: ("e2", b"", b"", b"", cid, b"", "", 0, 0, root))
        self.functions = F()


def _setup():
    resolver = InMemoryBlobResolver()
    cid = resolver.put(encode_agent_mint_blob(MINT, concluded=CONCLUDED))
    root = mint_merkle_root(MINT, concluded=CONCLUDED)
    bond = FakeBond()
    w3 = SimpleNamespace(to_checksum_address=lambda a: a,
                         eth=SimpleNamespace(contract=lambda address, abi: bond,
                                             get_block=lambda b: {"timestamp": 0}))

    def send(call: Any) -> Dict[str, Any]:
        bond.sent.append(call)
        _, d, index, bps, proof = call
        assert index == 1                       # e2's anchor index
        assert verify_concluded_proof(root, d.hex(), bps, proof)
        bond.verdict[d] = [100, 1 if bps * 2 > 10_000 else 0, False, bps * 10 ** 14, 0]
        return {"success": True}

    vs = VerdictSubmitter(w3=w3, tool_bond_address="0xb", substrate=FakeSubstrate(cid, root),
                          resolver=resolver, send=send)
    return vs, bond


def test_every_concluded_leaf_is_shown_once():
    vs, bond = _setup()
    rows = vs.conclude_epoch("e2")
    assert {r["digest"]: r.get("concluded") for r in rows} == {CLEAN: True, BAD: True}
    assert {r["digest"]: r["malware"] for r in rows} == {CLEAN: False, BAD: True}
    assert [c[3] for c in bond.sent] == [1000, 7000]
    # A second pass (another daemon concluded first, or a retry) sends nothing.
    bond.sent.clear()
    rows = vs.conclude_epoch("e2")
    assert bond.sent == [] and {r.get("concluded") for r in rows} == {"already"}


def test_a_conclude_lost_to_another_daemon_counts_as_concluded():
    """Every operator shows ToolBond the same leaves: when another daemon's
    conclude lands between the read and this one's transaction, the revert
    is no error."""
    vs, bond = _setup()
    inner = vs._send

    def racing(call: Any) -> Dict[str, Any]:
        inner(call)                                  # theirs lands
        return {"success": False, "error": "would revert: WrongStage()"}
    vs._send = racing
    rows = vs.conclude_epoch("e2")
    assert {r["digest"]: r.get("concluded") for r in rows} == {CLEAN: "already",
                                                               BAD: "already"}
    assert not any(r.get("error") for r in rows)


def test_the_verdict_abi_matches_toolbond():
    """conclude, verdicts, getRound and the review calls as the daemon makes
    them are trustless-contracts ToolBond's (8dcc1be)."""
    import json
    from pathlib import Path

    import pytest
    from atn.runtime.review_batch import TOOL_BOND_REVIEW_ABI
    from nodes.common.verdict_submitter import TOOL_BOND_VERDICT_ABI
    art = Path("C:/code/dao/trustless-contracts/artifacts/contracts/"
               "ToolBond.sol/ToolBond.json")
    if not art.exists():
        pytest.skip("trustless-contracts ToolBond artifact not built")
    fns = {e["name"]: e for e in json.loads(art.read_text())["abi"]
           if e["type"] == "function"}

    def shape(e):
        return ([i["type"] for i in e["inputs"]],
                [o["type"] for o in e.get("outputs", [])])
    mine = {e["name"]: e for e in TOOL_BOND_VERDICT_ABI + TOOL_BOND_REVIEW_ABI}
    for name in ("conclude", "verdicts", "getRound", "claim", "finalize"):
        assert shape(mine[name]) == shape(fns[name]), name
    assert "settleMalware" not in fns and "claimBondShare" not in fns
