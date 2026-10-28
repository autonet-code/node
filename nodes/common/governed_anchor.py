"""Governed anchoring: an epoch anchor as an ordinary DAO proposal.

Decision 2026-10-16 (with the DAO side): ``Substrate.anchor(...)`` is
governor-only (the DAO timelock), so an anchor goes through the DAO
Governor like any other decision:

    propose -> (votingDelay) -> vote For -> Succeeded -> queue
            -> (timelock delay) -> execute -> Substrate.anchor runs

The DAO Governor decides a proposal early once the outcome cannot change
(For above half of total REP at the snapshot, or Against at half), so with
the anchor agent's delegated REP an anchor lands one voting delay plus one
timelock delay after the close. Every daemon derives the identical proposal
from the bit-identical close: same target, calldata and description
(``Anchor epoch <epochId> <epochRoot>``) give the same proposal id, so a
second propose reverts and that daemon votes on the existing proposal.

Dev networks may deploy Substrate with an EOA as governor. When that EOA is
this daemon's key, :class:`GovernedAnchorer` calls ``anchor`` directly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .tx_nonce import next_nonce

logger = logging.getLogger(__name__)

# OZ IGovernor.ProposalState
PENDING, ACTIVE, CANCELED, DEFEATED, SUCCEEDED, QUEUED, EXPIRED, EXECUTED = range(8)
STATE_NAMES = ["pending", "active", "canceled", "defeated", "succeeded",
               "queued", "expired", "executed"]
VOTE_FOR = 1

_ADDR_ARRAY = {"internalType": "address[]", "name": "targets", "type": "address[]"}
_UINT_ARRAY = {"internalType": "uint256[]", "name": "values", "type": "uint256[]"}
_BYTES_ARRAY = {"internalType": "bytes[]", "name": "calldatas", "type": "bytes[]"}
_PID = {"internalType": "uint256", "name": "proposalId", "type": "uint256"}
_U256_OUT = [{"internalType": "uint256", "name": "", "type": "uint256"}]


def _fn(name, inputs, outputs=None, mutability="view"):
    return {"type": "function", "name": name, "inputs": inputs,
            "outputs": outputs or [], "stateMutability": mutability}


GOVERNOR_ABI = [
    _fn("hashProposal", [_ADDR_ARRAY, _UINT_ARRAY, _BYTES_ARRAY,
                         {"internalType": "bytes32", "name": "descriptionHash",
                          "type": "bytes32"}], _U256_OUT, "pure"),
    _fn("propose", [_ADDR_ARRAY, _UINT_ARRAY, _BYTES_ARRAY,
                    {"internalType": "string", "name": "description",
                     "type": "string"}], _U256_OUT, "nonpayable"),
    _fn("queue", [_ADDR_ARRAY, _UINT_ARRAY, _BYTES_ARRAY,
                  {"internalType": "bytes32", "name": "descriptionHash",
                   "type": "bytes32"}], _U256_OUT, "nonpayable"),
    _fn("execute", [_ADDR_ARRAY, _UINT_ARRAY, _BYTES_ARRAY,
                    {"internalType": "bytes32", "name": "descriptionHash",
                     "type": "bytes32"}], _U256_OUT, "payable"),
    _fn("castVote", [_PID, {"internalType": "uint8", "name": "support",
                            "type": "uint8"}], _U256_OUT, "nonpayable"),
    _fn("state", [_PID], [{"internalType": "uint8", "name": "", "type": "uint8"}]),
    _fn("hasVoted", [_PID, {"internalType": "address", "name": "account",
                            "type": "address"}],
        [{"internalType": "bool", "name": "", "type": "bool"}]),
    _fn("proposalSnapshot", [_PID], _U256_OUT),
    _fn("proposalEta", [_PID], _U256_OUT),
    _fn("proposalThreshold", [], _U256_OUT),
    _fn("clock", [], [{"internalType": "uint48", "name": "", "type": "uint48"}]),
    _fn("getVotes", [{"internalType": "address", "name": "account", "type": "address"},
                     {"internalType": "uint256", "name": "timepoint", "type": "uint256"}],
        _U256_OUT),
]


def anchor_description(epoch_id: str, epoch_root: bytes) -> str:
    """The proposal description every daemon uses (agreed with the DAO side)."""
    return f"Anchor epoch {epoch_id} 0x{bytes(epoch_root).hex()}"


@dataclass
class Voter:
    address: str
    private_key: str


@dataclass
class AnchorProposal:
    epoch_id: str
    targets: List[str]
    values: List[int]
    calldatas: List[bytes]
    description: str
    description_hash: bytes
    proposal_id: int
    created: float = field(default_factory=time.monotonic)
    last_state: str = "nonexistent"
    tx_hashes: List[str] = field(default_factory=list)


class GovernedAnchorer:
    """Drives one epoch anchor through the DAO Governor (or anchors
    directly when this daemon's key is Substrate's governor)."""

    def __init__(self, *, w3: Any, substrate: Any, governor_address: str,
                 chain_id: int, daemon_key: str) -> None:
        self._w3 = w3
        self._substrate = substrate
        self._chain_id = int(chain_id)
        self._daemon_key = daemon_key or ""
        self._daemon_addr = _addr_of(self._daemon_key) if self._daemon_key else ""
        self._timelock = w3.to_checksum_address(substrate.functions.governor().call())
        self._governor = None
        if governor_address:
            self._governor = w3.eth.contract(
                address=w3.to_checksum_address(governor_address), abi=GOVERNOR_ABI)

    @property
    def direct(self) -> bool:
        """Substrate's governor is this daemon's own key (dev networks)."""
        return bool(self._daemon_addr) and self._timelock.lower() == self._daemon_addr.lower()

    @property
    def available(self) -> bool:
        return self.direct or self._governor is not None

    # -- direct (dev) -----------------------------------------------------

    def anchor_direct(self, args: tuple) -> str:
        return self._send(self._substrate.functions.anchor(*args), self._daemon_key)

    # -- governed ---------------------------------------------------------

    def build(self, epoch_id: str, args: tuple) -> AnchorProposal:
        """The proposal every honest daemon derives for this anchor."""
        from web3 import Web3
        data = self._substrate.encode_abi("anchor", args=list(args))
        calldata = bytes.fromhex(data[2:] if data.startswith("0x") else data)
        description = anchor_description(epoch_id, args[1])
        dh = Web3.keccak(text=description)
        targets = [self._substrate.address]
        pid = int(self._governor.functions.hashProposal(
            targets, [0], [calldata], dh).call())
        return AnchorProposal(epoch_id=epoch_id, targets=targets, values=[0],
                              calldatas=[calldata], description=description,
                              description_hash=bytes(dh), proposal_id=pid)

    def state(self, p: AnchorProposal) -> Optional[int]:
        """Proposal state, or None while it does not exist yet."""
        try:
            return int(self._governor.functions.state(p.proposal_id).call())
        except Exception:
            return None

    def step(self, p: AnchorProposal, voters: List[Voter], *,
             may_propose: bool = True) -> str:
        """Advance the proposal by at most one action per actor. Returns the
        state name after the step ("nonexistent" before a propose lands)."""
        st = self.state(p)
        if st is None:
            if may_propose:
                self._propose(p, voters)
            st = self.state(p)
        if st == ACTIVE:
            self._vote(p, voters)
            st = self.state(p)
        if st == SUCCEEDED:
            self._try(p, "queue", self._governor.functions.queue(
                p.targets, p.values, p.calldatas, p.description_hash))
            st = self.state(p)
        if st == QUEUED:
            eta = int(self._governor.functions.proposalEta(p.proposal_id).call())
            if eta and self._now() >= eta:
                self._try(p, "execute", self._governor.functions.execute(
                    p.targets, p.values, p.calldatas, p.description_hash))
                st = self.state(p)
        p.last_state = "nonexistent" if st is None else STATE_NAMES[st]
        return p.last_state

    def _propose(self, p: AnchorProposal, voters: List[Voter]) -> None:
        threshold = int(self._governor.functions.proposalThreshold().call())
        clock = int(self._governor.functions.clock().call())
        for v in voters:
            try:
                votes = int(self._governor.functions.getVotes(
                    self._w3.to_checksum_address(v.address), max(clock - 1, 0)).call())
            except Exception:
                votes = 0
            if votes < threshold:
                continue
            if self._try(p, "propose", self._governor.functions.propose(
                    p.targets, p.values, p.calldatas, p.description), v.private_key):
                return
            if self.state(p) is not None:
                return  # another daemon's identical proposal landed first
        logger.info("epoch %s: no local agent meets the proposal threshold (%d); "
                    "waiting for another daemon to propose", p.epoch_id, threshold)

    def _vote(self, p: AnchorProposal, voters: List[Voter]) -> None:
        snap = int(self._governor.functions.proposalSnapshot(p.proposal_id).call())
        for v in voters:
            acct = self._w3.to_checksum_address(v.address)
            try:
                if self._governor.functions.hasVoted(p.proposal_id, acct).call():
                    continue
                if int(self._governor.functions.getVotes(acct, snap).call()) == 0:
                    continue
            except Exception:
                continue
            self._try(p, f"vote {acct[:10]}", self._governor.functions.castVote(
                p.proposal_id, VOTE_FOR), v.private_key)

    # -- tx plumbing ------------------------------------------------------

    def _now(self) -> int:
        return int(self._w3.eth.get_block("latest")["timestamp"])

    def _try(self, p: AnchorProposal, what: str, fn: Any,
             key: Optional[str] = None) -> bool:
        try:
            tx = self._send(fn, key or self._daemon_key)
            p.tx_hashes.append(tx)
            logger.info("epoch %s: %s tx=%s", p.epoch_id, what, tx[:18])
            return True
        except Exception as e:
            # Races are normal: every daemon proposes, queues and executes.
            logger.debug("epoch %s: %s skipped: %s", p.epoch_id, what, e)
            return False

    def _send(self, fn: Any, key: str) -> str:
        if not key:
            raise RuntimeError("no key to sign with")
        sender = _addr_of(key)
        try:
            gas = int(fn.estimate_gas({"from": sender}) * 13 // 10) + 50_000
        except Exception as e:
            raise RuntimeError(f"would revert: {e}") from e
        tx = fn.build_transaction({
            "from": sender,
            "nonce": next_nonce(self._w3, sender),
            "gas": gas,
            "gasPrice": self._w3.eth.gas_price,
            "chainId": self._chain_id,
        })
        signed = self._w3.eth.account.sign_transaction(tx, key)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        h = self._w3.eth.send_raw_transaction(raw)
        receipt = self._w3.eth.wait_for_transaction_receipt(h, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"tx reverted: {h.hex()}")
        return h.hex()


def _addr_of(key: str) -> str:
    from eth_account import Account
    return Account.from_key(key if key.startswith("0x") else "0x" + key).address
