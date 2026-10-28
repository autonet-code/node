"""Epoch anchors as single-choice elections (Decision 2026-10-26, items 5-6;
contracts/core/AnchorElection.sol).

Each epoch's anchor is one DAO proposal whose only action is
``AnchorElection.anchorWinner(epochNumber, attempt)``, with a description
the contract derives, so every daemon computes the same proposal id. Its
life, as one daemon drives it with its local agents' keys:

    nonexistent -> propose (any daemon; identical proposals collide)
    Pending     -> each drawn local agent registers this daemon's option
                   (the 8 anchor fields of its own close), once per option
    Active      -> every local agent with REP votes for the option whose
                   hash equals this daemon's own; no match means no vote
    Succeeded   -> queue
    Queued      -> execute once the timelock delay passed (reverts while
                   the previous epoch is unanchored; retried next tick)
    Defeated / Expired / Canceled -> the next attempt (a new proposal id)

The draw itself is on chain (``isDrawn``), so a daemon never guesses who
may register: it asks for each of its agents.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .tx_nonce import next_nonce
from .governed_anchor import (
    ACTIVE, CANCELED, DEFEATED, EXPIRED, GOVERNOR_ABI, PENDING,
    QUEUED, STATE_NAMES, SUCCEEDED, Voter, _addr_of,
)

logger = logging.getLogger(__name__)

_RETRY = {DEFEATED, EXPIRED, CANCELED}
# A queued election whose winner still hasn't executed this long after its
# eta can't anchor (the winning option is one Substrate rejects): re-post
# as a new attempt instead of waiting out the driver's give-up.
EXECUTE_GRACE_S = 600


def epoch_number_of(epoch_id: str) -> Optional[int]:
    """The number an ``epoch-<n>-<root>-<cutoff>`` id carries."""
    parts = str(epoch_id).split("-")
    if len(parts) == 4 and parts[0] == "epoch":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


@dataclass
class Election:
    """One epoch's election as this daemon follows it."""
    epoch_id: str
    epoch_number: int
    option: Tuple[Any, ...]          # the 8 Substrate.anchor arguments
    option_hash: bytes = b""
    attempt: int = 0
    created: float = field(default_factory=time.monotonic)
    last_state: str = "nonexistent"
    tx_hashes: List[str] = field(default_factory=list)


class ElectionAnchorer:
    """Drives epoch elections through the DAO Governor and AnchorElection."""

    def __init__(self, *, w3: Any, substrate: Any, election_address: str,
                 governor_address: str, chain_id: int, daemon_key: str,
                 election_abi: Optional[list] = None) -> None:
        from .contract_artifacts import load_abi

        self._w3 = w3
        self._substrate = substrate
        self._chain_id = int(chain_id)
        self._daemon_key = daemon_key or ""
        self._election = w3.eth.contract(
            address=w3.to_checksum_address(election_address),
            abi=election_abi or load_abi("AnchorElection"))
        self._governor = w3.eth.contract(
            address=w3.to_checksum_address(governor_address), abi=GOVERNOR_ABI)

    # -- identity ---------------------------------------------------------

    def open(self, epoch_id: str, option: Tuple[Any, ...]) -> Election:
        number = epoch_number_of(epoch_id)
        if number is None:
            number = int(self._substrate.functions.anchorCount().call()) + 1
        h = bytes(self._election.functions.optionHashOf(tuple(option)).call())
        return Election(epoch_id=epoch_id, epoch_number=number,
                        option=tuple(option), option_hash=h)

    def proposal_id(self, e: Election) -> int:
        return int(self._election.functions.proposalIdOf(
            e.epoch_number, e.attempt).call())

    def state(self, e: Election) -> Optional[int]:
        try:
            return int(self._governor.functions.state(self.proposal_id(e)).call())
        except Exception:
            return None   # not proposed yet

    # -- the life cycle ---------------------------------------------------

    def step(self, e: Election, voters: List[Voter], *,
             may_propose: bool = True) -> str:
        """Advance the election by at most one action per actor. Returns
        the state name after the step ("nonexistent" before a propose
        lands, "retry" when the attempt was bumped)."""
        st = self.state(e)
        if st is None and may_propose:
            self._propose(e, voters)
            st = self.state(e)
        if st == PENDING:
            self._register(e, voters)
        elif st == ACTIVE:
            self._vote(e, voters)
            st = self.state(e)
        if st == SUCCEEDED:
            self._try(e, "queue", self._governor.functions.queue(
                *self._proposal_args(e)[:3], self._description_hash(e)))
            st = self.state(e)
        if st == QUEUED:
            pid = self.proposal_id(e)
            eta = int(self._governor.functions.proposalEta(pid).call())
            now = self._now()
            if eta and now >= eta:
                self._try(e, "execute", self._governor.functions.execute(
                    *self._proposal_args(e)[:3], self._description_hash(e)))
                st = self.state(e)
                if st == QUEUED and now >= eta + EXECUTE_GRACE_S:
                    logger.warning("epoch %s: attempt %d won but its option "
                                   "has not executed %ds past its eta",
                                   e.epoch_id, e.attempt, now - eta)
                    st = CANCELED   # treated like a failed attempt below
        if st in _RETRY:
            logger.warning("epoch %s: election attempt %d %s; re-posting as "
                           "attempt %d", e.epoch_id, e.attempt,
                           STATE_NAMES[st], e.attempt + 1)
            e.attempt += 1
            e.last_state = "retry"
            return e.last_state
        e.last_state = "nonexistent" if st is None else STATE_NAMES[st]
        return e.last_state

    def _proposal_args(self, e: Election) -> tuple:
        call = bytes(self._election.functions.callOf(
            e.epoch_number, e.attempt).call())
        desc = self._election.functions.descriptionOf(
            e.epoch_number, e.attempt).call()
        return [self._election.address], [0], [call], desc

    def _description_hash(self, e: Election) -> bytes:
        return bytes(self._w3.keccak(text=self._proposal_args(e)[3]))

    def _propose(self, e: Election, voters: List[Voter]) -> None:
        targets, values, calldatas, desc = self._proposal_args(e)
        threshold = int(self._governor.functions.proposalThreshold().call())
        clock = int(self._governor.functions.clock().call())
        keys = [v.private_key for v in voters] + (
            [self._daemon_key] if self._daemon_key else [])
        for key in keys:
            try:
                votes = int(self._governor.functions.getVotes(
                    self._w3.to_checksum_address(_addr_of(key)),
                    max(clock - 1, 0)).call())
            except Exception:
                votes = 0
            if votes < threshold:
                continue
            if self._try(e, "propose", self._governor.functions.propose(
                    targets, values, calldatas, desc), key):
                return
            if self.state(e) is not None:
                return   # another daemon's identical proposal landed first
        logger.info("epoch %s: no local key meets the proposal threshold (%d); "
                    "waiting for another daemon to propose", e.epoch_id, threshold)

    def _option_index(self, e: Election) -> Optional[int]:
        pid = self.proposal_id(e)
        n = int(self._election.functions.optionCount(pid).call())
        for i in range(n):
            if bytes(self._election.functions.optionHash(pid, i).call()) == e.option_hash:
                return i
        return None

    def _register(self, e: Election, voters: List[Voter]) -> None:
        if self._option_index(e) is not None:
            return   # our option is in (registered by us or an agreeing peer)
        pid = self.proposal_id(e)
        tp = int(self._election.functions.drawTimepoint(pid).call())
        if tp >= self._now():
            return   # the draw's REP timepoint is not in the past yet
        for v in voters:
            acct = self._w3.to_checksum_address(v.address)
            try:
                drawn = bool(self._election.functions.isDrawn(
                    e.epoch_number, e.attempt, acct, tp).call())
            except Exception:
                drawn = False
            if drawn and self._try(e, f"register {acct[:10]}",
                                   self._election.functions.registerOption(
                                       e.epoch_number, e.attempt, e.option),
                                   v.private_key):
                return

    def _vote(self, e: Election, voters: List[Voter]) -> None:
        index = self._option_index(e)
        if index is None:
            logger.info("epoch %s: no registered option matches this daemon's "
                        "close; not voting", e.epoch_id)
            return
        pid = self.proposal_id(e)
        snap = int(self._governor.functions.proposalSnapshot(pid).call())
        for v in voters:
            acct = self._w3.to_checksum_address(v.address)
            try:
                if self._governor.functions.hasVoted(pid, acct).call():
                    continue
                if int(self._governor.functions.getVotes(acct, snap).call()) == 0:
                    continue
            except Exception:
                continue
            self._try(e, f"vote {acct[:10]} for option {index}",
                      self._governor.functions.castVote(pid, index),
                      v.private_key)

    # -- tx plumbing ------------------------------------------------------

    def _now(self) -> int:
        return int(self._w3.eth.get_block("latest")["timestamp"])

    def _try(self, e: Election, what: str, fn: Any,
             key: Optional[str] = None) -> bool:
        try:
            tx = self._send(fn, key or self._daemon_key)
            e.tx_hashes.append(tx)
            logger.info("epoch %s: %s tx=%s", e.epoch_id, what, tx[:18])
            return True
        except Exception as ex:
            # Races are normal: every daemon proposes, queues and executes.
            logger.debug("epoch %s: %s skipped: %s", e.epoch_id, what, ex)
            return False

    def _send(self, fn: Any, key: str) -> str:
        if not key:
            raise RuntimeError("no key to sign with")
        sender = _addr_of(key)
        try:
            gas = int(fn.estimate_gas({"from": sender}) * 13 // 10) + 50_000
        except Exception as ex:
            raise RuntimeError(f"would revert: {ex}") from ex
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
