"""Decision 2026-10-26, item 8: the consensus findings that ship with the
epoch redesign.

  - a failed voice refresh blocks the close instead of reusing old state;
  - log scans start at the configured deployment block, and a failed read
    while searching for it raises instead of starting the scan late;
  - the kernels use platform-independent exp/log.

(Address case and the payload encoding are covered in test_tool_mint.py
and test_carry_over_commitment.py.)
"""

from __future__ import annotations

import math
import random
from unittest.mock import MagicMock

import pytest

from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
from nodes.common.federated_close_driver import FederatedCloseDriver
from world_model.generalized import detmath


def _event(seq: int, agent: str):
    return {
        "kind": "observation_added",
        "seq": seq,
        "author_agent": agent,
        "obs_id": f"obs_{agent}_{seq}",
        "coords": [0.5, 0.1 * seq, 0.0, 0.0, 0.0, 0.0],
        "label": f"{agent}_{seq}",
    }


def _batch(kp: Keypair, seq: int, agent: str) -> EventBatch:
    return EventBatch(
        rpb_address="rpb_test",
        sender_pubkey=kp.public_key,
        batch_seq=seq,
        events=[_event(seq, agent)],
        prev_batch_hash=b"",
        timestamp=float(seq),
    )


def _gossip(kp: Keypair, epochs):
    g = MagicMock(spec=EventGossip)
    g.drain_epoch_batches.side_effect = [list(b) for b in epochs]
    g.known_senders.return_value = [kp.public_key]
    g.sender_pubkey = kp.public_key
    return g


def _driver(gossip, state_dir):
    return FederatedCloseDriver(gossip=gossip)


class TestVoiceRefresh:
    def test_failed_refresh_blocks_and_keeps_nothing_partial(self, tmp_path):
        kp = Keypair.generate()
        d = _driver(_gossip(kp, [[_batch(kp, 1, "alice")]]), tmp_path)
        d.agent_owner_map = {"old": "owner"}

        class Flaky(dict):
            # owner_map reads fine, then the pool read blows up.
            def get(self, key, default=None):
                if key == "emission_pool":
                    raise ConnectionError("rpc down")
                return super().get(key, default)

        d.voice_source = lambda: Flaky(
            owner_map={"new": "owner"}, snapshot_block=10)
        d.enforce_preconditions = True
        assert d.run({"epoch_id": "e1"}) is None
        assert d.agent_owner_map == {"old": "owner"}
        assert any("voice-state refresh failed" in b
                   for b in d.participation_blockers())

        d.voice_source = lambda: {"owner_map": {"new": "owner"},
                                  "snapshot_block": None}
        d._refresh_voice()
        assert d.agent_owner_map == {"new": "owner"}
        assert not any("voice-state" in b for b in d.participation_blockers())


class TestDeployBlock:
    def test_configured_start_skips_the_search(self, monkeypatch):
        from nodes.common import voice_state

        def boom(*a, **k):
            raise AssertionError("searched despite a configured block")

        monkeypatch.setattr(voice_state, "_deployment_block", boom)
        event = MagicMock()
        event.get_logs.return_value = ["log"]
        assert voice_state._get_logs(event, None, "0x1", 50, 7) == ["log"]
        event.get_logs.assert_called_once_with(from_block=7, to_block=50)

    def test_failed_read_raises_instead_of_starting_late(self):
        from nodes.common import voice_state

        w3 = MagicMock()
        w3.provider = object()
        w3.eth.block_number = 1000

        def get_code(address, block_identifier):
            if block_identifier == 1000:
                return b"\x60"
            raise ValueError("missing trie node")

        w3.eth.get_code.side_effect = get_code
        with pytest.raises(ValueError):
            voice_state._deployment_block(w3, "0xdead")


class TestDetMath:
    def test_close_to_libm(self):
        rng = random.Random(7)
        for _ in range(20000):
            x = rng.uniform(-50, 50)
            assert detmath.exp(x) == pytest.approx(math.exp(x), rel=5e-16)
            y = rng.uniform(1e-9, 1e9)
            assert detmath.log(y) == pytest.approx(math.log(y), rel=5e-16,
                                                   abs=1e-300)
        for n in range(1000):
            assert detmath.log1p(n) == pytest.approx(math.log1p(n), rel=5e-16)

    def test_bits_are_fixed(self):
        # Built from IEEE basic operations only, so these bits are the same
        # on every platform; a change here is a consensus change.
        assert detmath.exp(0.5).hex() == "0x1.a61298e1e069cp+0"
        assert detmath.exp(-3.75).hex() == "0x1.81509354f0d29p-6"
        assert detmath.log(10.0).hex() == "0x1.26bb1bbb55516p+1"
        assert detmath.log1p(7).hex() == "0x1.0a2b23f3bab73p+1"

    def test_edges(self):
        assert detmath.exp(-800.0) == 0.0
        with pytest.raises(OverflowError):
            detmath.exp(710.0)
        with pytest.raises(ValueError):
            detmath.log(0.0)
        assert detmath.log1p(1e-20) == 1e-20
