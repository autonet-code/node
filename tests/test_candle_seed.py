"""Chain-derived candle cutoff seed.

The candle cut must be (a) identical across daemons and (b) unknowable
while the epoch is open. ChainCandleSeed gets (a) from public chain
state and (b) from the hash of the first block past the window's end.
These tests pin the block search, seed determinism, the not-yet-mined
case, and the scheduler integration (shared source → shared cutoff;
source failure → local fallback, never a crash).
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nodes.common.candle_seed import ChainCandleSeed, first_block_at_or_after
from nodes.common.epoch_scheduler import EpochScheduler, EpochSchedulerConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeBlock:
    number: int
    timestamp: int
    hash: bytes


class FakeEth:
    def __init__(self, blocks: List[FakeBlock]):
        self._blocks = blocks

    def get_block(self, ident):
        if ident == "latest":
            return self._blocks[-1]
        return self._blocks[int(ident)]


class FakeW3:
    def __init__(self, blocks: List[FakeBlock]):
        self.eth = FakeEth(blocks)


class FakeAnchorFn:
    def __init__(self, anchor_by_block: Dict[int, bytes], head: bytes):
        self._by_block = anchor_by_block
        self._head = head

    def call(self, block_identifier: Optional[int] = None):
        if block_identifier is None:
            return self._head
        return self._by_block.get(int(block_identifier), self._head)


class FakeContract:
    def __init__(self, anchor_by_block: Dict[int, bytes], head: bytes):
        self.functions = type("F", (), {})()
        self.functions.latestAnchorHash = lambda: FakeAnchorFn(
            anchor_by_block, head)


def _chain(n_blocks: int, t0: int = 1000, dt: int = 10) -> List[FakeBlock]:
    return [
        FakeBlock(number=i, timestamp=t0 + i * dt,
                  hash=hashlib.sha256(f"block_{i}".encode()).digest())
        for i in range(n_blocks)
    ]


# ---------------------------------------------------------------------------
# Block search
# ---------------------------------------------------------------------------


def test_first_block_at_or_after_finds_boundary():
    w3 = FakeW3(_chain(20))                  # timestamps 1000..1190
    b = first_block_at_or_after(w3, 1055)    # first ts >= 1055 is 1060
    assert b.number == 6 and b.timestamp == 1060
    # Exact hit
    assert first_block_at_or_after(w3, 1060).number == 6
    # Before genesis -> genesis
    assert first_block_at_or_after(w3, 0).number == 0


def test_first_block_none_when_chain_behind():
    w3 = FakeW3(_chain(5))                   # latest ts = 1040
    assert first_block_at_or_after(w3, 2000) is None


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def test_seed_deterministic_across_instances():
    blocks = _chain(20)
    anchor = b"\x11" * 32
    s1 = ChainCandleSeed(FakeContract({}, anchor), FakeW3(blocks))
    s2 = ChainCandleSeed(FakeContract({}, anchor), FakeW3(blocks))
    a, b = s1.seed_for(1075), s2.seed_for(1075)
    assert a == b and len(a) == 32


def test_seed_none_while_window_unmined():
    s = ChainCandleSeed(FakeContract({}, b"\x11" * 32), FakeW3(_chain(3)))
    assert s.seed_for(99_999) is None


def test_seed_reads_anchor_as_of_boundary_block():
    """Daemons sampling late (after a NEW anchor landed) must still
    derive the seed from the anchor state at the boundary block."""
    blocks = _chain(20)
    old_anchor, new_anchor = b"\x22" * 32, b"\x33" * 32
    # Boundary block for ts=1075 is block 8; historical read returns
    # the old anchor there even though head moved on.
    contract = FakeContract({8: old_anchor}, head=new_anchor)
    seed = ChainCandleSeed(contract, FakeW3(blocks)).seed_for(1075)
    expected = hashlib.sha256(old_anchor + blocks[8].hash).digest()
    assert seed == expected


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------


class _StubWorldService:
    current_epoch_id = "e_test"
    epoch_history: list = []

    def open_epoch(self, *a, **k):
        return {}

    def close_epoch(self, *a, **k):
        return {"epoch_id": "e_test"}


def _scheduler(seed_source):
    sched = EpochScheduler(
        world_service=_StubWorldService(),
        config=EpochSchedulerConfig(
            candle_min_seconds=100.0,
            candle_window_seconds=50.0,
            open_first_epoch_on_start=False,
        ),
        candle_seed_source=seed_source,
    )
    sched._last_open_at = time.time() - 200.0   # window fully elapsed
    return sched


def test_two_schedulers_same_source_same_cut():
    seed = hashlib.sha256(b"shared").digest()
    calls = []

    def source(window_end):
        calls.append(window_end)
        return seed

    s1, s2 = _scheduler(source), _scheduler(source)
    s1._last_open_at = s2._last_open_at   # same open time -> same window
    cut1, cut2 = s1._draw_candle_cut(), s2._draw_candle_cut()
    assert cut1 == cut2
    # Cut lands inside [min, min+window] of the open time.
    offset = cut1 - s1._last_open_at
    assert 100.0 <= offset <= 150.0
    # Source was asked about the window END, not the open time.
    assert calls[0] == s1._last_open_at + 150.0


def test_scheduler_falls_back_when_source_returns_none():
    s = _scheduler(lambda window_end: None)
    cut = s._draw_candle_cut()
    offset = cut - s._last_open_at
    assert 100.0 <= offset <= 150.0


def test_scheduler_falls_back_when_source_raises():
    def source(window_end):
        raise RuntimeError("rpc down")

    s = _scheduler(source)
    cut = s._draw_candle_cut()
    offset = cut - s._last_open_at
    assert 100.0 <= offset <= 150.0
