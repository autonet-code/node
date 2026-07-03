"""§12 InputArbiter liveness — ghost-holder recovery + dead-token hand-off skip.

The bug (2026-07-04 finding 3): the mic could be held by a dead WS surface
forever; ghosts stayed in _connected and recaptured the mic on every hand-off,
wedging every input surface with ``input_not_active``.

These tests cover the arbiter half of the fix:
  - hand-off (release_for) skips + prunes dead ws: surfaces.
  - a non-ws surface (voice/chat) is never judged dead by the predicate.
  - a missing predicate = legacy "everything is live".
"""
import asyncio

import pytest

from atn.events import EventBus
from atn.input_arbiter import InputArbiter, SurfaceId


def _ws(conn_id: str, label: str = "") -> SurfaceId:
    return SurfaceId(kind="ws", instance=conn_id, label=label or conn_id)


def _voice() -> SurfaceId:
    return SurfaceId(kind="voice", instance="local", label="Voice", in_process=True)


def make_arbiter(live_tokens: set[str] | None = None) -> InputArbiter:
    """Arbiter whose liveness predicate treats ws tokens in ``live_tokens`` as
    live; a None set means no predicate (legacy all-live)."""
    events = EventBus()
    if live_tokens is None:
        return InputArbiter(events)
    return InputArbiter(events, liveness=lambda tok: tok in live_tokens)


# ---------------------------------------------------------------------------
# Hand-off skips dead ws surfaces
# ---------------------------------------------------------------------------

def test_handoff_skips_dead_ws_successor():
    """When the holder leaves, a dead ws surface must NOT inherit the mic; the
    arbiter walks past it to the most-recent LIVE surface."""
    live = {"ws:live"}
    arb = make_arbiter(live)

    dead = _ws("dead")
    alive = _ws("live")
    holder = _ws("holder")

    # Register alive first, dead second (dead is newer → would win legacy).
    arb.register(alive)
    arb.register(dead)
    arb.register(holder)

    # holder takes the mic, then disconnects.
    assert arb.request_input is not None
    arb._holder = holder  # simulate holder having the mic
    arb._connected[holder.token] = holder

    arb.release_for(holder)

    # Dead surface must be skipped; live one inherits.
    assert arb.holder_token() == "ws:live"
    # Dead surface pruned from the connected set.
    assert "ws:dead" not in arb.state()["connected"] and \
        all(c["token"] != "ws:dead" for c in arb.state()["connected"])


def test_handoff_frees_mic_when_only_dead_remain():
    """If every remaining surface is a dead ws, the mic goes free (None)."""
    arb = make_arbiter(set())  # nothing is live

    dead1 = _ws("d1")
    dead2 = _ws("d2")
    arb.register(dead1)
    arb.register(dead2)
    arb._holder = dead1
    arb._connected[dead1.token] = dead1

    arb.release_for(dead1)

    assert arb.holder_token() is None


def test_handoff_prefers_newest_live_surface():
    """Live-surface selection is still most-recently-registered first."""
    live = {"ws:a", "ws:b"}
    arb = make_arbiter(live)
    a = _ws("a")
    b = _ws("b")
    holder = _ws("h")
    live.add("ws:h")
    arb.register(a)
    arb.register(b)   # newest live
    arb.register(holder)
    arb._holder = holder
    arb._connected[holder.token] = holder

    arb.release_for(holder)
    assert arb.holder_token() == "ws:b"


# ---------------------------------------------------------------------------
# Non-ws surfaces are never judged by the predicate
# ---------------------------------------------------------------------------

def test_voice_surface_never_dead():
    """A voice/chat surface is always live even when no ws token is live."""
    arb = make_arbiter(set())  # predicate says every ws token is dead
    voice = _voice()
    dead_ws = _ws("dead")
    arb.register(voice)     # oldest
    arb.register(dead_ws)   # newest, but dead
    holder = _ws("holder")
    arb.register(holder)
    arb._holder = holder
    arb._connected[holder.token] = holder

    arb.release_for(holder)
    # dead_ws skipped; voice (non-ws, always live) inherits.
    assert arb.holder_token() == "voice:local"


def test_token_alive_only_judges_ws():
    arb = make_arbiter({"ws:x"})
    assert arb._token_alive("ws:x") is True
    assert arb._token_alive("ws:y") is False       # ws, not in live set
    assert arb._token_alive("voice:local") is True  # non-ws → always live
    assert arb._token_alive("chat:discord") is True


def test_no_predicate_is_legacy_all_live():
    """Without a predicate, every token is live — exact legacy hand-off."""
    arb = make_arbiter(None)  # no predicate
    old = _ws("old")
    new = _ws("new")
    holder = _ws("holder")
    arb.register(old)
    arb.register(new)
    arb.register(holder)
    arb._holder = holder
    arb._connected[holder.token] = holder

    arb.release_for(holder)
    # newest wins, nothing pruned.
    assert arb.holder_token() == "ws:new"


def test_set_liveness_predicate_after_construction():
    """The backward-compat setter installs the predicate on an arbiter built
    without one."""
    arb = make_arbiter(None)
    assert arb._token_alive("ws:gone") is True   # legacy: live
    arb.set_liveness_predicate(lambda tok: tok in {"ws:here"})
    assert arb._token_alive("ws:gone") is False
    assert arb._token_alive("ws:here") is True


def test_liveness_predicate_error_fails_open():
    """A raising predicate must never wedge input — treat the token as live."""
    def boom(_tok):
        raise RuntimeError("predicate blew up")

    arb = InputArbiter(EventBus(), liveness=boom)
    assert arb._token_alive("ws:whatever") is True


# ---------------------------------------------------------------------------
# is_active auto-acquire still works with a predicate present
# ---------------------------------------------------------------------------

def test_is_active_auto_acquire_free_mic():
    arb = make_arbiter({"ws:s"})
    sid = _ws("s")
    # Free mic → first genuine inbound auto-acquires.
    assert arb.is_active(sid) is True
    assert arb.holder_token() == "ws:s"


def test_ghost_recovery_end_to_end():
    """Full ghost scenario: a dead surface holds the mic, then a live surface
    arrives and, after the dead one is released, takes the mic — mirroring the
    _arbiter_gate recovery path (release dead holder, retry)."""
    live = {"ws:new"}
    arb = make_arbiter(live)

    ghost = _ws("ghost")     # session already gone, not in live set
    arb.register(ghost)
    arb._holder = ghost      # ghost holds the mic
    arb._connected[ghost.token] = ghost

    newcomer = _ws("new")
    arb.register(newcomer)

    # Newcomer is denied while ghost holds the mic.
    assert arb.is_active(newcomer) is False
    # Recovery: holder is a dead ws token → release it.
    assert arb.holder_token() == "ws:ghost"
    assert arb._token_alive("ws:ghost") is False
    arb.release_for("ws:ghost")
    # Now the live newcomer can acquire.
    assert arb.is_active(newcomer) is True
    assert arb.holder_token() == "ws:new"
