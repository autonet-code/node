"""epoch_history and network_status's epochs_closed report anchored epochs
(Substrate anchors), never the WorldService's local 60 s projection ticks."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from atn.ws_server import WebSocketBridge


def _server(**rpb):
    cfg = SimpleNamespace(network_joined=True, substrate_address="0x" + "11" * 20,
                          rpc_url="http://rpc.invalid", **rpb)
    local = SimpleNamespace(epoch_history=[{"epoch_id": "tick"}] * 9,
                            epochs_recorded=9, epoch_status=lambda: {"running": True})
    autonet = SimpleNamespace(get_status=lambda: {}, _p2p_host=None,
                              _service=SimpleNamespace(_world_service=local))
    srv = WebSocketBridge.__new__(WebSocketBridge)
    srv.runtime = SimpleNamespace(_config=SimpleNamespace(rpb=cfg), autonet=autonet)
    return srv


def test_anchors_are_reported(monkeypatch):
    import nodes.common.voice_state as vs
    monkeypatch.setattr(vs, "read_recent_anchors", lambda s, r, n: {
        "anchor_count": 3,
        "anchors": [{"number": 3, "epoch_id": "e3", "anchored_at": 100}][-n:] if n else []})
    srv = _server()
    got = asyncio.run(srv._anchored_epochs(5))
    assert got["total_closed"] == 3 and got["epochs"][-1]["epoch_id"] == "e3"
    assert got["source"] == "anchored" and "reason" not in got
    status = srv._network_status_payload(asyncio.run(srv._anchored_epochs(0)))
    assert status["epochs_closed"] == 3


def test_unreadable_anchors_give_an_empty_list_with_a_reason(monkeypatch):
    import nodes.common.voice_state as vs

    def boom(*a):
        raise ConnectionError("rpc down")
    monkeypatch.setattr(vs, "read_recent_anchors", boom)
    srv = _server()
    got = asyncio.run(srv._anchored_epochs(5))
    assert got["epochs"] == [] and got["total_closed"] == 0
    assert "rpc down" in got["reason"]
    status = srv._network_status_payload(got)
    assert status["epochs_closed"] == 0 and "rpc down" in status["epochs_closed_reason"]
