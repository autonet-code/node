"""Readiness handshake of the subprocess embedder against a slow worker.

Regression for the 2026-09-10 Services-page stall: a ``ready_within`` wait
that timed out during a slow model load left a reader thread blocked on the
worker's stdout; when READY finally arrived that orphan swallowed it, and
every later wait blocked on a line that never came. The embedder then
reported "not ready" for the rest of the daemon's life.

A scripted stand-in worker sleeps before its READY line so the first wait
can be made to time out deterministically.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nodes.common.world_model_substrate.usefulness_coords import (
    SubprocessEmbedder,
    usefulness_embedder_if_ready,
)


SLOW_WORKER = textwrap.dedent(
    """
    import json, sys, time
    dim = int(sys.argv[1])
    time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
    sys.stdout.write("not json: a stray line the protocol must skip\\n")
    sys.stdout.write(json.dumps({"ready": True, "backend": "scripted"}) + "\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        text = json.loads(line)["text"]
        vec = [float(len(text) % 7)] + [0.0] * (dim - 1)
        sys.stdout.write(json.dumps({"coords": vec}) + "\\n")
        sys.stdout.flush()
    """
)


@pytest.fixture()
def slow_worker(tmp_path: Path):
    script = tmp_path / "slow_worker.py"
    script.write_text(SLOW_WORKER, encoding="utf-8")
    # model_name doubles as the sleep argument for the scripted worker.
    emb = SubprocessEmbedder(dim=8, model_name="1.5", backend="scripted",
                             request_timeout=20.0)
    emb.worker_path = script
    yield emb
    emb.close()


def test_ready_line_survives_a_timed_out_wait(slow_worker):
    # First wait gives up before the worker is up (the slow-boot case).
    assert slow_worker.ready_within(0.2) is False
    # The READY line arrives later and must reach the NEXT waiter.
    assert slow_worker.ready_within(10.0) is True
    # And the protocol still works end to end afterwards.
    vec = slow_worker("hello")
    assert len(vec) == 8


def test_stray_output_before_ready_is_skipped(slow_worker):
    assert slow_worker.ready_within(10.0) is True


def test_if_ready_returns_none_while_loading(monkeypatch, tmp_path: Path):
    import nodes.common.world_model_substrate.usefulness_coords as uc

    script = tmp_path / "slow_worker.py"
    script.write_text(SLOW_WORKER, encoding="utf-8")
    emb = SubprocessEmbedder(dim=8, model_name="1.5", backend="scripted")
    emb.worker_path = script
    monkeypatch.setattr(uc, "_shared_subprocess_embedder", lambda dim: emb)
    monkeypatch.delenv("ATN_USEFULNESS_EMBEDDER", raising=False)
    try:
        # Interactive callers get "not yet" instead of a 30s block.
        assert usefulness_embedder_if_ready(dim=8, wait=0.1) is None
        # ...and the real embedder once it is up.
        assert usefulness_embedder_if_ready(dim=8, wait=10.0) is emb
    finally:
        emb.close()
