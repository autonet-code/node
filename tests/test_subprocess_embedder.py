"""Out-of-process embedder: protocol, parity, fallback, env routing.

The worker is exercised with the hashing backend so these tests stay
fast and dependency-free; the sentence-transformers backend differs
only in which embedder class the worker constructs.
"""

import os

import pytest

from nodes.common.world_model_substrate.usefulness_coords import (
    HashingEmbedder,
    SubprocessEmbedder,
    default_usefulness_embedder,
)


@pytest.fixture()
def worker():
    emb = SubprocessEmbedder(dim=16, backend="hashing", request_timeout=30.0)
    yield emb
    emb.close()


def test_worker_coords_match_inprocess(worker):
    """Subprocess hashing output must equal in-process hashing output —
    proves the stdio protocol is transparent."""
    assert worker.ready_within(30.0)
    local = HashingEmbedder(dim=16)
    for text in ["solve the maze", "the quick brown fox", ""]:
        assert worker(text) == local(text), text


def test_worker_survives_many_requests(worker):
    assert worker.ready_within(30.0)
    local = HashingEmbedder(dim=16)
    for i in range(50):
        text = f"request number {i} with some words"
        assert worker(text) == local(text)


def test_dead_worker_respawns(worker):
    assert worker.ready_within(30.0)
    first = worker("hello world test")
    worker._proc.kill()
    worker._proc.wait()
    # Next call respawns transparently (one retry path: the call after
    # the kill detects the dead proc in _ensure_proc and restarts).
    assert worker("hello world test") == first


def test_env_hashing_routes_to_hashing(monkeypatch):
    monkeypatch.setenv("ATN_USEFULNESS_EMBEDDER", "hashing")
    emb = default_usefulness_embedder(dim=16)
    assert isinstance(emb, HashingEmbedder)


def test_env_subprocess_falls_back_while_loading(monkeypatch):
    """A worker that can't become ready within the timeout must yield
    a hashing fallback, not a hang or a crash."""
    monkeypatch.setenv("ATN_USEFULNESS_EMBEDDER", "subprocess")
    import nodes.common.world_model_substrate.usefulness_coords as uc

    class NeverReady:
        dim = 16

        def ready_within(self, timeout):
            return False

    monkeypatch.setattr(uc, "_shared_subprocess_embedder", lambda dim: NeverReady())
    monkeypatch.setattr(uc, "_ST_IMPORT_TIMEOUT_SECONDS", 0.1)
    emb = uc.default_usefulness_embedder(dim=16)
    assert isinstance(emb, HashingEmbedder)
