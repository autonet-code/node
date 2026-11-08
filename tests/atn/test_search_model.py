"""The search model's fetch (atn/search_model.py): download, hash check,
re-download on mismatch, resume, peers, backoff, progress, preinstalled
files. Small stand-in files replace the pinned ones (load=False)."""
from __future__ import annotations

import asyncio
import hashlib

import httpx
import pytest

from atn.events import EventBus, EventType
from atn.search_model import SearchModel, backoff_delay, file_url, status_for
from nodes.common.world_model_substrate.minilm import ModelFile

A = b"model bytes " * 5000          # 60 kB
B = b'{"tokenizer": true}'


def _pin(path: str, data: bytes) -> ModelFile:
    return ModelFile(path, len(data), hashlib.sha256(data).hexdigest())


FILES = (_pin("onnx/model.onnx", A), _pin("tokenizer.json", B))
BODY = {f.path: d for f, d in zip(FILES, (A, B))}


def _server(calls, *, bad_first=0, fail=False):
    """A fake Hugging Face: serves BODY by path, honours Range, can send
    corrupt bytes the first ``bad_first`` times or fail outright."""
    state = {"bad": bad_first}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/resolve/")[1].split("/", 1)[1]
        calls.append((path, request.headers.get("range")))
        if fail:
            return httpx.Response(503)
        body = BODY[path]
        if state["bad"] > 0:
            state["bad"] -= 1
            return httpx.Response(200, content=b"x" * len(body))
        rng = request.headers.get("range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            return httpx.Response(206, content=body[start:])
        return httpx.Response(200, content=body)
    return httpx.MockTransport(handler)


async def _no_sleep(_s):
    return None


def _model(tmp_path, transport, **kw):
    return SearchModel(models_dir=tmp_path / "models", transport=transport,
                       load=False, files=FILES, sleep=_no_sleep, **kw)


async def test_downloads_verifies_and_keeps_files_by_hash(tmp_path):
    calls: list = []
    events = EventBus()
    seen = []

    async def on_event(ev):
        seen.append(ev.data)
    events.subscribe(EventType.SEARCH_MODEL_STATUS, on_event)
    m = _model(tmp_path, _server(calls), events=events)
    ready = []
    m.on_ready(lambda mm: ready.append(mm))
    await m.run()
    assert m.ready and ready == [m]
    for f, data in zip(FILES, (A, B)):
        assert (tmp_path / "models" / f.sha256).read_bytes() == data
    assert sorted(p for p, _ in calls) == ["onnx/model.onnx", "tokenizer.json"]
    st = m.status()
    assert st["state"] == "ready" and st["percent"] == 100.0
    assert {f["source"] for f in st["files"]} == {"huggingface"}
    assert m.held_blob_digests() == [f.sha256 for f in FILES]
    assert m.blob_bytes(FILES[1].sha256) == B
    states = [e["state"] for e in seen]
    assert "downloading" in states and states[-1] == "ready"
    assert all("status" in e for e in seen)


async def test_later_start_only_checks_hashes(tmp_path):
    calls: list = []
    await _model(tmp_path, _server(calls)).run()
    calls.clear()
    m = _model(tmp_path, _server(calls))
    await m.run()
    assert m.ready and calls == []
    assert {f["source"] for f in m.status()["files"]} == {"disk"}


async def test_corrupt_file_on_disk_is_downloaded_again(tmp_path):
    calls: list = []
    await _model(tmp_path, _server(calls)).run()
    target = tmp_path / "models" / FILES[0].sha256
    target.write_bytes(b"y" * FILES[0].size)            # same size, wrong bytes
    calls.clear()
    m = _model(tmp_path, _server(calls))
    await m.run()
    assert m.ready and [p for p, _ in calls] == ["onnx/model.onnx"]
    assert target.read_bytes() == A


async def test_mismatched_download_is_retried_with_backoff(tmp_path):
    calls: list = []
    slept: list = []

    async def sleep(s):
        slept.append(s)
    m = SearchModel(models_dir=tmp_path / "models", transport=_server(calls, bad_first=2),
                    load=False, files=FILES, sleep=sleep, backoff_base=1.0)
    await m.run()
    assert m.ready
    assert slept == [1.0, 2.0]                        # two bad bodies, doubling
    assert (tmp_path / "models" / FILES[0].sha256).read_bytes() == A
    assert not list((tmp_path / "models").glob("*.part"))


async def test_partial_download_resumes_with_range(tmp_path):
    calls: list = []
    d = tmp_path / "models"
    d.mkdir()
    (d / (FILES[0].sha256 + ".part")).write_bytes(A[:1000])
    m = _model(tmp_path, _server(calls))
    await m.run()
    assert m.ready
    assert ("onnx/model.onnx", "bytes=1000-") in calls
    assert (d / FILES[0].sha256).read_bytes() == A


async def test_peers_serve_when_hugging_face_fails(tmp_path):
    calls: list = []
    asked = []

    async def peer(digest):
        asked.append(digest)
        return {f.sha256: data for f, data in zip(FILES, (A, B))}[digest]
    m = _model(tmp_path, _server(calls, fail=True))
    m.peer_fetcher = peer
    await m.run()
    assert m.ready and asked == [f.sha256 for f in FILES]
    assert {f["source"] for f in m.status()["files"]} == {"peer"}


async def test_a_peer_sending_wrong_bytes_is_refused(tmp_path):
    slept: list = []

    async def sleep(s):
        slept.append(s)
        if len(slept) >= 2:
            m.peer_fetcher = good

    async def bad(_digest):
        return b"z" * 10

    async def good(digest):
        return {f.sha256: data for f, data in zip(FILES, (A, B))}[digest]
    m = SearchModel(models_dir=tmp_path / "models",
                    transport=_server([], fail=True), load=False, files=FILES,
                    sleep=sleep)
    m.peer_fetcher = bad
    await m.run()
    assert m.ready and len(slept) == 2
    assert (tmp_path / "models" / FILES[0].sha256).read_bytes() == A


async def test_failure_reports_retry_state(tmp_path):
    gate = asyncio.Event()

    async def sleep(_s):
        await gate.wait()
    m = SearchModel(models_dir=tmp_path / "models", transport=_server([], fail=True),
                    load=False, files=FILES, sleep=sleep)
    task = m.start()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if m.status()["state"] == "retrying":
            break
    st = m.status()
    assert st["state"] == "retrying" and not st["ready"]
    assert "HTTP 503" in st["error"] and "peers: not connected" in st["error"]
    assert st["retry_in"] is not None and "retry" in m.status_line()
    task.cancel()


async def test_preinstalled_dir_is_used_without_downloading(tmp_path):
    pre = tmp_path / "image_models"
    pre.mkdir()
    for f, data in zip(FILES, (A, B)):
        (pre / f.sha256).write_bytes(data)
    calls: list = []
    m = _model(tmp_path, _server(calls), preinstalled_dir=pre)
    await m.run()
    assert m.ready and calls == []
    assert not (tmp_path / "models").exists()


def test_backoff_doubles_and_caps():
    assert [backoff_delay(n, 5.0, 600.0) for n in (1, 2, 3, 8, 20)] == [
        5.0, 10.0, 20.0, 600.0, 600.0]


def test_urls_name_the_pinned_revision():
    from nodes.common.world_model_substrate.minilm import MODEL_FILE
    assert file_url(MODEL_FILE) == (
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx")


def test_status_without_a_model_says_idle():
    from types import SimpleNamespace
    st = status_for(SimpleNamespace())
    assert st["ready"] is False and st["state"] == "idle"
