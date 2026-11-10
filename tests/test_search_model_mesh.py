"""The search model's files travel like any blob on the mesh
(atn/search_model.py, docs/p2p_networking.md "Blobs"): a daemon holding
them provides their sha256 in the DHT, and a daemon that cannot reach
Hugging Face finds a holder with find_providers and fetches from it,
directly or through a relay circuit. The bytes are checked against the pin
whoever sent them.

Real localhost hosts run in their own trio thread, as in the daemon; the
model's fetch runs on asyncio in the test's thread. Small stand-in files
replace the pinned ones (load=False).

Run: python -m pytest tests/test_search_model_mesh.py -v
"""

import asyncio
import hashlib
import sys
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import trio

from atn.search_model import SearchModel, peer_fetcher_for
from nodes.common.library.minilm import ModelFile
from tests.test_p2p_mesh import _host, _tcp_addr, _until

DATA = b"stand-in model weights " * 20000          # 460 kB
FILE = ModelFile("onnx/model.onnx", len(DATA), hashlib.sha256(DATA).hexdigest())


async def _no_sleep(_s):
    return None


def _hugging_face_down():
    return httpx.MockTransport(lambda _r: httpx.Response(503))


class _Mesh:
    """A relay and a NAT'd holder (reserved on the relay) in a trio thread;
    ``join`` adds a NAT'd fetcher that dials the relay only. No mesh loops:
    they would dial holder and fetcher to each other, and the fetch would
    find the holder as a neighbour instead of through the DHT."""

    def __init__(self):
        self.hosts = {}
        self._error = []
        self._ready = threading.Event()
        self._joined = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        async def main():
            async with AsyncExitStack() as stack:
                relay = _host("relay", relay_hop=True)
                holder = _host("holder", advertise_listen_addrs=False, mesh_interval=0)
                await stack.enter_async_context(relay.run())
                await stack.enter_async_context(holder.run())
                assert await holder.connect_to_peer(_tcp_addr(relay))
                assert await _until(lambda: relay.peer_id in holder.reservations, 20)
                assert await _until(
                    lambda: holder._dht is not None and relay.peer_id in
                    holder._dht.routing_table.get_peer_ids(), 20)
                self.stop, self._join_req = trio.Event(), trio.Event()
                self.token = trio.lowlevel.current_trio_token()
                self.hosts.update(relay=relay, holder=holder)
                self._ready.set()
                await self._join_req.wait()
                # Entered by this task: a nursery must close in the task
                # that opened it.
                fetcher = _host("fetcher", advertise_listen_addrs=False, mesh_interval=0)
                await stack.enter_async_context(fetcher.run())
                assert await fetcher.connect_to_peer(_tcp_addr(relay))
                assert await _until(lambda: fetcher._dht is not None and relay.peer_id in
                                    fetcher._dht.routing_table.get_peer_ids(), 20)
                self.hosts["fetcher"] = fetcher
                self._joined.set()
                await self.stop.wait()
        try:
            trio.run(main)
        except BaseException as exc:  # noqa: BLE001
            self._error.append(exc)
            self._ready.set()
            self._joined.set()

    def join(self):
        """Start the fetcher: NAT'd, dialling the relay only."""
        trio.from_thread.run_sync(self._join_req.set, trio_token=self.token)
        assert self._joined.wait(60), "the fetcher did not join"
        assert not self._error, self._error
        return self.hosts["fetcher"]

    def __enter__(self):
        self._thread.start()
        assert self._ready.wait(90), "mesh did not come up"
        assert not self._error, self._error
        return self

    def __exit__(self, *exc):
        if self._thread.is_alive():
            trio.from_thread.run_sync(self.stop.set, trio_token=self.token)
        self._thread.join(30)


def test_model_files_are_provided_in_the_dht_and_fetched_from_a_provider(tmp_path):
    held = tmp_path / "holder"
    held.mkdir()
    (held / FILE.sha256).write_bytes(DATA)
    holder_model = SearchModel(models_dir=held, load=False, files=(FILE,))
    asyncio.run(holder_model.ensure_once())
    assert holder_model.held_blob_digests() == [FILE.sha256]

    with _Mesh() as mesh:
        holder = mesh.hosts["holder"]
        # The holder serves the file and provides its hash: the record
        # lands on the relay, which does not hold the file.
        holder_model.serve_on(holder)
        deadline = time.time() + 45
        while FILE.sha256 not in holder._provided_at and time.time() < deadline:
            time.sleep(0.25)
        assert FILE.sha256 in holder._provided_at, "no provider record placed"

        # A new daemon joins through the relay; Hugging Face is down.
        fetcher = mesh.join()
        assert holder.peer_id not in fetcher.get_connected_peers()
        model = SearchModel(models_dir=tmp_path / "fetcher", load=False,
                            files=(FILE,), transport=_hugging_face_down(),
                            sleep=_no_sleep)
        model.peer_fetcher = peer_fetcher_for(fetcher)
        asyncio.run(model.ensure_once())

        assert model.ready
        assert (tmp_path / "fetcher" / FILE.sha256).read_bytes() == DATA
        assert model.status()["files"][0]["source"] == "peer"
        # Found through the provider record, not by asking a neighbour.
        assert fetcher.last_fetch_source == "dht"
        assert holder.peer_id in fetcher.connection_kinds()


def test_a_peer_that_sends_other_bytes_is_refused(tmp_path):
    """Whoever answers, the bytes must hash to the pin."""
    async def liar(_digest):
        return b"x" * FILE.size

    model = SearchModel(models_dir=tmp_path / "m", load=False, files=(FILE,),
                        transport=_hugging_face_down(), sleep=_no_sleep)
    model.peer_fetcher = liar
    try:
        asyncio.run(model.ensure_once())
    except Exception as exc:  # noqa: BLE001
        assert "pinned sha256" in str(exc)
    else:
        raise AssertionError("a wrong file was installed")
    assert not (tmp_path / "m" / FILE.sha256).exists()


def test_peer_fetcher_without_a_running_host_finds_nothing():
    class Idle:
        _trio_token = None

    assert asyncio.run(peer_fetcher_for(Idle())(FILE.sha256)) is None


def test_hf_endpoint_names_a_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    assert SearchModel(models_dir=tmp_path, load=False)._hf_base == "https://mirror.example"
    monkeypatch.delenv("HF_ENDPOINT")
    assert SearchModel(models_dir=tmp_path, load=False)._hf_base == "https://huggingface.co"
