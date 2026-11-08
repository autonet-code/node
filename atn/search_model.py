"""The local search model: fetch, verify, load (Decision 2026-11-08).

Tool search is local to each daemon (docs/tool_substrate.md). It needs one
model, all-MiniLM-L6-v2 pinned to a revision of its official repo
(``nodes/common/world_model_substrate/minilm.py`` holds the pins). On the
first run the daemon downloads the two files in the background while
everything else starts normally; until the model is loaded, search answers
with keywords only and says so.

- **Where**: ``<data_dir>/models/<sha256>``, one file per pinned file, named
  by its content hash (the same addressing as every other blob). A
  read-only directory named by ``ATN_MODEL_DIR`` (the Docker image builds
  the model into one) is checked first.
- **Check**: size and sha256 of each file at every start. A file that fails
  is downloaded again.
- **Sources**: Hugging Face first (``HTTPS_PROXY`` and friends respected),
  then peers: the files are content-addressed blobs, so any daemon holding
  them serves them over the blob protocol (``serve_on``), and the bytes are
  verified against the pinned hash whoever sent them.
- **Failure**: retried with exponential backoff (5 s doubling, capped at 10
  minutes), forever: search keeps working on keywords meanwhile.
- **Progress**: logged, and emitted as ``search.model_status`` events (the
  terminal prints them); the WebSocket message ``search_model_status``
  returns the same dict.

``python -m atn.search_model --fetch <dir>`` downloads and verifies the
files into ``<dir>`` and exits (used by docker/Dockerfile.daemon).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from nodes.common.library import minilm
from nodes.common.library.minilm import (
    MODEL_FILE,
    MODEL_FILES,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_FILE,
    ModelFile,
)

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co"
BACKOFF_BASE = 5.0
BACKOFF_CAP = 600.0
# A progress event at most this often, and on every state change.
EVENT_INTERVAL = 1.0
_CHUNK = 1 << 20


class ModelFetchError(RuntimeError):
    """One attempt to obtain a file failed (network, HTTP status, hash)."""


def file_url(f: ModelFile, base: str = HF_BASE) -> str:
    return f"{base.rstrip('/')}/{MODEL_ID}/resolve/{MODEL_REVISION}/{f.path}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, f: ModelFile) -> bool:
    """True when ``path`` holds exactly the pinned file (size, then sha256)."""
    try:
        if not path.is_file() or path.stat().st_size != f.size:
            return False
        return _sha256_file(path) == f.sha256
    except OSError:
        return False


def backoff_delay(attempt: int, base: float = BACKOFF_BASE,
                  cap: float = BACKOFF_CAP) -> float:
    """Seconds to wait after failed attempt number ``attempt`` (1-based)."""
    return float(min(cap, base * (2 ** max(0, attempt - 1))))


class SearchModel:
    """The daemon's search model: files on disk, their check, the loaded
    embedder (installed process-wide through ``minilm.set_shared``)."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        *,
        models_dir: Optional[Path] = None,
        preinstalled_dir: Optional[Path] = None,
        events: Any = None,
        hf_base: str = HF_BASE,
        transport: Any = None,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
        backoff_base: float = BACKOFF_BASE,
        backoff_cap: float = BACKOFF_CAP,
        load: bool = True,
        files: tuple = MODEL_FILES,
    ) -> None:
        if models_dir is None:
            models_dir = Path(data_dir or Path.home() / ".atn") / "models"
        self.dir = Path(models_dir)
        if preinstalled_dir is None and os.environ.get("ATN_MODEL_DIR"):
            preinstalled_dir = Path(os.environ["ATN_MODEL_DIR"])
        self.preinstalled = Path(preinstalled_dir) if preinstalled_dir else None
        self._events = events
        self._hf_base = hf_base
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self._backoff = (float(backoff_base), float(backoff_cap))
        self._load_model = load
        # The pinned files (tests pass small stand-ins with load=False).
        self.files: tuple[ModelFile, ...] = tuple(files)
        # Async callable digest -> bytes | None: a blob from peers. Wired by
        # the autonet bridge once the libp2p host is up.
        self.peer_fetcher: Optional[Callable[[str], Awaitable[Optional[bytes]]]] = None
        self._ready_callbacks: list[Callable[["SearchModel"], Any]] = []
        self._task: Optional[asyncio.Task] = None
        self._paths: dict[str, Path] = {}          # sha256 -> verified file
        self._files: dict[str, dict[str, Any]] = {
            f.sha256: {"path": f.path, "sha256": f.sha256, "size": f.size,
                       "bytes_done": 0, "state": "pending", "source": ""}
            for f in self.files
        }
        self._state = "idle"
        self._source = ""
        self._attempt = 0
        self._retry_at: Optional[float] = None
        self._error = ""
        self._last_event = 0.0
        self._last_logged_decile = -1

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._state == "ready"

    def status(self) -> dict[str, Any]:
        total = sum(f.size for f in self.files)
        done = sum(int(self._files[f.sha256]["bytes_done"]) for f in self.files)
        retry_in = None
        if self._retry_at is not None:
            retry_in = round(max(0.0, self._retry_at - time.monotonic()), 1)
        return {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "dim": minilm.MODEL_DIM,
            "state": self._state,
            "ready": self.ready,
            "bytes_done": done,
            "bytes_total": total,
            "percent": round(100.0 * done / total, 1) if total else 0.0,
            "source": self._source,
            "files": [dict(self._files[f.sha256]) for f in self.files],
            "attempt": self._attempt,
            "retry_in": retry_in,
            "error": self._error,
        }

    def status_line(self) -> str:
        s = self.status()
        mb = 1024 * 1024
        if s["state"] == "ready":
            return f"ready ({MODEL_ID.split('/')[-1]})"
        if s["state"] == "downloading":
            src = f" from {s['source']}" if s["source"] else ""
            return (f"downloading {s['percent']:.0f}% "
                    f"({s['bytes_done'] // mb}/{s['bytes_total'] // mb} MB){src}")
        if s["state"] == "retrying":
            return f"retry {s['attempt'] + 1} in {s['retry_in']:.0f}s: {s['error']}"
        return s["state"]

    async def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_event < EVENT_INTERVAL:
            return
        self._last_event = now
        if self._events is None:
            return
        try:
            from .events import Event, EventType
            await self._events.emit(Event(
                type=EventType.SEARCH_MODEL_STATUS, source="search_model",
                data={**self.status(), "status": self.status_line()}))
        except Exception:  # noqa: BLE001 (progress is best effort)
            log.debug("search model event failed", exc_info=True)

    async def _set_state(self, state: str) -> None:
        self._state = state
        await self._emit(force=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_ready(self, callback: Callable[["SearchModel"], Any]) -> None:
        """Call ``callback(self)`` once the model is loaded (at once if it
        already is). Exceptions are logged, never raised."""
        if self.ready:
            self._call(callback)
        else:
            self._ready_callbacks.append(callback)

    def _call(self, callback: Callable[["SearchModel"], Any]) -> None:
        try:
            out = callback(self)
            if asyncio.iscoroutine(out):
                asyncio.ensure_future(out)
        except Exception:  # noqa: BLE001
            log.warning("search model ready callback failed", exc_info=True)

    def start(self) -> asyncio.Task:
        """Run ``run()`` in the background (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.run())
        return self._task

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def run(self) -> None:
        """Until the model is loaded: check, fetch what is missing, load.
        A failed attempt waits ``backoff_delay(attempt)`` and tries again."""
        while not self.ready:
            self._attempt += 1
            try:
                await self.ensure_once()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                delay = backoff_delay(self._attempt, *self._backoff)
                self._error = str(exc) or exc.__class__.__name__
                self._retry_at = time.monotonic() + delay
                await self._set_state("retrying")
                log.warning("search model: attempt %d failed (%s); retrying "
                            "in %.0fs. Search is keyword-only until then.",
                            self._attempt, self._error, delay)
                await self._sleep(delay)
                self._retry_at = None

    async def ensure_once(self) -> None:
        """One attempt: every file verified (or fetched and verified), then
        the model loaded. Raises on the first file that cannot be had."""
        await self._set_state("checking")
        for f in self.files:
            if f.sha256 in self._paths:
                continue
            found = await asyncio.to_thread(self._find_verified, f)
            if found is not None:
                self._mark_verified(f, found, "disk")
                continue
            await self._fetch(f)
        if self._load_model:
            await self._set_state("loading")
            embedder = await asyncio.to_thread(
                minilm.load, self._paths[MODEL_FILE.sha256],
                self._paths[TOKENIZER_FILE.sha256])
            minilm.set_shared(embedder)
        self._error = ""
        await self._set_state("ready")
        if self._load_model:
            log.info("search model ready: %s@%s (semantic search on)",
                     MODEL_ID, MODEL_REVISION[:12])
        callbacks, self._ready_callbacks = self._ready_callbacks, []
        for cb in callbacks:
            self._call(cb)

    def _candidates(self, f: ModelFile) -> list[Path]:
        out = []
        if self.preinstalled is not None:
            out.append(self.preinstalled / f.sha256)
        out.append(self.dir / f.sha256)
        return out

    def _find_verified(self, f: ModelFile) -> Optional[Path]:
        for path in self._candidates(f):
            if path.exists():
                if verify_file(path, f):
                    return path
                log.warning("search model: %s at %s fails its hash check",
                            f.path, path)
                if path.parent == self.dir:
                    try:
                        path.unlink()
                    except OSError:
                        pass
        return None

    def _mark_verified(self, f: ModelFile, path: Path, source: str) -> None:
        self._paths[f.sha256] = path
        entry = self._files[f.sha256]
        entry.update({"bytes_done": f.size, "state": "verified",
                      "source": source})

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def _fetch(self, f: ModelFile) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        errors = []
        try:
            await self._fetch_hf(f)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            errors.append(f"huggingface: {exc}")
            log.info("search model: %s from Hugging Face failed: %s",
                     f.path, exc)
        if self.peer_fetcher is not None:
            try:
                await self._fetch_peer(f)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(f"peers: {exc}")
        else:
            errors.append("peers: not connected")
        self._files[f.sha256].update({"state": "pending"})
        raise ModelFetchError(f"{f.path}: " + "; ".join(errors))

    async def _progress(self, f: ModelFile, done: int) -> None:
        self._files[f.sha256]["bytes_done"] = done
        s = self.status()
        decile = int(s["percent"] // 10)
        if decile != self._last_logged_decile:
            self._last_logged_decile = decile
            log.info("search model: %s", self.status_line())
        await self._emit()

    async def _fetch_hf(self, f: ModelFile) -> None:
        import httpx

        part = self.dir / (f.sha256 + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if offset >= f.size:
            part.unlink(missing_ok=True)
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        self._source = "huggingface"
        entry = self._files[f.sha256]
        entry.update({"state": "downloading", "source": "huggingface",
                      "bytes_done": offset})
        await self._set_state("downloading")
        timeout = httpx.Timeout(30.0, read=120.0)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout,
                                     transport=self._transport,
                                     trust_env=True) as client:
            async with client.stream("GET", file_url(f, self._hf_base),
                                     headers=headers) as resp:
                if resp.status_code == 206 and offset:
                    mode = "ab"
                elif resp.status_code == 200:
                    mode, offset = "wb", 0
                else:
                    raise ModelFetchError(f"HTTP {resp.status_code}")
                done = offset
                with part.open(mode) as fh:
                    async for chunk in resp.aiter_bytes(_CHUNK):
                        fh.write(chunk)
                        done += len(chunk)
                        if done > f.size:
                            break
                        await self._progress(f, done)
        await self._install(f, part, "huggingface")

    async def _fetch_peer(self, f: ModelFile) -> None:
        self._source = "peer"
        self._files[f.sha256].update({"state": "downloading", "source": "peer",
                                      "bytes_done": 0})
        await self._set_state("downloading")
        data = await self.peer_fetcher(f.sha256)  # type: ignore[misc]
        if not data:
            raise ModelFetchError("no peer holds it")
        part = self.dir / (f.sha256 + ".part")
        await asyncio.to_thread(part.write_bytes, bytes(data))
        await self._install(f, part, "peer")

    async def _install(self, f: ModelFile, part: Path, source: str) -> None:
        ok = await asyncio.to_thread(verify_file, part, f)
        if not ok:
            part.unlink(missing_ok=True)
            self._files[f.sha256]["bytes_done"] = 0
            raise ModelFetchError(f"{source} sent a file that fails the "
                                  "pinned sha256")
        final = self.dir / f.sha256
        os.replace(part, final)
        self._mark_verified(f, final, source)
        await self._emit(force=True)

    # ------------------------------------------------------------------
    # Serving to peers
    # ------------------------------------------------------------------

    def held_blob_digests(self) -> list[str]:
        """The pinned files this daemon holds verified, by sha256: what it
        serves to peers and (with a DHT, the p2p-mesh branch) provides."""
        return [f.sha256 for f in self.files if f.sha256 in self._paths]

    def blob_bytes(self, digest: str) -> Optional[bytes]:
        """A blob-server source: the bytes of a verified model file."""
        path = self._paths.get(str(digest or "").lower())
        if path is None:
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def serve_on(self, host: Any) -> None:
        """Serve the verified files to peers from ``host`` (the libp2p
        host), and announce them. On this branch peers find a holder by
        asking their connections; ``provide_blobs_threadsafe`` (the DHT
        provider records of the p2p-mesh branch) is called when the host
        has it, so the files are announced like tool blobs once that
        branch is in. Called again when the model becomes ready."""
        try:
            from nodes.common.blob_resolver import host_blob_server
            host_blob_server(host).add_source(self.blob_bytes)  # deduplicated
        except Exception:  # noqa: BLE001
            log.debug("serving model blobs failed", exc_info=True)
            return
        digests = self.held_blob_digests()
        provide = getattr(host, "provide_blobs_threadsafe", None)
        if digests and callable(provide):
            try:
                provide(digests)
            except Exception:  # noqa: BLE001
                log.debug("providing model blobs failed", exc_info=True)


# ---------------------------------------------------------------------------
# The runtime's instance
# ---------------------------------------------------------------------------

def model_for(runtime: Any) -> Optional[SearchModel]:
    return getattr(runtime, "search_model", None)


def status_for(runtime: Any) -> dict[str, Any]:
    """``search_model_status`` for the WS API and search replies."""
    model = model_for(runtime)
    if model is not None:
        return model.status()
    ready = minilm.shared() is not None
    return {"model": MODEL_ID, "revision": MODEL_REVISION,
            "dim": minilm.MODEL_DIM, "state": "ready" if ready else "idle",
            "ready": ready, "bytes_done": 0,
            "bytes_total": sum(f.size for f in MODEL_FILES), "percent": 0.0,
            "source": "", "files": [], "attempt": 0, "retry_in": None,
            "error": ""}


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "--fetch":
        print("usage: python -m atn.search_model --fetch <dir>", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = SearchModel(models_dir=Path(argv[1]), load=False)
    try:
        asyncio.run(model.ensure_once())
    except Exception as exc:  # noqa: BLE001
        print(f"search model fetch failed: {exc}", file=sys.stderr)
        return 1
    print(f"search model files verified in {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
