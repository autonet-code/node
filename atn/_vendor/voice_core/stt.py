"""Speech-to-text: push-to-talk recording + Nemotron 3.5 ASR transcription.

PushToTalkRecorder is kevin's version (the more advanced one): it supports a
duck-vs-mute key mode and a 30-second release-timeout safeguard so a missed
key-up event can't hang the recorder forever — both of which autonet's variant
had dropped.

Transcription is delegated to an out-of-process Nemotron 3.5 ASR server running
in an isolated NeMo venv (NeMo mangles shared Python envs, so it must stay
isolated; and the model takes ~90s to load, so it must stay warm).  ``transcribe``
here is a thin socket client; ``ensure_asr_server`` spawns and warms the server.
See voice_core.set_asr_server(...) to configure it.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np
import sounddevice as sd


class PushToTalkRecorder:
    """Records mic audio while a push-to-talk key is held."""

    def __init__(self, keys: list[str] | str | None = None, sr: int = 16000,
                 mixer: Any = None, on_key: str = "duck", device: int | None = None,
                 on_record_start: Callable | None = None) -> None:
        if keys is None:
            keys = ["page down"]
        self.keys = keys if isinstance(keys, list) else [keys]
        self.sr = sr
        self.mixer = mixer
        self.on_key = on_key
        self.device = device
        self.on_record_start = on_record_start
        self._chunks: list[Any] = []
        self._recording = False
        # Build scan code set for fast matching
        self._scan_codes: set[int] = set()

    def _resolve_scan_codes(self) -> None:
        """Resolve key names to scan codes (must be called after keyboard is imported)."""
        import keyboard
        for k in self.keys:
            try:
                codes = keyboard.key_to_scan_codes(k)
                self._scan_codes.update(codes)
            except ValueError:
                pass

    def _is_ptt_key(self, event: Any) -> bool:
        return event.scan_code in self._scan_codes

    def _mic_cb(self, indata: Any, _frames: int, _time: Any, _status: Any) -> None:
        if self._recording:
            self._chunks.append(indata.copy())

    def wait_and_record(self) -> Any:
        import keyboard
        if not self._scan_codes:
            self._resolve_scan_codes()
        self._chunks = []
        pressed = threading.Event()
        released = threading.Event()

        def on_press(e):
            if self._is_ptt_key(e):
                print(f"  [PTT] Key down: {e.name} scan={e.scan_code}", flush=True)
                pressed.set()

        def on_release(e):
            # Only count release after press has been detected
            if pressed.is_set() and self._is_ptt_key(e):
                print(f"  [PTT] Key up: {e.name} scan={e.scan_code}", flush=True)
                released.set()

        # Hook both before waiting — release hook ignores events until
        # press is detected, so no race window.
        press_hook = keyboard.on_press(on_press)
        release_hook = keyboard.on_release(on_release)
        pressed.wait()
        keyboard.unhook(press_hook)

        if self.on_record_start:
            self.on_record_start()

        if self.mixer:
            if self.on_key == "duck":
                self.mixer.duck()
            else:
                self.mixer.mute()

        self._recording = True
        stream = sd.InputStream(
            samplerate=self.sr, channels=1, dtype="int16",
            blocksize=1024, callback=self._mic_cb, device=self.device,
        )
        stream.start()

        if not released.wait(timeout=30):
            print("  [PTT] Release timeout (30s) — forcing stop", flush=True)
        keyboard.unhook(release_hook)

        self._recording = False
        stream.stop()
        stream.close()

        if self.mixer:
            if self.on_key == "duck":
                self.mixer.unduck()
            else:
                self.mixer.unmute()

        if not self._chunks:
            return None
        return np.concatenate(self._chunks)


# ==============================================================
#  Nemotron 3.5 ASR — out-of-process server client
# ==============================================================
#
# The server (asr-venv/nemotron_server.py) loads NeMo + the model once and
# serves length-prefixed binary frames over localhost TCP.  Configure via
# set_asr_server(); the voice service calls ensure_asr_server() at startup to
# spawn + warm it, then transcribe() per utterance.

_ASR_HOST = "127.0.0.1"
_ASR_PORT = 9123
# Path to the NeMo venv's python and the server script.  Must be set by the
# host app (kevin) via set_asr_server(); there is no sensible default.
_ASR_PYTHON: str | None = None
_ASR_SCRIPT: str | None = None
_ASR_LANG = "en-US"

_asr_proc: subprocess.Popen | None = None
_asr_lock = threading.Lock()

# ── STT backend selection ─────────────────────────────────────
# "auto"     : use Nemotron if set_asr_server() was called, else whisper.
# "nemotron" : always the out-of-process server (raises if unconfigured).
# "whisper"  : always in-process faster-whisper.
#
# Nemotron is the better model but needs a hand-built, isolated NeMo venv.
# faster-whisper is a plain pip install, so it is the right default for any
# consumer that can't assume that venv exists.
_STT_BACKEND = "auto"
_WHISPER_MODEL_SIZE = "base.en"
_WHISPER_DEVICE = "auto"
_whisper_model: Any = None
_whisper_lock = threading.Lock()


def set_stt_backend(backend: str, whisper_model: str | None = None,
                    whisper_device: str | None = None) -> None:
    """Choose the transcription backend.

    backend : "auto" (default), "nemotron", or "whisper"
    whisper_model  : faster-whisper size, e.g. "base.en", "small.en"
    whisper_device : "auto" | "cuda" | "cpu"
    """
    global _STT_BACKEND, _WHISPER_MODEL_SIZE, _WHISPER_DEVICE, _whisper_model
    if backend not in ("auto", "nemotron", "whisper"):
        raise ValueError(
            f"Unknown STT backend {backend!r}; use auto, nemotron or whisper")
    _STT_BACKEND = backend
    if whisper_model is not None and whisper_model != _WHISPER_MODEL_SIZE:
        _WHISPER_MODEL_SIZE = whisper_model
        _whisper_model = None  # force reload at next use
    if whisper_device is not None and whisper_device != _WHISPER_DEVICE:
        _WHISPER_DEVICE = whisper_device
        _whisper_model = None


def get_stt_backend() -> str:
    """The backend transcribe() would actually use right now.

    Resolves "auto" to a concrete choice, so callers can log or display it.
    """
    if _STT_BACKEND != "auto":
        return _STT_BACKEND
    return "nemotron" if (_ASR_PYTHON and _ASR_SCRIPT) else "whisper"


def _get_whisper() -> Any:
    """Load faster-whisper once, preferring CUDA but falling back to CPU."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed.  Either "
                "`pip install autonet-voice-core[whisper]` or configure the "
                "Nemotron server via voice_core.set_asr_server(...)."
            ) from exc
        if _WHISPER_DEVICE in ("auto", "cuda"):
            try:
                _whisper_model = WhisperModel(
                    _WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
                return _whisper_model
            except Exception:
                if _WHISPER_DEVICE == "cuda":
                    raise
                # auto -> fall through to CPU rather than failing outright
        _whisper_model = WhisperModel(
            _WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        return _whisper_model


def preload_stt() -> str:
    """Warm whichever backend is active; returns the backend name.

    Lets a host app pay the model-load cost at startup instead of on the first
    utterance.  Safe to call more than once.
    """
    backend = get_stt_backend()
    if backend == "nemotron":
        ensure_asr_server()
    else:
        _get_whisper()
    return backend


def _transcribe_whisper(audio_i16: Any, sr: int = 16000) -> str:
    """In-process faster-whisper transcription."""
    import tempfile
    import wave
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(np.ascontiguousarray(
                audio_i16, dtype=np.int16).tobytes())
        tmp.close()
        segments, _ = _get_whisper().transcribe(
            tmp.name, language="en", beam_size=3, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def set_asr_server(python_exe: str, script: str,
                   host: str = "127.0.0.1", port: int = 9123,
                   lang: str = "en-US") -> None:
    """Configure the out-of-process Nemotron ASR server.

    python_exe : path to asr-venv's python (the isolated NeMo environment)
    script     : path to nemotron_server.py
    """
    global _ASR_PYTHON, _ASR_SCRIPT, _ASR_HOST, _ASR_PORT, _ASR_LANG
    _ASR_PYTHON = python_exe
    _ASR_SCRIPT = script
    _ASR_HOST = host
    _ASR_PORT = port
    _ASR_LANG = lang


def _ping(timeout: float = 1.0) -> bool:
    """Return True if the ASR server answers a readiness ping."""
    try:
        with socket.create_connection((_ASR_HOST, _ASR_PORT), timeout=timeout) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.sendall(struct.pack(">I", 0))  # zero-length frame = ping
            header = _recv_exactly(s, 4, timeout)
            if header is None:
                return False
            (n,) = struct.unpack(">I", header)
            if n:
                _recv_exactly(s, n, timeout)
            return True
    except OSError:
        return False


def ensure_asr_server(warmup_timeout: float = 180.0) -> None:
    """Spawn the ASR server (if not already up) and block until it's ready.

    Idempotent: if a server already answers on the port, returns immediately.
    The model load is ~90s on a consumer GPU, hence the generous timeout.
    """
    global _asr_proc
    if _ping():
        return
    with _asr_lock:
        if _ping():
            return
        if not _ASR_PYTHON or not _ASR_SCRIPT:
            raise RuntimeError(
                "ASR server not configured.  Call voice_core.set_asr_server("
                "python_exe, script) with the asr-venv python and "
                "nemotron_server.py path."
            )
        _asr_proc = subprocess.Popen(
            [_ASR_PYTHON, _ASR_SCRIPT,
             "--host", _ASR_HOST, "--port", str(_ASR_PORT),
             "--lang", _ASR_LANG],
        )
        deadline = time.monotonic() + warmup_timeout
        while time.monotonic() < deadline:
            if _asr_proc.poll() is not None:
                raise RuntimeError(
                    f"ASR server exited during startup (code {_asr_proc.returncode})"
                )
            if _ping():
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"ASR server did not become ready within {warmup_timeout:.0f}s"
        )


def _recv_exactly(sock: socket.socket, n: int, timeout: float | None = None) -> bytes | None:
    if timeout is not None:
        sock.settimeout(timeout)
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def transcribe(audio_i16: Any, sr: int = 16000) -> str:
    """Transcribe int16 mono PCM using the configured backend.

    Dispatches to the Nemotron ASR server or in-process faster-whisper — see
    set_stt_backend().  Under the default "auto", Nemotron is used only when
    set_asr_server() has been called, so a consumer that never configures it
    transparently gets whisper instead of a hard failure.

    The signature is unchanged across backends so call sites don't change.
    Both expect 16 kHz; ``sr`` is accepted for compatibility but the recorder
    already captures at 16 kHz.
    """
    if get_stt_backend() == "whisper":
        return _transcribe_whisper(audio_i16, sr)
    audio = np.ascontiguousarray(audio_i16, dtype=np.int16)
    payload = audio.tobytes()
    with socket.create_connection((_ASR_HOST, _ASR_PORT), timeout=30) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(struct.pack(">I", len(payload)) + payload)
        header = _recv_exactly(s, 4, timeout=30)
        if header is None:
            raise RuntimeError("ASR server closed connection before responding")
        (n,) = struct.unpack(">I", header)
        text = _recv_exactly(s, n, timeout=30) if n else b""
        return (text or b"").decode("utf-8").strip()
