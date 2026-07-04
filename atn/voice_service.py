"""Voice service — gives ATN a voice.

The voice Surface (see atn/surface.py). Like the chat Surface, it is a
long-lived bidirectional bridge from an external real-time channel (here,
audio) to agent sessions: it owns a persistent connection (the audio device),
reacts to inbound events (push-to-talk), routes them to agents through an INPUT
SEAM gated by an InputPolicy, and streams agent output back out via the
EventBus. VoiceService and ChatService are the same shape, different channel.

Optional module — install with ``pip install atn[voice]``.

Subscribes to the EventBus and speaks agent output aloud, plays tool
tones, handles push-to-talk STT, and routes voice input to the
orchestrator or a specific delegate.

Adapted from the kevin voice service for ATN's event-driven architecture
(no UDP, no hooks, no window management).

Dependency tiers (each is an optional extras group in pyproject.toml):

  atn[voice]        — numpy + sounddevice (core audio mixing & playback)
  atn[voice-kokoro] — kokoro-onnx (local TTS)
  atn[voice-edge]   — edge-tts + miniaudio (free cloud TTS)
  atn[voice-11labs]  — requests (paid cloud TTS via ElevenLabs API)
  atn[voice-ptt]    — keyboard + faster-whisper (push-to-talk STT)
  atn[voice-full]   — all of the above

The module can always be *imported* safely — the feature-availability
check happens at start() time via ``VOICE_AVAILABLE``.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time
import wave
from typing import Any, Callable, TYPE_CHECKING

from dataclasses import dataclass

from .config import VoiceConfig
from .events import Event, EventBus, EventType
from .input_arbiter import SurfaceId
from .surface import AllowAll, InputPolicy

if TYPE_CHECKING:
    import numpy as np
    import sounddevice as sd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature gate — single check for the core voice extras
# ---------------------------------------------------------------------------
# If atn[voice] is installed, numpy and sounddevice are available.
# Everything else (TTS backends, PTT) is checked per-feature.

try:
    import numpy as _np      # type: ignore[assignment]
    import sounddevice as _sd  # type: ignore[assignment]
    VOICE_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _sd = None  # type: ignore[assignment]
    VOICE_AVAILABLE = False

# PTT feature gate — keyboard + faster-whisper.
# NOTE: faster_whisper transitively imports torch (~590MB, ~5s). We must NOT
# pay that in every process that imports voice_service (e.g. an agent worker,
# the daemon boot). So the gate is a lightweight spec-check only; the heavy
# imports happen lazily at first use (_get_whisper / the PTT loop).
import importlib.util as _ilu
PTT_AVAILABLE = (
    _ilu.find_spec("keyboard") is not None
    and _ilu.find_spec("faster_whisper") is not None
)
_keyboard = None  # lazily bound in the PTT loop


# ── Constants ─────────────────────────────────────────────────
MIXER_SR = 44100
SENTENCE_END = re.compile(r'(?<=[.!?])\s')

# Tool name -> sound category
TOOL_SOUND_MAP = {
    "Bash": "execute", "Read": "read", "Edit": "edit", "Write": "write",
    "Grep": "search", "Glob": "search", "Task": "agent",
    "WebFetch": "web", "WebSearch": "web", "TodoWrite": "todo",
    "NotebookEdit": "edit",
    # Orchestrator tools
    "create_agent": "agent",
    "trigger_run": "execute", "get_execution": "read",
    "use_connector": "execute",
}

TOOL_FREQS = {
    "execute": [440, 554],
    "read":    [659],
    "edit":    [440, 659],
    "write":   [554, 440],
    "search":  [880, 659],
    "agent":   [440, 554, 659],
    "web":     [330, 440],
    "todo":    [554, 659],
    "default": [440],
}


# ── Markdown / path cleaning for TTS ─────────────────────────

def strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS reads natural prose."""
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'^[\s\-:]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Strip emojis so TTS doesn't read them as words
    text = re.sub(
        r'[\U0001F600-\U0001F64F'   # emoticons
        r'\U0001F300-\U0001F5FF'     # misc symbols & pictographs
        r'\U0001F680-\U0001F6FF'     # transport & map symbols
        r'\U0001F1E0-\U0001F1FF'     # flags
        r'\U0001F900-\U0001F9FF'     # supplemental symbols
        r'\U0001FA00-\U0001FA6F'     # chess symbols, extended-A
        r'\U0001FA70-\U0001FAFF'     # symbols extended-A continued
        r'\U00002702-\U000027B0'     # dingbats
        r'\U0000FE00-\U0000FE0F'     # variation selectors
        r'\U0000200D'                # zero-width joiner
        r'\U000023F0-\U000023FA'     # misc technical
        r'\U00002600-\U000026FF'     # misc symbols
        r'\U0000203C\U00002049'      # exclamation marks
        r'\U000020E3'                # combining enclosing keycap
        r'\U00002934\U00002935'      # arrows
        r'\U000025AA-\U000025FE'     # geometric shapes
        r']+', '', text)
    # Collapse Windows and Unix paths to basename
    text = re.sub(r'[A-Z]:\\(?:[^\\\s]+\\)*([^\\\s]+)', r'\1', text)
    text = re.sub(r'(?<!\w)/(?:[^/\s]+/)+([^/\s]+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Device lookup ─────────────────────────────────────────────

def find_device(name: str, kind: str | None = None) -> int | None:
    """Find an audio device by name substring."""
    for i, d in enumerate(_sd.query_devices()):
        if name.lower() in d['name'].lower():
            if kind == 'input' and d['max_input_channels'] == 0:
                continue
            if kind == 'output' and d['max_output_channels'] == 0:
                continue
            return i
    return None


def get_device_list() -> tuple[list[dict], list[dict]]:
    """Return (outputs, inputs) device lists for the UI."""
    all_devs = list(enumerate(_sd.query_devices()))
    default_out = _sd.default.device[1]
    default_in = _sd.default.device[0]

    outputs = [
        {"id": i, "name": d['name'], "default": i == default_out}
        for i, d in all_devs if d['max_output_channels'] > 0
    ]
    inputs = [
        {"id": i, "name": d['name'], "default": i == default_in}
        for i, d in all_devs if d['max_input_channels'] > 0
    ]
    return outputs, inputs


# ── Resampling ────────────────────────────────────────────────

def _resample(audio: Any, src_rate: int, dst_rate: int) -> Any:
    if src_rate == dst_rate:
        return audio
    n_out = int(len(audio) * dst_rate / src_rate)
    return _np.interp(
        _np.linspace(0, len(audio) - 1, n_out),
        _np.arange(len(audio)),
        audio,
    ).astype(_np.float32)


# ==============================================================
#  Audio channel & mixer
# ==============================================================

class AudioChannel:
    """A single audio channel with volume and fade support."""

    def __init__(self, volume: float = 1.0) -> None:
        self.volume = volume
        self._chunks: list = []
        self._chunk_offset = 0
        self._total = 0
        self._lock = threading.Lock()
        self._fade_vol = 1.0
        self._fade_rate = 0.0
        # When paused, read() returns silence WITHOUT consuming the buffer, so
        # playback resumes exactly where it left off. Set by AudioMixer.pause().
        self.paused = False

    def write(self, audio_f32: Any) -> None:
        with self._lock:
            self._chunks.append(audio_f32)
            self._total += len(audio_f32)

    def read(self, n: int) -> Any:
        # Paused channels emit silence and keep their buffer intact. Do this
        # BEFORE touching the buffer or the fade envelope so a pause holds the
        # fade state frozen too (resume picks up mid-fade unchanged).
        if self.paused:
            return _np.zeros(n, dtype=_np.float32)
        with self._lock:
            if self._total == 0:
                return None
            out_parts: list = []
            remaining = n
            while remaining > 0 and self._chunks:
                chunk = self._chunks[0]
                avail = len(chunk) - self._chunk_offset
                take = min(avail, remaining)
                out_parts.append(chunk[self._chunk_offset:self._chunk_offset + take])
                self._chunk_offset += take
                remaining -= take
                self._total -= take
                if self._chunk_offset >= len(chunk):
                    self._chunks.pop(0)
                    self._chunk_offset = 0
            if not out_parts:
                return None
            out = _np.concatenate(out_parts) if len(out_parts) > 1 else out_parts[0].copy()
            if len(out) < n:
                out = _np.pad(out, (0, n - len(out)))

        if self._fade_rate != 0.0:
            start = self._fade_vol
            end = max(0.0, min(1.0, start + self._fade_rate * n))
            envelope = _np.linspace(start, end, n, dtype=_np.float32)
            out *= envelope
            self._fade_vol = end
            if self._fade_vol <= 0.0:
                self.clear()
                self._fade_rate = 0.0
        elif self._fade_vol < 1.0:
            out *= self._fade_vol

        return out

    def fade_out(self, duration_secs: float, sr: int = MIXER_SR) -> None:
        total = max(1, int(sr * duration_secs))
        self._fade_rate = -self._fade_vol / total

    def fade_reset(self) -> None:
        self._fade_rate = 0.0
        self._fade_vol = 1.0

    def has_data(self) -> bool:
        with self._lock:
            return self._total > 0

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._chunk_offset = 0
            self._total = 0


class AudioMixer:
    """Multi-channel audio mixer with real-time output."""

    def __init__(self, sr: int = MIXER_SR, blocksize: int = 2048,
                 device: int | None = None) -> None:
        self.sr = sr
        self._blocksize = blocksize
        self._device = device
        self.channels: dict[str, AudioChannel] = {}
        self.master_volume = 1.0
        self._duck_level = 0.25
        self._ducking = False
        self._muting = False
        self._stream: Any = None  # sd.OutputStream

    def start(self) -> None:
        """Start the audio output stream."""
        if self._stream is not None:
            return
        self._stream = _sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            blocksize=self._blocksize, callback=self._callback,
            device=self._device,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop the audio output stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_device(self, device: int | None) -> None:
        was_running = self._stream is not None
        if was_running:
            self.stop()
        self._device = device
        if was_running:
            self.start()

    def _callback(self, outdata: Any, frames: int,
                  _time_info: Any, _status: Any) -> None:
        mixed = _np.zeros(frames, dtype=_np.float32)
        for ch in self.channels.values():
            chunk = ch.read(frames)
            if chunk is not None:
                mixed += chunk * ch.volume
        if self._muting:
            vol = 0.0
        elif self._ducking:
            vol = self._duck_level
        else:
            vol = self.master_volume
        outdata[:, 0] = mixed * vol

    def add_channel(self, name: str, volume: float = 1.0) -> None:
        self.channels[name] = AudioChannel(volume=volume)

    def play(self, channel: str, audio_f32: Any,
             sr: int | None = None) -> None:
        if sr and sr != self.sr:
            audio_f32 = _resample(audio_f32, sr, self.sr)
        self.channels[channel].write(audio_f32)

    def duck(self) -> None:
        self._ducking = True
        self._muting = False

    def unduck(self) -> None:
        self._ducking = False

    def mute(self) -> None:
        self._muting = True
        self._ducking = False

    def unmute(self) -> None:
        self._muting = False

    def pause(self, name: str) -> None:
        """Freeze one channel: it emits silence but keeps its buffer + fade
        state, so unpause() resumes exactly where playback stopped. Distinct
        from mute (master-level) and stop_all (drains buffers)."""
        ch = self.channels.get(name)
        if ch is not None:
            ch.paused = True

    def unpause(self, name: str) -> None:
        ch = self.channels.get(name)
        if ch is not None:
            ch.paused = False

    def stop_all(self) -> None:
        for ch in self.channels.values():
            ch.clear()

    def is_playing(self, name: str | None = None) -> bool:
        if name:
            return self.channels[name].has_data()
        return any(ch.has_data() for ch in self.channels.values())

    def wait_channel(self, name: str) -> None:
        while self.channels[name].has_data():
            time.sleep(0.03)


# ==============================================================
#  Tone generation
# ==============================================================

def _gen_tone(freq: float, dur: float = 0.09, vol: float = 0.25,
              sr: int = MIXER_SR) -> Any:
    n = int(sr * dur)
    t = _np.linspace(0, dur, n, dtype=_np.float32)
    tone = vol * _np.sin(2 * _np.pi * freq * t)
    att = min(int(0.005 * sr), n)
    rel = min(int(0.015 * sr), n)
    tone[:att] *= _np.linspace(0, 1, att, dtype=_np.float32)
    tone[-rel:] *= _np.linspace(1, 0, rel, dtype=_np.float32)
    return tone


def make_tool_tone(tool_name: str) -> Any:
    cat = TOOL_SOUND_MAP.get(tool_name, "default")
    freqs = TOOL_FREQS.get(cat, TOOL_FREQS["default"])
    gap = _np.zeros(int(MIXER_SR * 0.015), dtype=_np.float32)
    parts: list = []
    for f in freqs:
        parts.append(_gen_tone(f))
        parts.append(gap)
    return _np.concatenate(parts)


def make_result_chime() -> Any:
    c5 = _gen_tone(523, dur=0.12, vol=0.18)
    g5 = _gen_tone(784, dur=0.15, vol=0.15)
    gap = _np.zeros(int(MIXER_SR * 0.04), dtype=_np.float32)
    return _np.concatenate([c5, gap, g5])


def make_delegate_spawn_tone() -> Any:
    """Ascending three-note tone for delegate spawn."""
    e4 = _gen_tone(330, dur=0.08, vol=0.2)
    a4 = _gen_tone(440, dur=0.08, vol=0.2)
    e5 = _gen_tone(659, dur=0.10, vol=0.18)
    gap = _np.zeros(int(MIXER_SR * 0.02), dtype=_np.float32)
    return _np.concatenate([e4, gap, a4, gap, e5])


def make_startup_chime() -> Any:
    """C-E-G startup chime."""
    c5 = _gen_tone(523, dur=0.08, vol=0.3)
    e5 = _gen_tone(659, dur=0.08, vol=0.3)
    g5 = _gen_tone(784, dur=0.12, vol=0.33)
    gap = _np.zeros(int(MIXER_SR * 0.015), dtype=_np.float32)
    return _np.concatenate([c5, gap, e5, gap, g5])


def _make_failure_tone() -> Any:
    """Descending two-note tone for failure."""
    return _np.concatenate([
        _gen_tone(440, dur=0.15, vol=0.3),
        _np.zeros(int(MIXER_SR * 0.03), dtype=_np.float32),
        _gen_tone(330, dur=0.2, vol=0.3),
    ])


# ==============================================================
#  TTS backends — all lazy-imported, all optional
# ==============================================================

_kokoro_model = None
_kokoro_lock = threading.Lock()

# Configurable search paths for Kokoro model.  Set via VoiceConfig.kokoro_model_dir
# or falls back to looking next to this file.
_kokoro_model_dir: str | None = None


def _get_kokoro():
    global _kokoro_model
    if _kokoro_model is None:
        with _kokoro_lock:
            if _kokoro_model is None:
                try:
                    import kokoro_onnx
                except ImportError:
                    raise ImportError(
                        "Kokoro backend requires kokoro-onnx.  "
                        "Install with: pip install atn[voice-kokoro]"
                    )
                # Build search paths — configurable dir first, then alongside this file
                search_paths = []
                if _kokoro_model_dir:
                    search_paths.append(
                        os.path.join(_kokoro_model_dir, "kokoro-v1.0.onnx")
                    )
                search_paths.append(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "kokoro-v1.0.onnx")
                )
                onnx_path = None
                voices_path = None
                for p in search_paths:
                    if os.path.exists(p):
                        onnx_path = p
                        v = os.path.join(os.path.dirname(p), "voices-v1.0.bin")
                        if os.path.exists(v):
                            voices_path = v
                        break
                if onnx_path is None:
                    raise FileNotFoundError(
                        "kokoro-v1.0.onnx not found.  Set voice.kokoro_model_dir "
                        "in config.yaml or place the model alongside voice_service.py"
                    )
                _kokoro_model = kokoro_onnx.Kokoro(onnx_path, voices_path)
    return _kokoro_model


def generate_kokoro(text: str, voice: str = "af_heart") -> tuple[Any, int]:
    kokoro = _get_kokoro()
    with _kokoro_lock:
        samples, sr = kokoro.create(text, voice=voice, speed=1.0)
    return samples, sr


# Edge TTS (free, cloud)
_edge_loop = None
_edge_lock = threading.Lock()


def _get_edge_loop():
    global _edge_loop
    if _edge_loop is None:
        with _edge_lock:
            if _edge_loop is None:
                _edge_loop = asyncio.new_event_loop()
                t = threading.Thread(target=_edge_loop.run_forever, daemon=True)
                t.start()
    return _edge_loop


def generate_edge(text: str, voice: str = "en-US-JennyNeural") -> tuple[Any, int]:
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "Edge backend requires edge-tts.  "
            "Install with: pip install atn[voice-edge]"
        )
    try:
        import miniaudio
    except ImportError:
        raise ImportError(
            "Edge backend requires miniaudio.  "
            "Install with: pip install atn[voice-edge]"
        )

    async def _synth():
        communicate = edge_tts.Communicate(text, voice)
        mp3_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]
        return mp3_bytes

    loop = _get_edge_loop()
    future = asyncio.run_coroutine_threadsafe(_synth(), loop)
    mp3_bytes = future.result(timeout=30)
    if not mp3_bytes:
        raise RuntimeError("Edge TTS returned no audio")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        tmp.write(mp3_bytes)
        tmp.close()
        decoded = miniaudio.decode_file(
            tmp.name, output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1, sample_rate=24000,
        )
        audio = _np.frombuffer(
            decoded.samples, dtype=_np.int16
        ).astype(_np.float32) / 32768.0
        return audio, 24000
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ElevenLabs (paid, cloud)
def generate_elevenlabs(text: str, voice_id: str | None = None) -> tuple[Any, int]:
    try:
        import requests
    except ImportError:
        raise ImportError(
            "ElevenLabs backend requires requests.  "
            "Install with: pip install atn[voice-11labs]"
        )
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    voice_id = voice_id or os.environ.get(
        "ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"
    )
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_v3")
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        f"?output_format=pcm_24000"
    )
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id}
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    pcm = _np.frombuffer(
        resp.content, dtype=_np.int16
    ).astype(_np.float32) / 32768.0
    return pcm, 24000


# Piper (offline, fallback) — requires the voice module
_piper_module_dir: str | None = None  # configurable via VoiceConfig


def generate_piper(text: str, voice: str = "male") -> tuple[Any, int]:
    voice_dir = _piper_module_dir
    if not voice_dir:
        raise RuntimeError(
            "Piper backend requires voice.piper_module_dir in config.yaml "
            "pointing to the piper voice module directory"
        )
    if voice_dir not in sys.path:
        sys.path.insert(0, voice_dir)
    try:
        from voice import _load_voices, _voice_lock, is_local_tts_available
    except ImportError:
        raise ImportError(
            f"Could not import voice module from {voice_dir}.  "
            f"Ensure the directory contains voice.py with Piper TTS support."
        )
    if not is_local_tts_available():
        raise RuntimeError("Piper TTS not available (piper package not installed)")
    with _voice_lock:
        _load_voices()
        from voice import _male_voice, _female_voice
        v = _male_voice if voice == "male" else _female_voice
        if v is None:
            raise RuntimeError(f"Piper {voice} voice not loaded")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(v.config.sample_rate)
            v.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    audio = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
    return audio, sr


# Backend availability check — used for graceful fallback
def _available_backends() -> list[str]:
    """Return list of TTS backends that can be used."""
    avail = []
    # Edge — only needs edge-tts (lightweight)
    try:
        import edge_tts  # noqa: F401
        avail.append("edge")
    except ImportError:
        pass
    # Kokoro — needs kokoro_onnx + model files
    try:
        import kokoro_onnx  # noqa: F401
        avail.append("kokoro")
    except ImportError:
        pass
    # ElevenLabs — needs requests + API key
    if os.environ.get("ELEVENLABS_API_KEY"):
        try:
            import requests  # noqa: F401
            avail.append("elevenlabs")
        except ImportError:
            pass
    # Piper — needs voice module
    if _piper_module_dir:
        avail.append("piper")
    return avail


# ==============================================================
#  STT — push-to-talk (fully optional)
# ==============================================================

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                # Lazy import — this is where torch actually loads.
                from faster_whisper import WhisperModel as _WhisperModel
                try:
                    _whisper_model = _WhisperModel(
                        "base.en", device="cuda", compute_type="float16"
                    )
                except Exception:
                    _whisper_model = _WhisperModel(
                        "base.en", device="cpu", compute_type="int8"
                    )
    return _whisper_model


def transcribe(audio_i16: Any, sr: int = 16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_i16.tobytes())
        tmp.close()
        model = _get_whisper()
        segments, _ = model.transcribe(
            tmp.name, language="en", beam_size=3, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


class PushToTalkRecorder:
    """Records audio while a push-to-talk key is held.

    Requires `keyboard` and `sounddevice`.  If keyboard is not available
    the VoiceService starts without PTT (TTS-only mode).
    """

    def __init__(self, keys: list[str] | None = None, sr: int = 16000,
                 mixer: AudioMixer | None = None,
                 device: int | None = None,
                 on_record_start: Callable | None = None) -> None:
        # Lazy-bind the keyboard module here (only when PTT is actually
        # constructed) so importing voice_service never pulls it in.
        global _keyboard
        if _keyboard is None:
            import keyboard as _keyboard  # type: ignore[no-redef]
        self.keys = keys or ["page down"]
        self.sr = sr
        self.mixer = mixer
        self.device = device
        self.on_record_start = on_record_start
        self._chunks: list = []
        self._recording = False
        self._scan_codes: set[int] = set()

    def _resolve_scan_codes(self) -> None:
        for k in self.keys:
            try:
                codes = _keyboard.key_to_scan_codes(k)
                self._scan_codes.update(codes)
            except ValueError:
                pass

    def _mic_cb(self, indata: Any, _frames: int,
                _time: Any, _status: Any) -> None:
        if self._recording:
            self._chunks.append(indata.copy())

    def wait_and_record(self) -> Any:
        if not self._scan_codes:
            self._resolve_scan_codes()
        self._chunks = []
        pressed = threading.Event()
        released = threading.Event()

        def on_press(e):
            if e.scan_code in self._scan_codes:
                pressed.set()

        def on_release(e):
            if pressed.is_set() and e.scan_code in self._scan_codes:
                released.set()

        press_hook = _keyboard.on_press(on_press)
        release_hook = _keyboard.on_release(on_release)
        pressed.wait()
        _keyboard.unhook(press_hook)

        if self.on_record_start:
            self.on_record_start()

        if self.mixer:
            self.mixer.mute()

        self._recording = True
        stream = _sd.InputStream(
            samplerate=self.sr, channels=1, dtype="int16",
            blocksize=1024, callback=self._mic_cb, device=self.device,
        )
        stream.start()
        released.wait()
        _keyboard.unhook(release_hook)
        self._recording = False
        stream.stop()
        stream.close()

        if self.mixer:
            self.mixer.unmute()

        if not self._chunks:
            return None
        return _np.concatenate(self._chunks)


# ==============================================================
#  Tool narration helpers
# ==============================================================

def _tool_narration(name: str, inp: dict) -> str:
    if name == "Read":
        return f"Reading {os.path.basename(inp.get('file_path', 'file'))}"
    if name == "Edit":
        return f"Editing {os.path.basename(inp.get('file_path', 'file'))}"
    if name == "Write":
        return f"Writing {os.path.basename(inp.get('file_path', 'file'))}"
    if name == "Bash":
        cmd = inp.get("command", "")
        first = cmd.split()[0] if cmd.split() else "command"
        return f"Running {os.path.basename(first)}"
    if name == "Grep":
        return f"Searching for {inp.get('pattern', '')[:40]}"
    if name == "Glob":
        return f"Finding files matching {inp.get('pattern', '')[:40]}"
    if name == "create_agent":
        mode = inp.get("mode", "pipeline")
        if mode == "cognitive" and inp.get("prompt"):
            return f"Spawning agent: {inp.get('name', inp.get('prompt', '')[:40])}"
        return f"Creating agent: {inp.get('name', inp.get('id', ''))}"
    if name == "trigger_run":
        return f"Triggering {inp.get('agent_id', '')}"
    return f"Using {name}"


# ==============================================================
#  VoiceService — the main integration point
# ==============================================================

@dataclass(frozen=True)
class VoiceAuthor:
    """The platform-neutral sender for voice input at the input seam.

    Voice has no per-message platform identity the way chat does (a Discord
    user). The local human at the mic is one speaker; the policy sees them as
    a stable id. Exposes the minimum the InputPolicy contract needs —
    ``id`` and ``is_bot`` — so the same gates that govern chat (operator,
    credits, …) can govern voice unchanged."""

    id: str = "voice:local"
    is_bot: bool = False


class VoiceService:
    """ATN voice service — EventBus-driven TTS/STT with audio mixing.

    Requires ``pip install atn[voice]`` for core functionality.

    Listens to:
      - STEP_OUTPUT: speaks orchestrator/delegate text, plays tool tones
      - DELEGATE_SPAWNED/COMPLETED/FAILED: announces delegate lifecycle
      - EXECUTION_COMPLETED/FAILED: result chimes

    Provides:
      - Push-to-talk -> Whisper STT -> route to orchestrator or delegate
      - Per-agent focus (which agent you're listening to)
      - Audio mixing with voice/tools/effects channels

    Feature tiers:
      - ``atn[voice]``: core audio (tones, mixer) — always works if installed
      - ``atn[voice-kokoro]``: local TTS via kokoro-onnx
      - ``atn[voice-edge]``: free cloud TTS via edge-tts
      - ``atn[voice-ptt]``: push-to-talk STT via keyboard + faster-whisper
    """

    def __init__(self, event_bus: EventBus, runtime: Any,
                 config: VoiceConfig | None = None,
                 *, policy: InputPolicy | None = None) -> None:
        self.events = event_bus
        self.runtime = runtime
        self.config = config or VoiceConfig()
        # Input-seam policy — gates voice input before it reaches an agent, the
        # same contract the chat Surface uses. Defaults to AllowAll (today's
        # behavior: PTT goes straight through). A deployment can pass a shared
        # gate (e.g. the chat surface's CreditPolicy) to govern both channels.
        self.policy: InputPolicy = policy or AllowAll()
        self._author = VoiceAuthor()
        # This surface's identity for the InputArbiter (single-writer gate). The
        # local mic is an in-process singleton: kind 'voice', instance 'local'.
        self._surface_id = SurfaceId(
            kind="voice", instance="local", label="Local mic", in_process=True)

        # Focus — which agent_id the user is listening to (per channel)
        self.voice_focus: str = "orchestrator"   # agent whose TTS plays on "voice"
        self.tools_focus: str = "orchestrator"   # agent whose narration plays on "tools"

        # Audio
        self.mixer: AudioMixer | None = None
        self.recorder: PushToTalkRecorder | None = None

        # TTS queue: (text, is_final) — processed by _tts_loop
        self._tts_q: queue.Queue = queue.Queue()
        self._tts_worker: threading.Thread | None = None

        # Narration queue
        self._narrate_q: queue.Queue = queue.Queue()
        self._narrate_worker: threading.Thread | None = None

        # PTT thread
        self._ptt_thread: threading.Thread | None = None
        self._ptt_available = False

        # Announcement cache: pre-rendered audio clips keyed by text
        self._announcement_cache: dict[str, tuple[Any, int]] = {}
        self._cache_lock = threading.Lock()

        # Main event loop reference (set in start())
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # State
        self._running = False
        self._voice_enabled = True
        self._last_spoken_text: dict[str, str] = {}

        # TTS transport flags (ported from kevin). Three orthogonal controls the
        # _tts_loop honors while a response plays:
        #   _tts_cancelled — abandon the WHOLE current item and drain (mute/kill)
        #   _tts_skip      — skip ONE sentence, keep playing the rest
        #   _tts_paused    — set = paused (channel emits silence), clear = running
        self._tts_cancelled = threading.Event()
        self._tts_skip = threading.Event()
        self._tts_paused = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the voice service — mixer, TTS, PTT."""
        if self._running:
            return

        if not VOICE_AVAILABLE:
            raise ImportError(
                "Voice service requires the voice extras.  "
                "Install with: pip install atn[voice]"
            )

        # Configure backend-specific paths from config
        global _kokoro_model_dir, _piper_module_dir
        if self.config.kokoro_model_dir:
            _kokoro_model_dir = self.config.kokoro_model_dir
        if self.config.piper_module_dir:
            _piper_module_dir = self.config.piper_module_dir

        # Validate backend — fall back if configured one isn't available
        avail = _available_backends()
        if self.config.backend not in avail and avail:
            old = self.config.backend
            self.config.backend = avail[0]
            log.warning(
                "TTS backend '%s' not available, falling back to '%s'",
                old, self.config.backend,
            )
        elif not avail:
            log.warning(
                "No TTS backends available — voice service will play tones only"
            )
            self._voice_enabled = False

        self._running = True

        # Resolve output device
        out_dev = None
        if self.config.output_device:
            try:
                out_dev = int(self.config.output_device)
            except ValueError:
                out_dev = find_device(self.config.output_device, kind='output')

        # Resolve input device
        in_dev = None
        if self.config.input_device:
            try:
                in_dev = int(self.config.input_device)
            except ValueError:
                in_dev = find_device(self.config.input_device, kind='input')

        # Build mixer
        self.mixer = AudioMixer(sr=MIXER_SR, device=out_dev)
        self.mixer.add_channel("voice", volume=self.config.voice_volume)
        self.mixer.add_channel("tools", volume=self.config.tools_volume)
        self.mixer.add_channel("effects", volume=self.config.effects_volume)
        self.mixer.start()

        # TTS worker
        self._tts_worker = threading.Thread(
            target=self._tts_loop, daemon=True, name="voice-tts"
        )
        self._tts_worker.start()

        # Narration worker
        self._narrate_worker = threading.Thread(
            target=self._narrate_loop, daemon=True, name="voice-narrate"
        )
        self._narrate_worker.start()

        # PTT — only if atn[voice-ptt] extras are installed
        self._ptt_available = PTT_AVAILABLE
        if self._ptt_available:
            self.recorder = PushToTalkRecorder(
                keys=self.config.ptt_keys,
                mixer=self.mixer,
                device=in_dev,
                on_record_start=self._on_record_start,
            )
            self._ptt_thread = threading.Thread(
                target=self._ptt_loop, daemon=True, name="voice-ptt"
            )
            self._ptt_thread.start()
            log.info("PTT enabled (keys=%s)", self.config.ptt_keys)
        else:
            log.info(
                "PTT disabled — install with: pip install atn[voice-ptt]"
            )

        # Subscribe to events
        self.events.subscribe(EventType.STEP_OUTPUT, self._on_step_output)
        self.events.subscribe(EventType.DELEGATE_SPAWNED, self._on_delegate_spawned)
        self.events.subscribe(EventType.DELEGATE_COMPLETED, self._on_delegate_completed)
        self.events.subscribe(EventType.DELEGATE_FAILED, self._on_delegate_failed)
        self.events.subscribe(EventType.EXECUTION_COMPLETED, self._on_execution_completed)
        self.events.subscribe(EventType.EXECUTION_STARTED, self._on_execution_started)
        self.events.subscribe(EventType.AGENT_REGISTERED, self._on_agent_registered)
        # Security-alarm consumer STUB. The full tamper monitor is the security
        # track; this consumer must be ready + correct now so voice speaks an
        # alert the instant that track emits SECURITY_ALARM.
        self.events.subscribe(EventType.SECURITY_ALARM, self._on_security_alarm)

        # Capture the main event loop for cross-thread scheduling
        self._main_loop = asyncio.get_running_loop()

        # Register with the single-writer arbiter (does NOT grab the mic; the
        # first push-to-talk utterance auto-acquires when the mic is free).
        arbiter = getattr(self.runtime, "input_arbiter", None)
        if arbiter is not None:
            arbiter.register(self._surface_id)

        # Startup chime
        self.mixer.play("effects", make_startup_chime())

        # Pre-render announcement verbs in background
        threading.Thread(
            target=self._warmup_announcement_cache, daemon=True,
            name="voice-cache-warmup",
        ).start()

        log.info(
            "Voice service started (backend=%s, ptt=%s, backends_available=%s)",
            self.config.backend, self._ptt_available, avail,
        )

        await self.events.emit(Event(
            type=EventType.VOICE_STARTED,
            source="voice",
            data={
                "backend": self.config.backend,
                "ptt": self._ptt_available,
                "available_backends": avail,
            },
        ))

    async def stop(self) -> None:
        """Stop the voice service."""
        if not self._running:
            return
        self._running = False

        # Deregister from the single-writer arbiter (hands the mic off if held).
        arbiter = getattr(self.runtime, "input_arbiter", None)
        if arbiter is not None:
            arbiter.release_for(self._surface_id)

        # Unsubscribe
        self.events.unsubscribe(EventType.STEP_OUTPUT, self._on_step_output)
        self.events.unsubscribe(EventType.DELEGATE_SPAWNED, self._on_delegate_spawned)
        self.events.unsubscribe(EventType.DELEGATE_COMPLETED, self._on_delegate_completed)
        self.events.unsubscribe(EventType.DELEGATE_FAILED, self._on_delegate_failed)
        self.events.unsubscribe(EventType.EXECUTION_COMPLETED, self._on_execution_completed)
        self.events.unsubscribe(EventType.EXECUTION_STARTED, self._on_execution_started)
        self.events.unsubscribe(EventType.AGENT_REGISTERED, self._on_agent_registered)
        self.events.unsubscribe(EventType.SECURITY_ALARM, self._on_security_alarm)

        # Stop TTS worker
        self._tts_q.put(None)

        # Stop mixer
        if self.mixer:
            self.mixer.stop_all()
            self.mixer.stop()

        log.info("Voice service stopped")

        await self.events.emit(Event(
            type=EventType.VOICE_STOPPED,
            source="voice",
        ))

    # ------------------------------------------------------------------
    # Focus management
    # ------------------------------------------------------------------

    def set_voice_focus(self, agent_id: str) -> None:
        """Set which agent's responses/thoughts play on the voice channel."""
        old = self.voice_focus
        self.voice_focus = agent_id
        if old != agent_id and self.mixer:
            self.mixer.channels["voice"].clear()
            self.mixer.channels["voice"].fade_reset()
        log.info("Voice focus: %s -> %s", old, agent_id)

    def set_tools_focus(self, agent_id: str) -> None:
        """Set which agent's tool narration plays on the tools channel."""
        old = self.tools_focus
        self.tools_focus = agent_id
        if old != agent_id and self.mixer:
            while not self._narrate_q.empty():
                try:
                    self._narrate_q.get_nowait()
                except queue.Empty:
                    break
            self.mixer.channels["tools"].clear()
            self.mixer.channels["tools"].fade_reset()
        log.info("Tools focus: %s -> %s", old, agent_id)

    def set_focus(self, agent_id: str) -> None:
        """Set both voice and tools focus to the same agent (backward compat)."""
        self.set_voice_focus(agent_id)
        self.set_tools_focus(agent_id)

    def set_voice_enabled(self, enabled: bool) -> None:
        """Enable or disable TTS (tones still play)."""
        self._voice_enabled = enabled

    def set_announcements(self, categories: list[str]) -> None:
        """Set which announcement categories are active."""
        self.config.announcements = list(categories)

    # ------------------------------------------------------------------
    # Announcement cache
    # ------------------------------------------------------------------

    def _cache_verb(self, verb: str) -> None:
        """Render and cache a verb clip."""
        try:
            audio, sr = generate_kokoro(verb, voice="am_michael")
            with self._cache_lock:
                self._announcement_cache[f"_verb_{verb}"] = (audio, sr)
        except Exception:
            try:
                audio, sr = generate_edge(verb)
                with self._cache_lock:
                    self._announcement_cache[f"_verb_{verb}"] = (audio, sr)
            except Exception:
                log.debug("Failed to cache verb: %s", verb)

    def _cache_name(self, name: str) -> None:
        """Render and cache an agent name clip."""
        key = f"_name_{name}"
        with self._cache_lock:
            if key in self._announcement_cache:
                return  # Already cached
        try:
            audio, sr = generate_kokoro(name, voice="am_michael")
            with self._cache_lock:
                self._announcement_cache[key] = (audio, sr)
        except Exception:
            try:
                audio, sr = generate_edge(name)
                with self._cache_lock:
                    self._announcement_cache[key] = (audio, sr)
            except Exception:
                log.debug("Failed to cache name: %s", name)

    def _warmup_announcement_cache(self) -> None:
        """Pre-render announcement verb clips on startup."""
        for verb in ["running", "completed", "failed", "created", "spawning"]:
            self._cache_verb(verb)
        log.info("Announcement verb cache warmed up")

    def _play_cached_announcement(self, name: str, verb: str) -> None:
        """Play a cached name + verb announcement on the tools channel."""
        name_key = f"_name_{name}"
        verb_key = f"_verb_{verb}"
        with self._cache_lock:
            name_clip = self._announcement_cache.get(name_key)
            verb_clip = self._announcement_cache.get(verb_key)
        if name_clip and verb_clip:
            name_audio = name_clip[0]
            verb_audio = verb_clip[0]
            if name_clip[1] != self.mixer.sr:
                name_audio = _resample(name_audio, name_clip[1], self.mixer.sr)
            if verb_clip[1] != self.mixer.sr:
                verb_audio = _resample(verb_audio, verb_clip[1], self.mixer.sr)
            gap = _np.zeros(int(self.mixer.sr * 0.08), dtype=_np.float32)
            combined = _np.concatenate([name_audio, gap, verb_audio])
            self.mixer.play("tools", combined)
        else:
            # Fallback: use narrate queue (renders from scratch)
            self._narrate_q.put(f"{name} {verb}")

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def _generate_tts(self, text: str) -> tuple[Any, int]:
        """Generate TTS audio using the configured backend.

        Falls back through available backends if the primary one fails.
        """
        backend = self.config.backend
        generators = {
            "kokoro": generate_kokoro,
            "edge": generate_edge,
            "elevenlabs": generate_elevenlabs,
            "piper": generate_piper,
        }
        # Try the configured backend first
        if backend in generators:
            try:
                return generators[backend](text)
            except Exception as exc:
                log.warning("TTS backend '%s' failed: %s", backend, exc)

        # Fall back through others
        for name, gen in generators.items():
            if name == backend:
                continue
            try:
                return gen(text)
            except Exception:
                continue

        raise RuntimeError("No TTS backend available")

    def _fade_and_clear_voice(self, fade_secs: float = 0.5) -> None:
        """Crossfade the voice channel out and reset it — the single transition
        primitive used before each new sentence and by skip_current()."""
        if not self.mixer:
            return
        if self.mixer.is_playing("voice"):
            self.mixer.channels["voice"].fade_out(fade_secs, sr=self.mixer.sr)
            time.sleep(fade_secs)
        self.mixer.channels["voice"].clear()
        self.mixer.channels["voice"].fade_reset()

    def _wait_if_paused(self) -> None:
        """Block between sentences while paused (until unpaused or cancelled)."""
        while self._tts_paused.is_set() and not self._tts_cancelled.is_set():
            time.sleep(0.05)

    def _wait_channel_pausable(self, name: str) -> None:
        """Like mixer.wait_channel but returns early on cancel or skip so the
        transport controls take effect mid-sentence."""
        if not self.mixer:
            return
        while self.mixer.channels[name].has_data():
            if self._tts_cancelled.is_set() or self._tts_skip.is_set():
                return
            time.sleep(0.03)

    def pause(self) -> None:
        """Pause TTS playback. The voice channel freezes (emits silence, keeps
        its buffer); the loop halts between sentences. Idempotent."""
        self._tts_paused.set()
        if self.mixer:
            self.mixer.pause("voice")

    def resume(self) -> None:
        """Resume paused TTS playback. Idempotent."""
        self._tts_paused.clear()
        if self.mixer:
            self.mixer.unpause("voice")

    def skip_current(self) -> bool:
        """Skip the currently-playing sentence and advance to the next one in
        the response (or the next queued item) WITHOUT draining the queue — that
        is mute/_kill_all_audio. Uses the dedicated _tts_skip flag so the rest of
        the response keeps playing. Returns False if nothing is playing/queued."""
        if not self.mixer:
            return False
        if not (self.mixer.is_playing("voice") or not self._tts_q.empty()):
            return False
        self._tts_skip.set()
        # If paused, unpause so the skip actually lands.
        if self._tts_paused.is_set():
            self._tts_paused.clear()
            self.mixer.unpause("voice")
        # Crossfade out the current sentence (same feel as a normal transition).
        self._fade_and_clear_voice(fade_secs=0.3)
        return True

    def _tts_loop(self) -> None:
        """Worker thread that processes the TTS queue, honoring the transport
        flags (_tts_cancelled / _tts_skip / _tts_paused)."""
        while True:
            item = self._tts_q.get()
            if item is None:
                break
            text_or_list, is_final = item
            if not self._voice_enabled or not self.mixer:
                self._tts_q.task_done()
                continue
            try:
                # A fresh item starts un-cancelled and un-skipped. (Pause is
                # sticky across items — the user stays paused until they resume.)
                self._tts_cancelled.clear()
                self._tts_skip.clear()
                if is_final:
                    sentences = text_or_list if isinstance(text_or_list, list) else [text_or_list]
                    for sentence in sentences:
                        if self._tts_cancelled.is_set():
                            break
                        self._wait_if_paused()
                        if self._tts_cancelled.is_set():
                            break
                        s = sentence.strip()
                        if not s:
                            continue
                        audio, sr = self._generate_tts(s)
                        if self._tts_cancelled.is_set():
                            break
                        self._fade_and_clear_voice(fade_secs=0.5)
                        self.mixer.play("voice", audio, sr=sr)
                        # Returns early on skip OR cancel. On skip, clear the flag
                        # and fall through to the next sentence (skip_current()
                        # already faded the current one out).
                        self._wait_channel_pausable("voice")
                        if self._tts_skip.is_set():
                            self._tts_skip.clear()
                else:
                    text = text_or_list if isinstance(text_or_list, str) else " ".join(text_or_list)
                    text = text.strip()
                    if not text:
                        self._tts_q.task_done()
                        continue
                    audio, sr = self._generate_tts(text)
                    if not self._tts_cancelled.is_set():
                        self._fade_and_clear_voice(fade_secs=0.5)
                        self.mixer.play("voice", audio, sr=sr)
            except Exception as exc:
                log.warning("TTS error: %s", exc)
            self._tts_q.task_done()

    def _narrate_loop(self) -> None:
        """Worker thread for tool narration (different voice)."""
        while True:
            narration = self._narrate_q.get()
            if narration is None:
                break
            if not self.mixer:
                continue
            try:
                try:
                    audio, sr = generate_kokoro(narration, voice="am_michael")
                except Exception:
                    try:
                        audio, sr = generate_edge(narration)
                    except Exception:
                        continue
                self.mixer.play("tools", audio, sr=sr)
            except Exception:
                pass

    def _speak(self, text: str, is_final: bool = False) -> None:
        """Queue text for TTS."""
        if not self._voice_enabled:
            return
        clean = strip_markdown(text)
        if not clean:
            return
        if is_final:
            parts = SENTENCE_END.split(clean)
            sentences = [s.strip() for s in parts if s.strip()]
            if sentences:
                self._tts_q.put((sentences, True))
        else:
            self._tts_q.put((clean, False))

    # ------------------------------------------------------------------
    # Event handlers (async, called from EventBus)
    # ------------------------------------------------------------------

    async def _on_step_output(self, event: Event) -> None:
        """Handle step.output events — route text or play tool tones."""
        data = event.data
        source = event.source
        channel = data.get("channel", "text")

        if channel == "tool_call":
            tool_name = data.get("tool_name", "")
            tool_input = data.get("tool_input", {})
            if source == self.tools_focus:
                if self.mixer:
                    self.mixer.play("effects", make_tool_tone(tool_name))
                if self.config.narrate_tools:
                    narration = _tool_narration(tool_name, tool_input)
                    while not self._narrate_q.empty():
                        try:
                            self._narrate_q.get_nowait()
                        except queue.Empty:
                            break
                    self._narrate_q.put(narration)
            return

        if channel == "text":
            content = data.get("content", "")
            if not content:
                return
            agent_id = data.get("agent_id", source)
            if agent_id != self.voice_focus:
                return
            last = self._last_spoken_text.get(agent_id, "")
            if content == last:
                return
            self._last_spoken_text[agent_id] = content
            self._speak(content, is_final=False)

    async def _on_delegate_spawned(self, event: Event) -> None:
        if self.mixer:
            self.mixer.play("effects", make_delegate_spawn_tone())
        if "delegate_lifecycle" not in self.config.announcements:
            return
        title = event.data.get("title", "")
        if title and self._voice_enabled:
            self._play_cached_announcement(title, "spawning")

    async def _on_delegate_completed(self, event: Event) -> None:
        if self.mixer:
            self.mixer.play("effects", make_result_chime())
        agent_id = event.data.get("agent_id", "")
        # Focused agent always gets result preview spoken
        if agent_id == self.voice_focus:
            preview = event.data.get("result_preview", "")
            if preview:
                self._speak(preview, is_final=True)
        # Announcement for non-focused delegates
        elif "agent_completed" in self.config.announcements and self._voice_enabled:
            title = event.data.get("title", agent_id)
            self._play_cached_announcement(title, "completed")

    async def _on_delegate_failed(self, event: Event) -> None:
        if self.mixer:
            self.mixer.play("effects", _make_failure_tone())
        if "delegate_lifecycle" not in self.config.announcements:
            return
        agent_id = event.data.get("agent_id", "")
        if agent_id and self._voice_enabled:
            title = event.data.get("title", agent_id)
            self._play_cached_announcement(title, "failed")

    async def _on_execution_completed(self, event: Event) -> None:
        if self.mixer:
            self.mixer.play("effects", make_result_chime())
        if "agent_completed" not in self.config.announcements:
            return
        agent_id = event.data.get("agent_id", event.source)
        if agent_id and self._voice_enabled:
            self._play_cached_announcement(agent_id, "completed")

    async def _on_execution_started(self, event: Event) -> None:
        """Announce when an agent execution starts."""
        if "agent_runs" not in self.config.announcements:
            return
        agent_id = event.data.get("agent_id", event.source)
        if self._voice_enabled:
            self._play_cached_announcement(agent_id, "running")

    async def _on_agent_registered(self, event: Event) -> None:
        """Cache agent name and announce when a new agent is created."""
        agent_id = event.data.get("agent_id", "")
        if agent_id:
            # Cache name clip in background (don't block event handler)
            threading.Thread(
                target=self._cache_name, args=(agent_id,), daemon=True,
            ).start()
        if "agent_created" not in self.config.announcements:
            return
        if agent_id and self._voice_enabled:
            self._play_cached_announcement(agent_id, "created")

    # ------------------------------------------------------------------
    # Security-alarm consumer (STUB — full monitor is the security track)
    # ------------------------------------------------------------------

    async def _on_security_alarm(self, event: Event) -> None:
        """Speak a security alarm — names only — the instant one fires.

        A tamper alert must NOT be swallowed by mute, a disabled voice, or a
        cloud-TTS outage: it deliberately ignores ``_voice_enabled`` and (when
        ``piper_mandatory_for_alarm``) renders through Piper, the offline
        backend, unmuting the mixer so the alert is audible. SECURITY_ALARM
        carries NO agent_id (no subtree scoping) — data is names only.
        """
        names = event.data.get("names") or []
        if isinstance(names, str):
            names = [names]
        names = [str(n).strip() for n in names if str(n).strip()]
        if not names:
            return
        spoken = ", ".join(names)
        # Render + play off the event loop so the handler doesn't block the bus.
        threading.Thread(
            target=self._speak_alarm, args=(spoken,), daemon=True,
            name="voice-alarm",
        ).start()

    def _speak_alarm(self, text: str) -> None:
        """Render an alarm clip through Piper (offline, always available) and
        force it out on the effects channel, bypassing mute and TTS enable."""
        if not self.mixer:
            return
        try:
            if self.config.piper_mandatory_for_alarm:
                audio, sr = generate_piper(text)
            else:
                audio, sr = self._generate_tts(text)
        except Exception as exc:
            log.warning("[voice] security alarm TTS failed: %s", exc)
            return
        # A security alert overrides mute — unmute, play, and let the caller's
        # normal state stand afterward (the alarm is a one-shot on effects).
        self.mixer.unmute()
        self.mixer.unpause("voice")
        self.mixer.play("effects", audio, sr=sr)

    # ------------------------------------------------------------------
    # Push-to-talk loop
    # ------------------------------------------------------------------

    def _on_record_start(self) -> None:
        self._kill_all_audio()
        # A physical PTT press is the human at the local mic asking to speak, so
        # proactively acquire the input token now (not lazily at send time). The
        # switch-authorizer is stubbed AllowSwitch today, so this always wins;
        # the seam is where a future preemption policy would gate it.
        arbiter = getattr(self.runtime, "input_arbiter", None)
        if arbiter is not None:
            try:
                self._run_on_main_loop(arbiter.request_input(self._surface_id))
            except Exception as exc:
                log.debug("[PTT] Failed to acquire input token: %s", exc)
        # Signal to UI that recording has started
        try:
            self._run_on_main_loop(self.events.emit(Event(
                type=EventType.VOICE_RECORDING,
                source="voice",
                data={"recording": True},
            )))
        except Exception as exc:
            log.debug("[PTT] Failed to emit recording start event: %s", exc)

    def _kill_all_audio(self) -> None:
        # Cancel the in-flight TTS item so the loop abandons the current
        # response mid-sentence; clear pause/skip so the next item starts clean
        # and unpause the channel (a paused channel would otherwise swallow it).
        self._tts_cancelled.set()
        self._tts_skip.clear()
        self._tts_paused.clear()
        if self.mixer:
            self.mixer.unpause("voice")
            self.mixer.stop_all()
            for ch in self.mixer.channels.values():
                ch.fade_reset()
        while not self._tts_q.empty():
            try:
                self._tts_q.get_nowait()
                self._tts_q.task_done()
            except queue.Empty:
                break
        while not self._narrate_q.empty():
            try:
                self._narrate_q.get_nowait()
            except queue.Empty:
                break

    def _run_on_main_loop(self, coro) -> None:
        """Schedule a coroutine on the main event loop and wait for it."""
        future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        future.result(timeout=60)

    def _ptt_loop(self) -> None:
        """Background thread: wait for PTT, record, transcribe, route."""
        while self._running:
            try:
                audio = self.recorder.wait_and_record()
                # Signal recording stopped
                try:
                    self._run_on_main_loop(self.events.emit(Event(
                        type=EventType.VOICE_RECORDING,
                        source="voice",
                        data={"recording": False},
                    )))
                except Exception:
                    pass
                if audio is None or len(audio) < self.recorder.sr * 0.3:
                    continue

                text = transcribe(audio, sr=self.recorder.sr)
                if not text:
                    continue

                log.info("[PTT] Transcribed: %s", text[:80])

                self._run_on_main_loop(self.events.emit(Event(
                    type=EventType.VOICE_TRANSCRIBED,
                    source="voice",
                    data={"text": text, "target": self.voice_focus},
                )))

                self._run_on_main_loop(
                    self._send_to_agent(self.voice_focus, text)
                )

            except Exception as exc:
                log.warning("[PTT] Error: %s", exc)
                time.sleep(0.5)

    async def _send_to_agent(self, agent_id: str, text: str) -> None:
        """Send voice input to any agent (orchestrator or delegate).

        This is the voice Surface's INPUT SEAM — the one place inbound voice
        becomes runtime.send_agent_message. The InputPolicy is evaluated here,
        before delivery, exactly as the chat Surface gates handle_inbound.

        Uses send_agent_message which handles all cases correctly:
        - Records user turn in conversation store
        - Injects mid-session if agent is running (via bridge)
        - Re-activates COMPLETED/ERROR agents and triggers new execution
        """
        from .orchestrator import ORCHESTRATOR_ID

        # Input-seam gate. A denied utterance is dropped (optionally spoken back
        # as the deny reason) and never reaches an agent.
        decision = self.policy.evaluate(self._author, agent_id, text)
        if not decision.allow:
            log.info("[voice] input denied for %s: %s", agent_id, decision.reason)
            if decision.reason and self._voice_enabled:
                self._speak(decision.reason, is_final=True)
            return

        tagged = f"🎤 [Voice Input] {text}"

        # Try to deliver to the focused agent first. send_agent_message ALWAYS
        # returns a dict; treat a non-empty "error"/"code" as failure (the old
        # `if not delivered` was dead code — a dict is always truthy).
        result = await self.runtime.send_agent_message(
            agent_id, tagged, surface=self._surface_id)

        # If the arbiter denied this surface the mic, the utterance is dropped
        # here (a different surface holds input) — do NOT reroute to the
        # orchestrator, which would just be denied too.
        if result.get("code") == "input_not_active":
            log.info("[voice] input not active (mic held by %s); dropping",
                     result.get("holder"))
            return

        # If delivery failed, fall back to the legacy root agent — but only
        # when one actually exists (rootless fleets have no fallback target).
        if result.get("error") and agent_id != ORCHESTRATOR_ID \
                and self.runtime.get_agent(ORCHESTRATOR_ID) is not None:
            log.warning(
                "[PTT] Agent '%s' is not available, routing to orchestrator",
                agent_id,
            )
            await self.runtime.send_agent_message(
                ORCHESTRATOR_ID, tagged, surface=self._surface_id)

    # ------------------------------------------------------------------
    # Status for WS/UI
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "backend": self.config.backend,
            "voice_enabled": self._voice_enabled,
            "voice_focus": self.voice_focus,
            "tools_focus": self.tools_focus,
            "focused_agent": self.voice_focus,  # backward compat
            "ptt_available": self._ptt_available,
            "ptt_keys": self.config.ptt_keys,
            "narrate_tools": self.config.narrate_tools,
            "announcements": self.config.announcements,
            "available_backends": _available_backends() if self._running else [],
            "playing": self.mixer.is_playing() if self.mixer else False,
            # Whether THIS (local mic) surface currently holds the input token.
            "owns_local_audio": self._owns_local_audio(),
            "tts_paused": self._tts_paused.is_set(),
        }

    def _owns_local_audio(self) -> bool:
        """True if the local mic surface holds the single-writer input token."""
        arbiter = getattr(self.runtime, "input_arbiter", None)
        if arbiter is None:
            return False
        return arbiter.holder_token() == self._surface_id.token
