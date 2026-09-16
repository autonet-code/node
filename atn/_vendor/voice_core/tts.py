"""TTS backends — Kokoro (local), Edge (free cloud), ElevenLabs (paid), Piper (offline).

All heavy deps are lazy-imported, so importing this module is cheap and each
backend fails with a clear error only when actually used without its dependency.

Each ``generate_*`` returns ``(audio_float32_mono, sample_rate)``.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import wave
from typing import Any

import numpy as np


# ==============================================================
#  Kokoro (local, default)
# ==============================================================

_kokoro_model = None
_kokoro_lock = threading.Lock()
# Configurable: directory holding kokoro-v1.0.onnx + voices-v1.0.bin.
# Falls back to looking next to this file.
_kokoro_model_dir: str | None = None


def set_kokoro_model_dir(path: str | None) -> None:
    """Override where the Kokoro model files are searched for."""
    global _kokoro_model_dir
    _kokoro_model_dir = path


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
                        "Install with: pip install kokoro-onnx"
                    )
                search_dirs = []
                if _kokoro_model_dir:
                    search_dirs.append(_kokoro_model_dir)
                search_dirs.append(os.path.dirname(os.path.abspath(__file__)))
                onnx_path = None
                voices_path = None
                for d in search_dirs:
                    p = os.path.join(d, "kokoro-v1.0.onnx")
                    if os.path.exists(p):
                        onnx_path = p
                        v = os.path.join(d, "voices-v1.0.bin")
                        if os.path.exists(v):
                            voices_path = v
                        break
                if onnx_path is None:
                    # No local model files — pull from HuggingFace (weights are
                    # NOT shipped in the package). Cached by huggingface_hub, so
                    # this download happens once per machine.
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError:
                        raise FileNotFoundError(
                            "kokoro-v1.0.onnx not found locally and "
                            "huggingface_hub is not installed. Either call "
                            "voice_core.set_kokoro_model_dir(...) with a dir "
                            "containing kokoro-v1.0.onnx + voices-v1.0.bin, or "
                            "`pip install huggingface_hub` to auto-download."
                        )
                    repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
                    onnx_path = hf_hub_download(repo, "onnx/model.onnx")
                    voices_path = hf_hub_download(repo, "voices.bin")
                _kokoro_model = kokoro_onnx.Kokoro(onnx_path, voices_path)
    return _kokoro_model


def generate_kokoro(text: str, voice: str = "af_heart") -> tuple[Any, int]:
    kokoro = _get_kokoro()
    with _kokoro_lock:
        samples, sr = kokoro.create(text, voice=voice, speed=1.0)
    return samples, sr


# ==============================================================
#  Edge TTS (free, cloud)
# ==============================================================

_edge_loop = None
_edge_lock = threading.Lock()


def _get_edge_loop():
    global _edge_loop
    if _edge_loop is None:
        with _edge_lock:
            if _edge_loop is None:
                import asyncio
                _edge_loop = asyncio.new_event_loop()
                t = threading.Thread(target=_edge_loop.run_forever, daemon=True)
                t.start()
    return _edge_loop


def generate_edge(text: str, voice: str = "en-US-JennyNeural") -> tuple[Any, int]:
    import asyncio

    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "Edge backend requires edge-tts.  Install with: pip install edge-tts"
        )
    try:
        import miniaudio
    except ImportError:
        raise ImportError(
            "Edge backend requires miniaudio.  Install with: pip install miniaudio"
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
        audio = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, 24000
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ==============================================================
#  ElevenLabs (paid, cloud)
# ==============================================================

def generate_elevenlabs(text: str, voice_id: str | None = None,
                        voice_settings: dict | None = None) -> tuple[Any, int]:
    try:
        import requests
    except ImportError:
        raise ImportError(
            "ElevenLabs backend requires requests.  Install with: pip install requests"
        )
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_v3")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_24000"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id}
    if voice_settings:
        payload["voice_settings"] = voice_settings
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    pcm = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, 24000


# ==============================================================
#  Piper (offline) — requires an external `voice` module
# ==============================================================

# Configurable: directory containing the `voice` package (voice.py) that
# provides Piper TTS.  kevin sets this to c:\code\voice; autonet sets it via
# VoiceConfig.  If None, the `voice` package must already be importable.
_piper_module_dir: str | None = None


def set_piper_module_dir(path: str | None) -> None:
    """Point the Piper backend at the directory holding the `voice` package."""
    global _piper_module_dir
    _piper_module_dir = path


def generate_piper(text: str, voice: str = "male") -> tuple[Any, int]:
    if _piper_module_dir and _piper_module_dir not in sys.path:
        sys.path.insert(0, _piper_module_dir)
    try:
        from voice import _load_voices, _voice_lock, is_local_tts_available
    except ImportError:
        raise ImportError(
            "Piper backend requires the `voice` package.  Call "
            "voice_core.set_piper_module_dir(...) to point at it, or ensure it "
            "is on sys.path."
        )
    if not is_local_tts_available():
        raise RuntimeError("Piper TTS not available")
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
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr
