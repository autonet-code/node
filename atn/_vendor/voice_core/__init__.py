r"""Vendored voice_core: shared low-level audio primitives.

Source of truth: C:\code\voice-core (distribution ``autonet-voice-core``).
Vendored here so ``pip install autonet-computer[voice]`` actually brings the
primitives the voice service imports. The published 0.1.0 wheel predates
``register_tool_sounds`` / ``set_stt_backend``, so depending on it would still
ImportError; carrying the source keeps the extras self-contained. Re-sync by
copying the package files over when upstream changes.

Original module docstring follows.

Shared low-level audio primitives for the kevin and autonet voice services.

This package holds ONLY the reusable primitives that both the standalone
``kevin`` voice service and the ``autonet`` EventBus-based ``VoiceService``
need: real-time audio mixing, TTS backends, push-to-talk recording, Whisper
transcription, tone generation, and TTS text cleanup.

Architecture-specific glue (UDP instance registry, hooks, window focus and the
overlay in kevin; the VoiceService class, EventBus wiring and VoiceConfig in
autonet) stays in each repo — it does NOT belong here.

Heavy / optional dependencies (kokoro_onnx, edge_tts, miniaudio, keyboard,
requests) are lazy-imported inside the functions that use them, so importing
this package only requires numpy + sounddevice.  Speech-to-text runs entirely
out-of-process (an isolated NeMo server, see stt.py), so voice_core has no
NeMo / faster-whisper dependency at all.
"""

from .audio import (
    MIXER_SR,
    AudioChannel,
    AudioMixer,
    find_device,
    get_device_list,
    resample,
)
from .tts import (
    generate_edge,
    generate_elevenlabs,
    generate_kokoro,
    generate_piper,
    set_kokoro_model_dir,
    set_piper_module_dir,
)
from .stt import (
    PushToTalkRecorder,
    transcribe,
    ensure_asr_server,
    set_asr_server,
    set_stt_backend,
    get_stt_backend,
    preload_stt,
)
from .tones import (
    TOOL_FREQS,
    TOOL_SOUND_MAP,
    gen_tone,
    make_failure_tone,
    make_result_chime,
    make_startup_chime,
    make_tool_tone,
    register_tool_sounds,
)
from .text import basename, strip_markdown, tool_narration, tool_summary

__all__ = [
    # audio
    "MIXER_SR",
    "AudioChannel",
    "AudioMixer",
    "find_device",
    "get_device_list",
    "resample",
    # tts
    "generate_kokoro",
    "generate_edge",
    "generate_elevenlabs",
    "generate_piper",
    "set_kokoro_model_dir",
    "set_piper_module_dir",
    # stt
    "PushToTalkRecorder",
    "transcribe",
    "ensure_asr_server",
    "set_asr_server",
    "set_stt_backend",
    "get_stt_backend",
    "preload_stt",
    # tones
    "TOOL_SOUND_MAP",
    "TOOL_FREQS",
    "gen_tone",
    "make_tool_tone",
    "register_tool_sounds",
    "make_result_chime",
    "make_startup_chime",
    "make_failure_tone",
    # text
    "strip_markdown",
    "basename",
    "tool_summary",
    "tool_narration",
]
