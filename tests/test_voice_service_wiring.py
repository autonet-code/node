"""Voice service wiring: packaging, focus resolution, tools voice.

Cheap, import-level checks for the three ways voice silently did nothing:
the shared audio primitives were not shipped, the focus gate compared against
a retired sentinel that names no agent, and the tool-narration channel ignored
the selected TTS backend.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def test_voice_core_is_vendored():
    """The primitives ship inside the wheel, so `[voice]` is self-contained."""
    mod = importlib.import_module("atn._vendor.voice_core")
    for name in (
        "MIXER_SR", "AudioMixer", "AudioChannel", "PushToTalkRecorder",
        "generate_kokoro", "generate_edge", "generate_elevenlabs",
        "generate_piper", "transcribe", "make_result_chime",
        "make_startup_chime", "make_tool_tone", "gen_tone", "strip_markdown",
        "register_tool_sounds", "set_stt_backend", "TOOL_FREQS",
        "TOOL_SOUND_MAP",
    ):
        assert hasattr(mod, name), f"vendored voice_core is missing {name}"


def test_voice_service_falls_back_to_the_vendored_copy():
    """With no separately installed voice_core, the import still resolves."""
    blocked = {"voice_core"}

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name in blocked or name.split(".")[0] in blocked:
                raise ImportError(f"blocked: {name}")
            return None

    saved = {k: v for k, v in sys.modules.items()
             if k == "atn.voice_service" or k.split(".")[0] == "voice_core"}
    for k in saved:
        sys.modules.pop(k, None)
    sys.meta_path.insert(0, _Block())
    try:
        vs = importlib.import_module("atn.voice_service")
        assert vs.AudioMixer.__module__.startswith("atn._vendor.voice_core")
    finally:
        sys.meta_path.pop(0)
        sys.modules.pop("atn.voice_service", None)
        sys.modules.update(saved)


def test_install_hint_names_the_real_distribution():
    """`pip install atn[voice]` installs an unrelated package."""
    import atn.voice_service as vs
    assert "autonet-computer[" in vs.VOICE_INSTALL_HINT
    assert "atn[" not in vs.VOICE_INSTALL_HINT


def test_tools_channel_uses_a_second_voice_on_the_selected_backend():
    """Tool narration must not hardcode kokoro, or the contrast disappears
    the moment the user picks another backend."""
    import atn.voice_service as vs
    from atn.config import VoiceConfig

    svc = vs.VoiceService.__new__(vs.VoiceService)
    svc.config = VoiceConfig(backend="edge", tools_voice="en-US-GuyNeural")

    seen = {}

    def _fake(text, *, backend=None, voice=None):
        seen["backend"] = backend
        seen["voice"] = voice
        return (None, 0)

    svc._generate_tts = _fake
    svc._generate_tools_tts("hello")
    assert seen == {"backend": "edge", "voice": "en-US-GuyNeural"}


def test_generate_tts_passes_the_voice_to_the_chosen_backend(monkeypatch):
    import atn.voice_service as vs
    from atn.config import VoiceConfig

    calls = []

    def _kokoro(text, voice="af_heart"):
        calls.append(("kokoro", voice))
        return (None, 24000)

    monkeypatch.setattr(vs, "generate_kokoro", _kokoro)
    svc = vs.VoiceService.__new__(vs.VoiceService)
    svc.config = VoiceConfig(backend="kokoro")
    svc._generate_tts("hi", backend="kokoro", voice="am_michael")
    assert calls == [("kokoro", "am_michael")]


def test_narrate_tools_setter_exists():
    """The 'everything vs responses only' choice needs a runtime setter for
    the WS handler to reach."""
    import atn.voice_service as vs
    svc = vs.VoiceService.__new__(vs.VoiceService)
    from atn.config import VoiceConfig
    svc.config = VoiceConfig()
    svc.set_narrate_tools(False)
    assert svc.config.narrate_tools is False
    svc.set_narrate_tools(True)
    assert svc.config.narrate_tools is True


def test_ws_resolves_the_legacy_focus_sentinel():
    """'orchestrator' names no agent on a fresh fleet; focus must remap onto
    the session root or the speak gate never matches."""
    from atn import ws_server

    server = ws_server.WebSocketBridge.__new__(ws_server.WebSocketBridge)
    runtime = types.SimpleNamespace(get_agent=lambda aid: None)
    server.runtime = runtime
    server._session_root_agent = lambda session: "kevin"

    session = object()
    assert server._resolve_focus_agent(session, "orchestrator") == "kevin"
    # A real agent id is passed through untouched.
    assert server._resolve_focus_agent(session, "kevin") == "kevin"
    # And a fleet that really does carry the legacy id keeps it.
    runtime.get_agent = lambda aid: object()
    assert server._resolve_focus_agent(session, "orchestrator") == "orchestrator"
