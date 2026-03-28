"""MCP server for voice / text-to-speech -- lets agents speak aloud.

Wraps an external voice module as MCP tools so that any
ATN agent can produce audible speech on the user's machine.

Supports two backends:
  - **local**: Piper TTS (offline, free)
  - **elevenlabs**: ElevenLabs cloud API (natural voices, requires API key)

Runs as a subprocess launched by ConnectorManager.

Usage (standalone):
    python server.py
    python server.py --voice-dir /path/to/voice

Usage (via ATN):
    Launched automatically by ConnectorManager.start("voice")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

# All logging to stderr -- stdout is reserved for MCP protocol.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice.mcp")

# -- Configuration -------------------------------------------------------------

VOICE_MODULE_DIR = os.environ.get(
    "VOICE_MODULE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "voice"),
)

# -- MCP Server ----------------------------------------------------------------

mcp = FastMCP("Voice")

# -- Lazy import of voice module -----------------------------------------------

_voice = None


def _get_voice():
    """Lazily import the voice module."""
    global _voice
    if _voice is not None:
        return _voice

    if VOICE_MODULE_DIR not in sys.path:
        sys.path.insert(0, VOICE_MODULE_DIR)

    try:
        import voice as v
        _voice = v
        log.info("voice module loaded from %s", VOICE_MODULE_DIR)
        return v
    except ImportError as exc:
        log.error("failed to import voice module from %s: %s", VOICE_MODULE_DIR, exc)
        raise


# -- Tools ---------------------------------------------------------------------


@mcp.tool()
async def say(text: str, backend: str = "", voice: str = "male") -> str:
    """Speak text aloud on the user's computer using text-to-speech.

    This is the primary tool for producing audible speech. The text is
    synthesized and played through the system speakers immediately.

    Args:
        text: The text to speak aloud. Keep it concise and natural-sounding.
        backend: TTS backend to use. Options:
            - "" (empty): use the current default backend
            - "local": offline Piper TTS (free, robotic)
            - "elevenlabs": cloud ElevenLabs API (natural, requires API key)
        voice: Voice gender for local TTS: "male" or "female".
               Ignored when using elevenlabs backend.
    """
    v = _get_voice()

    kwargs = {"voice": voice}
    if backend:
        kwargs["backend"] = backend

    try:
        ok = v.speak(text, **kwargs)
        return json.dumps({"ok": ok, "text": text, "backend": backend or v.get_backend()})
    except Exception as exc:
        log.exception("speak failed")
        return json.dumps({"ok": False, "error": str(exc)})


@mcp.tool()
async def announce(prefix: str, message: str, backend: str = "") -> str:
    """Make a two-part announcement with a beep.

    Plays a beep, then speaks the prefix (male voice), then the message
    (female voice) when using local TTS. With ElevenLabs, combines both
    into a single utterance.

    Useful for status announcements like announce("Weather update",
    "It's currently 72 degrees and sunny").

    Args:
        prefix: Short label or category (e.g. "Alert", "Reminder", "Update")
        message: The main announcement content
        backend: TTS backend: "" (default), "local", or "elevenlabs"
    """
    v = _get_voice()

    kwargs = {"blocking": True}
    if backend:
        kwargs["backend"] = backend

    try:
        v.announce(prefix, message, **kwargs)
        return json.dumps({
            "ok": True,
            "prefix": prefix,
            "message": message,
            "backend": backend or v.get_backend(),
        })
    except Exception as exc:
        log.exception("announce failed")
        return json.dumps({"ok": False, "error": str(exc)})


@mcp.tool()
async def play_beep() -> str:
    """Play a short alert beep sound.

    A quick two-tone chime useful for drawing attention before
    an important message or signaling that something happened.
    """
    v = _get_voice()
    try:
        v.play_beep()
        return json.dumps({"ok": True})
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


@mcp.tool()
async def set_backend(backend: str) -> str:
    """Switch the default TTS backend.

    Args:
        backend: "local" for offline Piper TTS, or "elevenlabs" for
                 cloud-based natural voices.
    """
    v = _get_voice()
    try:
        v.set_backend(backend)
        return json.dumps({"ok": True, "backend": backend})
    except ValueError as exc:
        return json.dumps({"ok": False, "error": str(exc)})


@mcp.tool()
async def voice_status() -> str:
    """Get the current status of the voice system.

    Returns which backend is active, whether each backend is available,
    and the current mute state.
    """
    v = _get_voice()
    return json.dumps(v.get_status())


@mcp.tool()
async def set_muted(muted: bool) -> str:
    """Mute or unmute voice output.

    Args:
        muted: True to mute all voice output, False to unmute.
    """
    v = _get_voice()
    v.set_muted(muted)
    return json.dumps({"ok": True, "muted": muted})


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice MCP server")
    parser.add_argument(
        "--voice-dir",
        default=VOICE_MODULE_DIR,
        help="Path to the voice module directory",
    )
    args = parser.parse_args()
    VOICE_MODULE_DIR = args.voice_dir
    mcp.run(transport="stdio")
