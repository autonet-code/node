"""Tone and chime generation for tool/result/status audio cues.

kevin's tool tones + result chime, plus autonet's startup and failure chimes.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .audio import MIXER_SR

# Tool name -> sound category
TOOL_SOUND_MAP = {
    "Bash": "execute", "Read": "read", "Edit": "edit", "Write": "write",
    "Grep": "search", "Glob": "search", "Task": "agent",
    "WebFetch": "web", "WebSearch": "web", "TodoWrite": "todo",
    "NotebookEdit": "edit",
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


def gen_tone(freq: float, dur: float = 0.09, vol: float = 0.25, sr: int = MIXER_SR) -> Any:
    n = int(sr * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    tone = vol * np.sin(2 * np.pi * freq * t)
    att = min(int(0.005 * sr), n)
    rel = min(int(0.015 * sr), n)
    tone[:att] *= np.linspace(0, 1, att, dtype=np.float32)
    tone[-rel:] *= np.linspace(1, 0, rel, dtype=np.float32)
    return tone


def register_tool_sounds(mapping: dict) -> None:
    """Teach the shared tone map about host-specific tools.

    A consumer with its own vocabulary (autonet's orchestrator tools, say)
    registers them here rather than shadowing TOOL_SOUND_MAP locally — a local
    copy would be ignored, because make_tool_tone reads this module's map.
    """
    TOOL_SOUND_MAP.update(mapping)


def make_tool_tone(tool_name: str) -> Any:
    cat = TOOL_SOUND_MAP.get(tool_name, "default")
    freqs = TOOL_FREQS.get(cat, TOOL_FREQS["default"])
    gap = np.zeros(int(MIXER_SR * 0.015), dtype=np.float32)
    parts = []
    for f in freqs:
        parts.append(gen_tone(f))
        parts.append(gap)
    return np.concatenate(parts)


def make_result_chime() -> Any:
    c5 = gen_tone(523, dur=0.12, vol=0.18)
    g5 = gen_tone(784, dur=0.15, vol=0.15)
    gap = np.zeros(int(MIXER_SR * 0.04), dtype=np.float32)
    return np.concatenate([c5, gap, g5])


def make_startup_chime() -> Any:
    """C-E-G ascending startup chime (from autonet)."""
    c5 = gen_tone(523, dur=0.08, vol=0.3)
    e5 = gen_tone(659, dur=0.08, vol=0.3)
    g5 = gen_tone(784, dur=0.12, vol=0.33)
    gap = np.zeros(int(MIXER_SR * 0.015), dtype=np.float32)
    return np.concatenate([c5, gap, e5, gap, g5])


def make_failure_tone() -> Any:
    """Descending two-note failure tone (from autonet)."""
    return np.concatenate([
        gen_tone(440, dur=0.15, vol=0.3),
        np.zeros(int(MIXER_SR * 0.03), dtype=np.float32),
        gen_tone(330, dur=0.2, vol=0.3),
    ])
