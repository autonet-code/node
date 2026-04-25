"""Per-model context-window and output-limit specs.

Centralised so bridges and cost/occupancy math share one table.  Uses
longest-prefix matching against the model ID, so ``gpt-5.5-preview`` falls
through to the ``gpt-5.5`` entry, ``claude-opus-4-7-20260320`` to
``claude-opus-4-7``, etc.
"""
from __future__ import annotations


_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic — all the 4.x family ship 200K by default; Sonnet/Opus 4.7
    # support 1M with the context-1m-2025-08-07 beta flag (enabled by the
    # claude-bridge for those models).
    "claude-opus-4-7":    1_000_000,
    "claude-sonnet-4-7":  1_000_000,
    "claude-sonnet-4-6":  1_000_000,
    "claude-opus-4-6":      200_000,
    "claude-haiku-4-5":     200_000,
    "claude-sonnet-4":      200_000,
    "claude-haiku-4":       200_000,
    "claude-":              200_000,

    # OpenAI (Codex CLI)
    "gpt-5.5":              400_000,
    "gpt-5":                400_000,
    "gpt-4.1":            1_000_000,
    "gpt-4o":               128_000,
    "gpt-4-turbo":          128_000,
    "gpt-4":                  8_192,
    "gpt-3.5":               16_385,
    "o3":                   200_000,
    "o4-mini":              200_000,
    "o1":                   200_000,

    # Google Gemini
    "gemini-3-pro":       1_000_000,
    "gemini-3-flash":     1_000_000,
    "gemini-2.5-pro":     2_000_000,
    "gemini-2.5-flash":   1_000_000,
    "gemini-2.0-pro":     2_000_000,
    "gemini-2.0-flash":   1_000_000,
    "gemini-1.5-pro":     2_000_000,
    "gemini-1.5-flash":   1_000_000,
}

_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "claude-opus-4-7":    64_000,
    "claude-sonnet-4-7":  64_000,
    "claude-sonnet-4-6":  64_000,
    "claude-opus-4-6":    32_000,
    "claude-haiku-4-5":    8_192,
    "claude-":             8_192,

    "gpt-5.5":           128_000,
    "gpt-5":             128_000,
    "gpt-4.1":            32_000,
    "gpt-4o":             16_384,
    "o3":                100_000,
    "o4-mini":            65_536,
    "o1":                 32_768,

    "gemini-3-pro":       64_000,
    "gemini-3-flash":     65_536,
    "gemini-2.5-pro":     65_536,
    "gemini-2.5-flash":   65_536,
}

_DEFAULT_CONTEXT = 128_000
_DEFAULT_MAX_OUTPUT = 8_192


def _lookup(table: dict[str, int], model_id: str, default: int) -> int:
    """Longest-prefix match against the table."""
    if not model_id:
        return default
    lower = model_id.lower()
    best_len = 0
    best_val = default
    for prefix, val in table.items():
        if lower.startswith(prefix.lower()) and len(prefix) > best_len:
            best_len = len(prefix)
            best_val = val
    return best_val


def context_window(model_id: str) -> int:
    """Return the total context window (input + output) for a model."""
    return _lookup(_CONTEXT_WINDOWS, model_id, _DEFAULT_CONTEXT)


def max_output_tokens(model_id: str) -> int:
    """Return the max output-token limit for a single response."""
    return _lookup(_MAX_OUTPUT_TOKENS, model_id, _DEFAULT_MAX_OUTPUT)
