"""Model store — single source of truth for available models.

Each entry carries everything the framework needs to know about a model:
identity, family/class, context window, output cap, and the relative
subscription cost weight used to bootstrap the per-model estimator.

Lookups use longest-prefix matching against the model ID, so
``claude-opus-4-7-20260320`` falls through to ``claude-opus-4-7``,
``gpt-5.5-preview`` to ``gpt-5.5``, etc.

The store is intentionally provider-agnostic: a model like Opus 4.7 can be
reached through claude_max bridge OR a direct anthropic API key. The
``default_channel`` is a hint, not a constraint — whichever provider is
configured will route the call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about a model that the framework needs to know.

    ``relative_cost`` is the model's cost weight relative to Sonnet (1.0).
    Used to bootstrap the per-model tokens-per-pct estimator before real
    observations come in. Anthropic publishes rough subscription multipliers;
    these are best-effort starting points, not guarantees.
    """
    id: str                             # canonical model id (prefix-matched)
    family: str                         # "claude" | "gpt" | "gemini" | "deepseek" | ...
    klass: str = "other"                # "haiku" | "sonnet" | "opus" | "other"
    display_name: str = ""              # human-readable label
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    relative_cost: float = 1.0          # multiplier vs. Sonnet for subscription burn
    default_channel: str = ""           # provider name typically used (hint only)
    aliases: tuple[str, ...] = ()       # alternate IDs that resolve here
    # Can sustain the multi-turn agentic loop. Grants no tools; enforced at
    # loop start only. See the loop_capable() docstring below.
    loop_capable: bool = False


_DEFAULT_SPEC = ModelSpec(
    id="default",
    family="unknown",
    klass="other",
    display_name="Unknown model",
    context_window=128_000,
    max_output_tokens=8_192,
    relative_cost=1.0,
)

# Local (ollama) models default conservatively: 16384 context, not 128k. An
# unknown local model that claims a 128k window would let ollama silently
# truncate a large prompt to its 4096 default and answer from hallucination
# (agentic_loop.md §7 / live finding 4). Unknown local ids fall through here.
_LOCAL_DEFAULT_SPEC = ModelSpec(
    id="local-default",
    family="local",
    klass="other",
    display_name="Unknown local model",
    context_window=16_384,
    max_output_tokens=8_192,
    relative_cost=0.0,
    default_channel="ollama",
    loop_capable=False,
)


# Curated list — extend here, not in scattered files.
_MODELS: tuple[ModelSpec, ...] = (
    # ---- Anthropic / Claude ----
    # Mythos-class (above Opus tier). Fable 5 is the GA model; Mythos 5 is the
    # same underlying model gated to Project Glasswing. Both 1M context / 128K
    # output. relative_cost reflects pricing above Opus tier ($10/$50 per MTok
    # vs Opus $5/$25 → ~2x Opus → 10x Sonnet). Adjust if the economics change.
    ModelSpec(
        id="claude-fable-5", family="claude", klass="opus",
        display_name="Claude Fable 5",
        context_window=1_000_000, max_output_tokens=128_000,
        relative_cost=10.0, default_channel="claude_max",
        aliases=("fable", "fable-5"), loop_capable=True,
    ),
    ModelSpec(
        id="claude-mythos-5", family="claude", klass="opus",
        display_name="Claude Mythos 5",
        context_window=1_000_000, max_output_tokens=128_000,
        relative_cost=10.0, default_channel="claude_max",
        aliases=("mythos", "mythos-5"), loop_capable=True,
    ),
    ModelSpec(
        id="claude-opus-4-8", family="claude", klass="opus",
        display_name="Claude Opus 4.8",
        context_window=1_000_000, max_output_tokens=128_000,
        relative_cost=5.0, default_channel="claude_max",
        loop_capable=True,
    ),
    ModelSpec(
        id="claude-opus-4-7", family="claude", klass="opus",
        display_name="Claude Opus 4.7",
        context_window=1_000_000, max_output_tokens=128_000,
        relative_cost=5.0, default_channel="claude_max",
        loop_capable=True,
    ),
    # Sonnet 5 — next-gen sonnet-class. No 1M-context evidence in-repo, so it
    # takes the standard 200k window; cost structure copied from Sonnet 4.6
    # (sonnet baseline). Referenced widely across the test suite.
    ModelSpec(
        id="claude-sonnet-5", family="claude", klass="sonnet",
        display_name="Claude Sonnet 5",
        context_window=200_000, max_output_tokens=64_000,
        relative_cost=1.0, default_channel="claude_max",
    ),
    ModelSpec(
        id="claude-sonnet-4-7", family="claude", klass="sonnet",
        display_name="Claude Sonnet 4.7",
        context_window=1_000_000, max_output_tokens=64_000,
        relative_cost=1.0, default_channel="claude_max",
    ),
    ModelSpec(
        id="claude-sonnet-4-6", family="claude", klass="sonnet",
        display_name="Claude Sonnet 4.6",
        context_window=1_000_000, max_output_tokens=64_000,
        relative_cost=1.0, default_channel="claude_max",
    ),
    ModelSpec(
        id="claude-opus-4-6", family="claude", klass="opus",
        display_name="Claude Opus 4.6",
        context_window=1_000_000, max_output_tokens=128_000,
        relative_cost=5.0, default_channel="claude_max",
        loop_capable=True,
    ),
    ModelSpec(
        id="claude-haiku-4-5", family="claude", klass="haiku",
        display_name="Claude Haiku 4.5",
        context_window=200_000, max_output_tokens=8_192,
        relative_cost=0.2, default_channel="claude_max",
    ),
    ModelSpec(
        id="claude-sonnet-4", family="claude", klass="sonnet",
        display_name="Claude Sonnet 4",
        context_window=200_000, max_output_tokens=8_192,
        relative_cost=1.0, default_channel="claude_max",
    ),
    ModelSpec(
        id="claude-haiku-4", family="claude", klass="haiku",
        display_name="Claude Haiku 4",
        context_window=200_000, max_output_tokens=8_192,
        relative_cost=0.2, default_channel="claude_max",
    ),

    # ---- OpenAI ----
    ModelSpec(
        id="gpt-5.5", family="gpt", klass="other",
        display_name="GPT-5.5",
        context_window=400_000, max_output_tokens=128_000,
        relative_cost=1.0, default_channel="openai",
        loop_capable=True,
    ),
    ModelSpec(
        id="gpt-5", family="gpt", klass="other",
        display_name="GPT-5",
        context_window=400_000, max_output_tokens=128_000,
        relative_cost=1.0, default_channel="openai",
        loop_capable=True,
    ),
    ModelSpec(
        id="gpt-4.1", family="gpt", klass="other",
        display_name="GPT-4.1",
        context_window=1_000_000, max_output_tokens=32_000,
        relative_cost=0.5, default_channel="openai",
        loop_capable=True,
    ),
    ModelSpec(
        id="gpt-4o", family="gpt", klass="other",
        display_name="GPT-4o",
        context_window=128_000, max_output_tokens=16_384,
        relative_cost=0.5, default_channel="openai",
    ),
    ModelSpec(
        id="o3", family="gpt", klass="other",
        display_name="o3",
        context_window=200_000, max_output_tokens=100_000,
        relative_cost=2.0, default_channel="openai",
        loop_capable=True,
    ),
    ModelSpec(
        id="o4-mini", family="gpt", klass="other",
        display_name="o4-mini",
        context_window=200_000, max_output_tokens=65_536,
        relative_cost=0.3, default_channel="openai",
    ),
    ModelSpec(
        id="o1", family="gpt", klass="other",
        display_name="o1",
        context_window=200_000, max_output_tokens=32_768,
        relative_cost=2.0, default_channel="openai",
    ),

    # ---- DeepSeek ----
    ModelSpec(
        id="deepseek-reasoner", family="deepseek", klass="other",
        display_name="DeepSeek Reasoner",
        context_window=1_000_000, max_output_tokens=64_000,
        relative_cost=0.1, default_channel="deepseek",
        loop_capable=True,
    ),
    ModelSpec(
        id="deepseek-chat", family="deepseek", klass="other",
        display_name="DeepSeek Chat",
        context_window=1_000_000, max_output_tokens=8_192,
        relative_cost=0.1, default_channel="deepseek",
    ),
    ModelSpec(
        id="deepseek-v4-pro", family="deepseek", klass="other",
        display_name="DeepSeek V4 Pro",
        context_window=1_000_000, max_output_tokens=8_192,
        relative_cost=0.1, default_channel="deepseek",
        loop_capable=True,
    ),
    ModelSpec(
        id="deepseek-v4-flash", family="deepseek", klass="other",
        display_name="DeepSeek V4 Flash",
        context_window=1_000_000, max_output_tokens=8_192,
        relative_cost=0.05, default_channel="deepseek",
    ),

    # ---- Google Gemini ----
    ModelSpec(
        id="gemini-3-pro", family="gemini", klass="other",
        display_name="Gemini 3 Pro",
        context_window=1_000_000, max_output_tokens=64_000,
        relative_cost=1.0, default_channel="gemini",
        loop_capable=True,
    ),
    ModelSpec(
        id="gemini-3-flash", family="gemini", klass="other",
        display_name="Gemini 3 Flash",
        context_window=1_000_000, max_output_tokens=65_536,
        relative_cost=0.2, default_channel="gemini",
    ),
    ModelSpec(
        id="gemini-2.5-pro", family="gemini", klass="other",
        display_name="Gemini 2.5 Pro",
        context_window=2_000_000, max_output_tokens=65_536,
        relative_cost=1.0, default_channel="gemini",
        loop_capable=True,
    ),
    ModelSpec(
        id="gemini-2.5-flash", family="gemini", klass="other",
        display_name="Gemini 2.5 Flash",
        context_window=1_000_000, max_output_tokens=65_536,
        relative_cost=0.2, default_channel="gemini",
    ),

    # ---- Local (Ollama) ----
    # relative_cost=0 — local inference doesn't burn the shared subscription.
    # context 16384 keeps ollama from silently truncating to its 4096 default.
    # loop_capable stays False until §14's prerequisites are proven per
    # model (structured tool round-trip + num_ctx + smoke test). Do NOT flip on.
    ModelSpec(
        id="qwen3.5:4b", family="qwen", klass="other",
        display_name="Qwen3.5 4B",
        context_window=16_384, max_output_tokens=8_192,
        relative_cost=0.0, default_channel="ollama",
        loop_capable=False,
    ),
    ModelSpec(
        id="vibethinker-3b", family="vibethinker", klass="other",
        display_name="VibeThinker 3B",
        context_window=16_384, max_output_tokens=8_192,
        relative_cost=0.0, default_channel="ollama",
        loop_capable=False,
    ),
)


# Sort once at import — longest prefix first so resolve() can short-circuit.
_MODELS_BY_PREFIX_LEN: tuple[ModelSpec, ...] = tuple(
    sorted(_MODELS, key=lambda s: -len(s.id))
)
_MODELS_BY_ID: dict[str, ModelSpec] = {s.id: s for s in _MODELS}


def _looks_local(model_id: str) -> bool:
    """Heuristic: an ollama/local model id, e.g. ``qwen3:4b``, ``llama3.1:8b``,
    ``vibethinker-3b``. Ollama tags carry a ``:size`` suffix; cloud model ids
    never do. This lets unknown local models default to the conservative
    16384-token window instead of the cloud 128k default (agentic_loop.md §7)."""
    return ":" in model_id


def resolve(model_id: str) -> ModelSpec:
    """Return the ModelSpec matching ``model_id`` (longest-prefix), or default.

    Unknown models that look local (ollama ``name:tag`` form) fall through to
    ``_LOCAL_DEFAULT_SPEC`` (16384 context), not the 128k cloud default.
    """
    if not model_id:
        return _DEFAULT_SPEC
    lower = model_id.lower()
    # Exact match wins.
    if lower in _MODELS_BY_ID:
        return _MODELS_BY_ID[lower]
    # Alias match.
    for spec in _MODELS_BY_PREFIX_LEN:
        if spec.aliases and lower in (a.lower() for a in spec.aliases):
            return spec
    # Longest-prefix match.
    for spec in _MODELS_BY_PREFIX_LEN:
        if lower.startswith(spec.id.lower()):
            return spec
    if _looks_local(model_id):
        return _LOCAL_DEFAULT_SPEC
    return _DEFAULT_SPEC


def list_models() -> tuple[ModelSpec, ...]:
    """Return every curated ModelSpec. UI consumers can iterate this."""
    return _MODELS


def list_by_family(family: str) -> tuple[ModelSpec, ...]:
    """Return models in a family ('claude', 'gpt', 'gemini', 'deepseek')."""
    return tuple(s for s in _MODELS if s.family == family.lower())


def list_by_class(klass: str) -> tuple[ModelSpec, ...]:
    """Return models in a class ('haiku', 'sonnet', 'opus', 'other')."""
    return tuple(s for s in _MODELS if s.klass == klass.lower())


# ---------------------------------------------------------------------------
# Backward-compat shims — replaced gradually as call sites migrate.
# ---------------------------------------------------------------------------

def get_context_window(model_id: str) -> int:
    """Context window for a model. Unknown local models default to 16384,
    unknown cloud models to 128k (see ``resolve``)."""
    return resolve(model_id).context_window


def context_window(model_id: str) -> int:
    """Legacy alias for ``get_context_window``."""
    return resolve(model_id).context_window


def loop_capable(model_id: str) -> bool:
    """Whether a model can sustain the multi-turn agentic loop.

    Grants no tools; what it gates at runtime is whether ``send_orchestrate``
    may start on this model at all — see
    ``providers/bridge.py:_model_is_loop_capable``, the one enforcement site.
    Sub-tier models are refused because they wedge rather than fail: a
    haiku-class model hot-spun the bridge at 100% CPU with zero output for 20+
    minutes (agentic_loop.md finding 7).

    Derived per-model from the spec flag; unknown models are treated as not
    capable."""
    return resolve(model_id).loop_capable


def max_output_tokens(model_id: str) -> int:
    """Legacy: prefer ``resolve(model_id).max_output_tokens``."""
    return resolve(model_id).max_output_tokens


def model_class(model_id: str) -> str:
    """Return the class bucket for a model (replaces classify_model)."""
    return resolve(model_id).klass


def relative_cost(model_id: str) -> float:
    """Subscription cost weight relative to Sonnet (1.0)."""
    return resolve(model_id).relative_cost
