"""Worker-side provider reconstruction from a run manifest (Phase 4).

The daemon does NOT ship the live provider *object* to the worker — provider
clients aren't serializable and, more importantly, the whole point of isolation
is that the worker builds its own. What crosses the pipe is a small
``provider_config`` dict (see ``ManifestProviderConfig`` below); the worker
rebuilds an equivalent API provider LOCALLY in its own process.

SCOPE (P4): only the API providers go through the worker —
``AnthropicProvider`` and ``OpenAICompatibleProvider`` (which also covers
Gemini/OpenAI/DeepSeek/custom OpenAI-compatible endpoints via base_url). The
bridge providers (claude_max / codex_max) and the rpb/substrate composites stay
IN-PROCESS even under the flag: the daemon must not route them here. This module
refuses anything it doesn't know how to build.

SECRETS
-------
The provider needs an API key to reach the endpoint. The worker's *environment*
is scrubbed of secrets (see worker_manager._scrub_env), so the key is carried in
``provider_config["api_key"]`` on the manifest frame instead. This is the P4
seam: the field is deliberately isolated in one place so the vault track (a
separate story) can later replace the inline key with an RPC-mediated,
PID-gated fetch WITHOUT touching the loop. Until then the key travels on the
(local, private, non-pickled) pipe — never on argv/env.
"""
from __future__ import annotations

import logging
from typing import Any

from ..providers.base import Provider

log = logging.getLogger(__name__)


# Provider names that MUST stay in-process (never built worker-side in P4).
# Kept explicit so a daemon-side routing bug surfaces as a clean refusal here
# rather than a subtle wrong-provider run.
_IN_PROCESS_ONLY = frozenset({
    "claude_max", "codex_max", "rpb", "substrate", "bridge",
})

# API provider kinds the worker CAN rebuild locally.
_KIND_ANTHROPIC = "anthropic"
_KIND_OPENAI_COMPAT = "openai_compat"


class UnsupportedProviderError(RuntimeError):
    """The manifest asked for a provider the worker must not build (bridge,
    rpb, substrate, or an unknown kind). The daemon should have kept these
    in-process; refusing here is a guard, not a normal path."""


def build_provider_from_manifest(cfg: dict[str, Any]) -> Provider:
    """Rebuild an API provider LOCALLY from the manifest ``provider_config``.

    ``cfg`` shape (see agent_worker manifest schema):
        {
          "kind": "anthropic" | "openai_compat",
          "name": <str>,              # the provider's budget/usage key
          "default_model": <str>,
          "base_url": <str>,          # "" for anthropic default endpoint
          "api_key": <str>,           # see SECRETS note above
        }

    Returns a fully-constructed Provider whose ``send_orchestrate`` runs the
    generic base-class loop (the same loop the in-process path uses for these
    providers). Raises ``UnsupportedProviderError`` for anything that must stay
    in-process.
    """
    kind = str(cfg.get("kind", "")).strip()
    name = str(cfg.get("name", "")).strip()

    if kind in _IN_PROCESS_ONLY or name in _IN_PROCESS_ONLY:
        raise UnsupportedProviderError(
            f"provider {name or kind!r} must run in-process; the daemon should "
            f"not have routed it to a worker (bridge/rpb/substrate are excluded "
            f"from isolation in P4)"
        )

    default_model = str(cfg.get("default_model", "") or "")
    base_url = str(cfg.get("base_url", "") or "")
    api_key = str(cfg.get("api_key", "") or "")

    if kind == _KIND_ANTHROPIC:
        from ..providers.anthropic import AnthropicProvider
        prov = AnthropicProvider(
            api_key=api_key, default_model=default_model, base_url=base_url,
        )
        return prov

    if kind == _KIND_OPENAI_COMPAT:
        from ..providers.openai_compat import OpenAICompatibleProvider
        prov = OpenAICompatibleProvider(
            name=name or "openai_compat",
            base_url=base_url,
            api_key=api_key,
            default_model=default_model,
        )
        return prov

    raise UnsupportedProviderError(
        f"unknown worker provider kind: {kind!r} (name={name!r})"
    )


def classify_manifest_provider(provider: Provider) -> dict[str, str]:
    """Daemon-side helper: derive the ``provider_config`` ``kind``/``name`` for a
    live provider instance so the manifest can be built.

    Lives here (not on the daemon) so the "which providers are worker-eligible"
    knowledge sits next to ``build_provider_from_manifest``. Returns a dict with
    at least ``kind`` and ``name``; raises ``UnsupportedProviderError`` if the
    provider is one that must stay in-process — the daemon uses that as the
    signal to take the in-process path even under the flag.
    """
    from ..providers.anthropic import AnthropicProvider
    from ..providers.openai_compat import OpenAICompatibleProvider

    if isinstance(provider, AnthropicProvider):
        return {"kind": _KIND_ANTHROPIC, "name": provider.name}
    if isinstance(provider, OpenAICompatibleProvider):
        return {"kind": _KIND_OPENAI_COMPAT, "name": provider.name}
    raise UnsupportedProviderError(
        f"provider {type(provider).__name__} is not worker-eligible in P4"
    )


__all__ = [
    "build_provider_from_manifest",
    "classify_manifest_provider",
    "UnsupportedProviderError",
]
