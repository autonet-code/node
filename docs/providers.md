# Inference providers

Status: BUILT (beta). Provider registration, credential storage, and model
routing live in `atn/runtime/provider_manager.py`; the adapters are in
`atn/providers/`. The **AI Input** tab in the app is the owner-facing surface
over them.

This is the doc for the **owner** deciding where their agents' inference
comes from. It covers the built-in providers, what "Add Provider" actually
does, and the two constraints that most often surprise people
(orchestrator-capability and model routing).

## What a provider is

A provider is a place the daemon can send a prompt and get tokens back. Each
agent picks one; different agents on the same daemon can use different
providers and different models simultaneously.

There are three authentication shapes:

| Shape | How it authenticates | Examples |
|-------|---------------------|----------|
| **bridge** | drives a CLI you are already logged into, using your existing subscription | Claude Max, Codex |
| **api_key** | a key you paste, stored in the daemon's credential store | Anthropic, OpenAI, Gemini, DeepSeek |
| **local** | nothing to authenticate; it runs on your machine | Ollama |

Bridges are the cheapest path if you already pay for a subscription: they
spend your existing plan rather than metered API credit. API-key providers
bill per token. Local providers cost nothing but your own hardware.

## Built-in providers

| Provider | Auth |
|----------|------|
| Claude Max | bridge |
| Codex (OpenAI) | bridge |
| Anthropic API | api key |
| OpenAI | api key |
| Google Gemini | api key |
| DeepSeek | api key |
| Ollama (local) | local |
| RPB Network | peer-to-peer |
| World-Model Substrate | local |

## Loop capability is a property of the MODEL

Every agent runs the same multi-turn tool-calling loop, and small/fast models
fail at it in an expensive way: a haiku-class model on the Claude Code bridge
does not error, it hot-spins at 100% CPU producing nothing for as long as you
let it run. So the daemon refuses to *start* the loop on a model below the
bar, with a clear error, rather than letting it wedge.

This is a reliability floor every agent needs, not a rank, and not a
privilege one agent holds. It is checked per model, so it follows the model
you pick, not the provider you picked it from. A capable provider serving a
haiku-class model is still refused.

For that reason the provider cards carry **no capability badge**. Nearly
every provider offers both low-tier and frontier models, so any
provider-level summary either says the same thing about every card or
implies a floor the provider does not actually hold. Ollama is the case to
watch: whether a given local model holds up in a tool-calling loop varies a
lot by model, so test before handing it real work.

## Adding a custom provider

"Add Provider" registers any **OpenAI-compatible** endpoint: vLLM, LM
Studio, OpenRouter, a self-hosted gateway, or any service exposing
`/v1/chat/completions`.

| Field | Notes |
|-------|-------|
| **Provider ID** | short slug, e.g. `my-llm`. Cannot collide with a built-in id. |
| **Display Name** | what you see in the picker. |
| **Base URL** | e.g. `https://api.example.com/v1`. |
| **API Key** | optional; omit for an unauthenticated local endpoint. |
| **Default Model** | optional; the model used when an agent does not name one. |

**The daemon validates before saving.** It derives a `/models` URL from your
base URL and probes it. You get a specific failure rather than a provider
that silently never works:

- unreachable host → *"Cannot connect to …, check the URL"*
- slow endpoint → *"Timeout connecting to …"*
- key rejected (401/403) → *"API key rejected by … "*

A probe that fails for any *other* reason is tolerated: some
OpenAI-compatible servers do not implement `/models` at all, and refusing
those would be wrong.

The API key is stored in the daemon's credential store under
`provider_<id>`, not in `config.yaml`. Removing the provider deletes the
stored key with it.

Custom providers register live: the running daemon picks the provider up
immediately, without a restart.

## Model routing

Two things are worth knowing, because both have bitten users:

**An unknown model string does not fail loudly.** Historically it fell
through to the bridge, which meant a typo'd local model name could quietly
spend subscription tokens. Prefer picking a model from the dropdown over
typing one.

**The model picker lists only providers the daemon has actually
registered.** If a provider you configured is missing from the list, the
daemon did not manage to register it at boot: check the daemon log rather
than re-adding it.

Ollama needs one extra consideration: its context window defaults low, and a
request exceeding it is **silently truncated** rather than rejected, which
surfaces as an agent confidently answering from a half-read prompt. The
daemon sets the context size explicitly; if you run Ollama yourself, set it
deliberately too.

## Cost control

Provider choice is the largest lever on what a fleet costs to run. Budgets
are enforced per agent; see the credit budget controls on the agent itself.
Bridges spend subscription capacity, so a runaway loop there costs you rate
limits rather than money; API-key providers spend real credit, so set a
budget before letting an agent run unattended.

## See also

- `docs/agentic_loop.md`: the internals, the loop behind
  anthropic/openai_compat/ollama agents and its hardening spec
- `docs/unified_agent_design.md`: how an agent picks and inherits a provider
- `docs/two_plane_inference.md`: substrate retrieval paired with an LLM
