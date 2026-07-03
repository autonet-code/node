# Generic agentic loop: hardening spec (2026-07-04)

Binding spec for upgrading `atn/providers/base.py:send_orchestrate` (the
loop behind anthropic / openai_compat / ollama agents) and its adapters
to parity with current harnesses (Claude Code / Agent SDK, opencode,
Codex CLI). Grounded in: the 2026-07-04 live daemon session (findings
1–7 below), the loop audit, and the SOTA survey. The bridge (Claude
Agent SDK) keeps its own loop; only §10–11 touch it.

Live findings this spec answers:
1. Unknown model strings silently route to BridgeProvider
   (`provider_manager.py:483`) — a `qwen3.5:4b` agent burned 45k
   subscription tokens on the bridge.
2. `create_agent` has no `provider` field; `update_agent` provider
   changes are inert (cached instance never evicted,
   `execution_engine.py:847`).
3. InputArbiter mic can be held by a dead WS surface forever; ghosts
   stay in `_connected` and recapture the mic on every hand-off.
   All input surfaces wedge with `input_not_active`.
4. Ollama requests never set `num_ctx`; ollama silently truncated a
   10,768-token request to 4096 (`server.log`) and the agent answered
   from hallucination. Non-bridge agents also get the full
   `_SHELL_TOOLS` schema block unconditionally (`execution_engine.py:909`).
5. Ollama multi-turn tool history is flattened to text
   (`_normalize_content`) — the structured `tool_calls` / `role:"tool"`
   linkage is lost after turn 1.
6. A transient stream drop mid-loop aborts the whole orchestration
   (no loop-level retry); compaction failure silently no-ops into a
   guaranteed overflow.
7. Bridge + `claude-haiku-4-5` main-loop model = silent hot-spin
   (100% core, zero events, 20+ min); `kill_agent` reported success
   but left the spinning bun subprocess alive.
8. `delegate_message` mid-turn injection races turn completion: a
   message written to bridge stdin as the orchestration ends is
   accepted ("User message injected") and silently dropped — the
   daemon reports `injected` with no consumption ack and no inbox
   fallback. (Observed live: injection at 01:32:45.685, execution
   completed 01:32:45.)

## Design principles

- **Error classes are routes, not retries.** Transient → bounded
  resample. Overflow → context reduction, never retry. Fatal (auth,
  invalid request) → structured abort. (opencode `retry.ts` split.)
- **Two-tier context reduction: prune before compact.** Dropping old
  tool bodies is free; an LLM summary costs tokens and can fail.
- **History must always be replayable.** No orphaned `tool_use`
  without a `tool_result` on any exit path.
- **Steering parity.** `delegate_message` must land mid-turn on the
  generic loop, not just the bridge.
- **Fail loud on routing.** A model the daemon can't place is an
  error, not a silent fallback onto the user's subscription.

## §1 Loop-level retry gate (base.py)

Wrap the per-turn `send_stream` call:
- Classify: `overflow` (provider 400s whose message matches context /
  token-limit patterns; adapters map their provider's overflow shape
  to a shared `ContextOverflowError`), `transient` (timeouts, dropped
  streams, 5xx/429 that escaped adapter retries), `fatal` (everything
  else).
- Transient: up to 3 loop-level retries, exp backoff 2s/8s/30s +
  jitter; honor `Retry-After` when surfaced. Retry re-sends the same
  request (history unchanged).
- Overflow: jump to §2 reduction, then re-send once. Overflow after
  reduction → abort `context_overflow`.
- Fatal: abort with `stop_reason="provider_error"`, error text in the
  result, orphan repair per §3.

## §2 Context reduction: prune, then compact

Trigger **pre-send**: estimated next-request tokens (chars/4 estimate,
calibrated by the last turn's actual `input_tokens` when available) >
`ctx_window − max_tokens − 8k buffer`. `ctx_window` from
`model_specs.get_context_window` (see §7).

Tier 1 — prune (no LLM call): replace `tool_result` bodies older than
the last 2 assistant turns with `[pruned: <n> chars, tool=<name>]`,
protecting the most recent 40k chars of tool output. Only prune when
it saves ≥ 20k estimated chars.

Tier 2 — compact (only if still over): summarize everything except
(a) the original task message, (b) the last 2 turns, (c) the last
~20k chars of user messages, using the structured template: Goal /
Progress / Key decisions / Critical context / Next steps.
- On summarizer failure: **fallback hard truncation** — drop oldest
  turns (keeping (a)–(c)), never no-op.
- **Spiral guard**: max 2 compactions per run; if a compaction reduces
  the estimate by <20%, abort `context_overflow` instead of looping.
  (Codex v0.112 / opencode #27924 class of bug.)

## §3 Orphan repair on every exit path

Any return/abort that leaves a trailing assistant message containing
`tool_use` blocks without matching results (budget_exceeded,
per_turn_input_exceeded, repeat abort, interrupt, provider_error)
appends synthetic `tool_result` blocks
(`{"cancelled": true, "reason": <abort reason>}`, `is_error=True`)
before returning, so persisted history replays cleanly against
Anthropic-shape APIs.

## §4 Loop detection

- Exact-repeat threshold drops 5 → 3 (opencode `DOOM_LOOP_THRESHOLD`).
- Oscillation guard: over the last 6 calls, if ≤ 2 distinct
  (tool, args-hash) fingerprints and every fingerprint appeared ≥ 2
  times → abort `loop_detected`.
- Before aborting, inject one warning turn: a user message telling the
  model it appears to be looping and must change approach or conclude;
  abort only if the next call repeats again. (The model gets one
  chance to self-correct; my own harness does the same.)

## §5 Mid-turn steering for the generic loop

- `Provider` base gains `send_user_message(content) -> bool` backed by
  an `asyncio.Queue` (`_steering_queue`). Returns True if an
  orchestration is active to consume it.
- `send_orchestrate` drains the queue at each iteration boundary
  (after tool results are appended, before the next send) and appends
  each item as a `user` message.
- `execution_control.send_delegate_message` already prefers
  `provider.send_user_message` — with this, API-provider agents get
  live injection instead of the inbox fallback.
- **Delivery ack (finding 8, applies to bridge too):** injection is
  only "delivered" when consumed. Bridge: `claude-bridge.ts` emits
  `@@EVENT@@ user_message_consumed` when the SDK loop actually reads
  the injected message; `bridge.py` tracks pending injections and, on
  orchestration end with unconsumed injections, re-posts them to the
  agent's inbox (HIGH WORK) so the next run picks them up. Generic
  loop: same contract — steering-queue items not drained by run end
  are re-posted to the inbox. `delegate_message` result distinguishes
  `injected` (consumed or pending with fallback guarantee) so callers
  are never lied to.

## §6 Tool-result truncation: head + tail

Replace the tail-chop at `_TOOL_RESULT_CHAR_MAX` (40k) with head+tail:
first 24k + `\n…[elided <n> chars of <total>]…\n` + last 12k. Keeps
the command echo and the error tail (Codex shape). Cap unchanged.

## §7 Model metadata unification

- `get_context_window` / output caps come from `model_specs.py`
  everywhere; `base.py:CONTEXT_WINDOWS` becomes a thin re-export or is
  deleted. Per-request `max_tokens = min(16384, spec.max_output_tokens)`
  instead of the hardcoded 16384.
- Register local models in `model_specs.py`: `qwen3.5:4b`,
  `vibethinker-3b` (context 16384 default, `relative_cost=0`,
  tier: tool-use). Unknown ollama models default to 16384, not 128k.

## §8 Malformed tool arguments

When an adapter degrades unparseable tool-call args to `{"raw": ...}`,
the loop must NOT execute the tool. Return a synthetic error
`tool_result` ("arguments were not valid JSON — re-emit the call")
and continue the loop. Applies to all three adapters' parse paths.

## §9 Ollama adapter: structured history + num_ctx

- Translate canonical history natively: assistant `tool_use` blocks →
  ollama `message.tool_calls`; canonical `tool_result` blocks →
  `role:"tool"` messages (with `tool_name`). `_normalize_content`
  remains only for plain-text blocks. Round-trip test required:
  2-turn tool conversation re-sent to ollama preserves the linkage.
- Every request sets `options.num_ctx` = `get_context_window(model)`
  (per §7; 16384 default for local models). Never rely on ollama's
  4096 default.
- Lean tool surface: `execution_engine.py:909` stops appending
  `_SHELL_TOOLS` unconditionally for non-bridge providers; shell tools
  ride the existing `tools` categories (add a `"shell"` category) so
  a lean agent's prompt stays small.

## §10 Provider routing + lifecycle

- `_resolve_provider_for_model`: unknown prefix → probe the ollama
  tags cache; if the model is installed locally → OllamaProvider.
  Otherwise raise `ProviderError("unknown model '<m>': set provider
  explicitly")`. **No silent BridgeProvider fallback.** (Explicit
  `provider: claude_max` config keeps working; gemini/openai keyless
  fallbacks at provider_manager.py:459/471 also become errors.)
- `create_agent` tool schema gains optional `provider` (enum of known
  providers); plumbed into `AgentDefinition.provider`.
- `update_agent`: when `provider` or `model` changes, evict + close
  `_active_providers[agent_id]` (same cleanup as unregister_agent,
  minus file deletion) so the change is live.

## §11 Bridge: spin guard + kill ladder

- Root-cause the haiku hot-spin in `bridge/claude-bridge.ts` (SDK
  `query()` with a non-orchestrator model). Whatever the mechanism:
  errors from `query()` must surface as an `@@EVENT@@ error` +
  request failure, never a bare loop continue.
- First-event watchdog: if a sent orchestrate request produces zero
  bridge events within 120s (configurable `ATN_BRIDGE_FIRST_EVENT_TIMEOUT`),
  kill the subprocess and fail the request — don't wait for the 1800s
  idle ceiling.
- Kill ladder: `kill_agent` → provider interrupt must escalate to
  subprocess termination (bun PID) within a 5s grace window, verified
  by the daemon (poll PID, `taskkill /F` fallback). A "killed"
  execution must never leave the provider subprocess running.
- Model-tier guard: reject orchestrate requests on models whose spec
  says non-orchestrator-capable (haiku) with a clear error before
  spawning the SDK loop, since the SDK loop is known to misbehave.

## §12 InputArbiter liveness

- Ghost holder fix: `WebSocketBridge` validates the holder on every
  gate denial — if the holder token is a `ws:` surface with no live
  session, call `release_for(holder)` and re-run the gate. Cheap,
  contained, no arbiter API change.
- Hand-off successor selection skips `ws:` tokens with no live
  session (ws_server passes a liveness predicate to the arbiter at
  construction).
- `is_active` auto-acquire keeps `setdefault` but the predicate
  prevents dead tokens from re-entering via hand-off.

## §13 Usage self-awareness for agents (owner request 2026-07-04)

- New tool `get_usage` (category `budget`): returns the caller's
  provider usage — for bridge agents, the cached
  `refresh_usage` payload (5h/7d utilization + reset times) plus the
  agent's own budget consumption (`registry` budget counters); for API
  providers, cumulative session tokens vs configured budget. Cached
  ≤60s so agents polling it don't burn a probe call per turn.
- Purpose: agents can self-pace long-horizon work against the shared
  subscription the way the operator does, instead of running blind
  until a budget abort.

## §14 Local orchestrator eligibility (owner request 2026-07-04)

- The "no cloud available" contingency: ollama models may serve as the
  orchestrator/root cognitive agent. Gate, don't hardcode: a model spec
  flag `orchestrator_capable` that local models can earn; the
  provider-level `orchestrator_capable: False` on ollama
  (`provider_manager.py:216`) becomes per-model.
- Prerequisites before flipping any local model on: §9 structured tool
  history (multi-turn tool use must round-trip), §7 num_ctx, and a
  smoke test (spawn child → collect → summarize) passing on that model.
- Fallback chains (`provider: [claude_max, ..., ollama]`) already
  exist; with this flag the chain can legally end in ollama so the
  fleet degrades to local instead of halting when cloud auth/quota is
  gone.

## Testing (targeted only — never the full suite)

New/updated files: `tests/atn/test_loop_hardening.py` (§1–§6, §8:
mock provider raising transient/overflow/fatal; compaction fallback;
orphan repair; oscillation; steering queue), extend
`tests/atn/test_provider_base.py` (head+tail truncation, max_tokens
from spec), `tests/atn/test_ollama_translation.py` (§9 round-trip,
num_ctx presence), `tests/atn/test_provider_routing.py` (§10 fail-loud
+ eviction), `tests/atn/test_input_arbiter.py` (§12 ghost recovery).
Run alongside existing `test_provider_base.py`,
`test_long_horizon_safeguards.py`, `test_budget_inner_loop.py`.

Out of scope: worker-isolation default, voice/chat surfaces, sponsor
paths, any consensus/substrate code.
