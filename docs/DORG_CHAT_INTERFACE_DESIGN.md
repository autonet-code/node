# dОrg Deployment & Chat-Interface Layer: Design

Status: **design, mostly not yet built** (2026-06-01). Captures the reasoned
architecture from an extended design conversation so it survives compaction and
a fresh session can build from it. Marks clearly what's built vs designed vs
directional.

---

## 1. The strategic frame (why this exists)

dОrg is a long-standing Web3 builder DAO, mid identity-crisis as client work
dries up industry-wide (not a Web3-specific problem: AI is eating commodity
dev everywhere). The bet: don't pivot to "general dev shop"; **leverage dОrg's
DAO legitimacy + rep-weighted human governance to tackle the industry-wide
unsolved problem of governance across BOTH humans and AI**, i.e. distributed
human input into AI operations. Build it, test it in-house, then do outreach.

**North star:** The dОrg deployment of autonet should differ from standard
single-user autonet by **exactly one thing: distributed input through chat
(Discord first)**. Everything else is stock autonet. Chat interfaces
(Discord/Telegram/Matrix/Signal) are conceived as a *general* capability,
useful to single users too, not a dОrg special-case. But "general" ≠ "in
autonet core": it's built as an external client (see §2); absorption into
autonet is a later, deliberate choice, not assumed.

**Crucial unknown (do not over-commit):** dОrg doesn't yet know what its "AI
product" is: it could be the shared governance/control interface, or it could be
"just an API that member-operated agents report work into." The architecture
must make BOTH possible and let which-one-matters emerge. Don't bet the design
on the governance interface being the product.

---

## 2. Layering

**Where it lives (decided 2026-06-01): build the chat interface as an EXTERNAL
CLIENT of autonet by default, NOT a change to autonet core.** It is exactly
what `fleet_bot` already is: a process that connects to the daemon's ws_server,
subscribes to events (read-only), and calls existing methods (`send_agent_message`).
This means **zero autonet-core change and zero atn_web risk.** Absorbing the
ChatService *into* autonet (as a general single-user feature, VoiceService-style)
is a DELIBERATE LATER CHOICE, done only if/when we want to ship it as a framework
feature, and then with atn_web regression-tested. Do not change autonet's core
(agent/session model, event shapes, ws protocol): those are what atn_web depends
on. (Connectors are outbound-tools-only; NOT the home for chat.)

1. **autonet core**: the agent framework, unchanged. Orchestrator + delegates,
   one (linear) session per agent, EventBus, on-chain registration, jurisdictions.
   **Untouched.** atn_web depends on it.
2. **Chat-interface layer (`ChatService`)**: platform-neutral. An EXTERNAL
   client process (like fleet_bot): consumes the EventBus + ws_server API.
   VoiceService is the *architectural template* (subscribe-to-events for output,
   call-existing-method for input), but it lives OUTSIDE autonet, not as a core
   subsystem, unless/until deliberately promoted. An autonet agent can have a
   conversation that lives on a chat platform. Input from platform →
   agent session; agent output → platform. Single-user (1 channel ↔ 1 agent) and
   multi-user both ride this.
3. **Platform adapters**: Discord first, then Telegram/Matrix/Signal. Implement
   the `MessagingClient` protocol. They live in the external client process (the
   hackathon repo / wherever the client runs), NOT in autonet, so autonet never
   grows discord.py et al. as deps.
4. **Policy / distributed-input layer (the dОrg delta)**: lives at the
   ChatService **input seam** (where a platform message becomes
   `send_agent_message`). Access control, credits, whitelist, AND eventual
   rep-weighted distributed input ALL live here. This is the only dОrg-specific
   layer. autonet core + the orchestrator model stay pristine.

---

## 3. Architectural template: mirror the VoiceService

Exploration (see file:line refs below) found the **VoiceService is the exact
existing analog** for "external real-time surface bound to agent sessions." The
`ChatService` mirrors it:

| Aspect | VoiceService (existing) | ChatService (to build) |
|---|---|---|
| Lifecycle | singleton, async start/stop | same |
| Input → agent | PTT→Whisper→`runtime.send_agent_message()` | platform msg →`send_agent_message()` |
| Output → surface | subscribes EventBus `STEP_OUTPUT` etc → TTS | subscribes EventBus → platform send |
| Routing | "focus" (`voice_focus`/`tools_focus`) | channel/thread→agent routing table |
| Platform code | optional audio backends | optional adapters (`MessagingClient`) |

Key refs: `atn/voice_service.py:800-987` (lifecycle), `:1414-1435`
(`_send_to_agent` input template), `:1241-1337` (EventBus output handlers),
`:1023-1049` (focus). Input entry point: `atn/runtime/session_manager.py:101-144`
(`send_agent_message`, which handles mid-session injection vs inbox automatically).
Output: EventBus `STEP_OUTPUT` channel="text"/"thinking"/"tool_call",
`EXECUTION_STARTED/COMPLETED/FAILED` (`atn/events.py:20-87`).

**Decisive constraint found:** autonet is **strictly one linear session per
agent** (`atn/conversation.py`; reset is destructive, no branching). So the
native grain is **one agent ↔ one conversation**. Don't fight it: a Discord
thread = one agent. (Not "one agent, many threads.")

---

## 4. The interaction model (in-house agents)

Render autonet's own orchestrator+delegates model directly into Discord, in
**one dedicated channel** (NOT guild-wide: containment avoids the mess):

- **Channel = the orchestrator's conversation.** You chat with the orchestrator
  in the channel, exactly like autonet's main chat. ChatService binds the
  channel to the orchestrator agent.
- **Thread per top-level delegate.** When the orchestrator calls `delegate` /
  `create_agent`, a `delegate.spawned` event fires; the **ChatService reacts to
  that event by creating a Discord thread** for the new agent and binding
  thread↔agent. Thread = that delegate's full conversation. (Thread-creation
  logic is NOT bespoke: it falls out of the agent hierarchy + event stream.
  The orchestrator owns "when to spawn"; ChatService owns "render spawn as
  thread".)
- **Nested sub-agents (depth ≥ 2):** Discord threads are one level deep only.
  So a sub-delegate (`orch.1.2`, and any deeper descendant) does NOT get its own
  thread: it renders as a **distinct pinned embedded message inside its
  top-level delegate's thread**, styled differently: a rolling ~200-word window
  of its latest thoughts/conversation, edited in place. All transitive
  descendants of a top-level delegate render as pinned tiles in that delegate's
  thread, lineage shown in the label (`orch.1.2.3`).

### Addressing / routing
- **Channel (unprefixed)** → orchestrator.
- **In a thread (unprefixed)** → that thread's delegate.
- **Reply to a pinned sub-agent tile** → routes to that sub-agent. Pins serve
  double duty: easy-to-find AND the addressing handle. Uses `reply_to_id`
  (already in the `messaging` `Message` type). Preferred over a prefix syntax.
- (Optional fallback: an `@id`/prefix syntax.)

### Mechanics to handle (noted, not blockers)
- Rolling-window pinned tile = edit-in-place (reuse the `AgentExecutionEmbed`
  pattern from `hackathon/fleet_view.py`, body policy = rolling tail, pinned).
- Discord pin cap = 50/thread → unpin completed sub-agents, keep active ones.
- Adapter must surface `reply_to_id` (the protocol type already has it).

---

## 5. Member / external agents (the OTHER half: do not drop)

Member agents are not a side feature; they're the second half of the same
thesis. A **spectrum of integration depth**, build shallow→deep:

1. **Report-in (shallow, partly built):** member runs *whatever* agent
   framework they like, on their own machine, and **calls the dОrg API to report
   work**. Essential for sales AND for work that must run on the member's own
   machine (client projects) where the agent can't live on dОrg's daemon. The
   hackathon API is the simple version (token + 3 MCP tools: claim_lead,
   surface_lead, send_message).
2. **Open-ended conversational registration (designed, not built):** instead of
   the fixed 3-tool menu, registration is a **conversation with Kevin**: Kevin
   asks the owner the agent's name, "what type of activity will your agent
   conduct?", writes a short description, and provisions an appropriate endpoint
   set. Key realization: there's not much diversity in endpoints, they're
   "functions you call with payloads", so this is "make the toolset
   configurable per agent," not "build N department modules."
3. **Member-run daemons + on-chain + jurisdiction (directional, endgame):**
   members run their *own* autonet daemon, register agents **on-chain** (autonet
   already supports this via `register_agent_on_chain`), and **join the dОrg
   jurisdiction** (autonet has jurisdictions). The deep end of distributed,
   member-operated, on-chain-governed agents. The report-in API is the on-ramp
   to this.

The report-in relationship and the jurisdiction endgame are the **same
relationship at different depths**. Build the shallow end; keep the deep end as
the direction.

---

## 6. What's built vs designed vs directional

**Built (local, working):**
- Daemon fleet → Discord per-execution threads (read-only): `hackathon/fleet_view.py`, `fleet_bot.py`.
- Operator-gated control: `>>`/@mention in a channel relays to the orchestrator.
- Read-only dashboard (`hackathon-app`): both agent populations, physics cards,
  agent windows (config), search/filter/sort/hierarchy, K3V|N FAB (orchestrator
  presented as K3V|N), agent_type icons, daemon pipe (api.py `/daemon/agents` +
  `/daemon/stream` SSE), session-stats enrichment (turns/context).
- `kevin-support` pipeline agent polling `/support/stats` (support usage:
  per-thread/per-user tokens), surfaced via get_output through the pipe.
  EXCLUDED from thread rendering + status-flicker.
- `messaging/protocol.py` (hackathon repo): Author/Channel/Message/EmbedSpec/
  `MessagingClient` protocol/`StatusEmbedHandle`: **Phase-1 types only, no
  adapter**. This is the contract for the chat adapter.

**Designed here, NOT built:**
- `ChatService` (the layer-2 abstraction, VoiceService-shaped).
- Discord adapter implementing `MessagingClient`.
- Channel=orchestrator / thread=delegate / pinned-tile=sub-agent rendering.
- Reply-based addressing.
- Open-ended conversational member-agent registration.

**Directional (future):**
- Distributed input (rep-weighted, multi-user) at the policy seam.
- Member-run daemons, on-chain agent registration, dОrg jurisdiction.
- Telegram/Matrix/Signal adapters.

**Superseded / dissolving:**
- `general_support.py` (hackathon) is a bespoke Discord-agent runtime that
  duplicates what the ChatService (external client) should provide generically.
  Direction: it dissolves into the ChatService; use it as the working reference
  spec (per-thread sessions, tier/credit gating → the latter moves to the policy
  seam). Support can stay DOWN meanwhile (hackathon is over; no demo pressure).
- The "represent support threads as dashboard cards via a stats side-channel"
  plan was dropped: once each support thread is backed by its own autonet agent
  (the ChatService spawns one agent per thread via the existing daemon API),
  they're fleet members for free; no side-channel needed. (This is the client
  USING autonet's existing agent-spawn API, not a change to autonet.)

---

## 7. First build (when we resume)

Not distributed input, not even Discord first. The foundation:
**a `ChatService` + `MessagingClient` skeleton as an EXTERNAL client (own
process, like fleet_bot, NOT in autonet), modeled on the VoiceService pattern,
proving the agent-per-conversation binding end-to-end against a stub/echo
adapter via the existing ws_server API.** Then: Discord adapter → channel/thread
rendering → reply addressing → (separately) conversational member registration →
(later) policy seam / distributed input. Autonet stays untouched throughout;
absorbing ChatService into autonet is a separate, deliberate, atn_web-tested
decision for later (if ever).

No demo deadline pressure: build the right foundation, not the stopgap.
Workflows (parallel subagents) are NOT for the design/capture step, but may help
later for parallel exploration/implementation once building, opt-in only.
