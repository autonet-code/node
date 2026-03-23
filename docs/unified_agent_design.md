# Unified Agent Architecture — Definitive Design

**Author:** Architecture review delegate (Opus 4.6)
**Date:** 2026-03-24
**Status:** Design proposal — ready for implementation planning

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Real Problems](#the-real-problems)
3. [Unified AgentDefinition Schema](#unified-agentdefinition-schema)
4. [Agent Modes: Pipeline vs Cognitive](#agent-modes)
5. [Lifecycle States and Transitions](#lifecycle-states)
6. [The Intrinsic Heartbeat](#intrinsic-heartbeat)
7. [Hierarchy and Supervision](#hierarchy-and-supervision)
8. [Communication Patterns](#communication-patterns)
9. [Tool Surface Per Agent Mode](#tool-surface)
10. [Persistence Model](#persistence-model)
11. [The Orchestrator in the Unified Model](#orchestrator)
12. [Token Budgets and Autonomous Agents](#token-budgets)
13. [Migration Strategy](#migration-strategy)
14. [What We Are NOT Changing](#not-changing)
15. [Open Questions](#open-questions)

---

## 1. Executive Summary <a id="executive-summary"></a>

ATN currently has two parallel sub-agent systems that share no data model:

- **Persistent agents** (`AgentDefinition` + step pipeline): YAML-defined, scheduled, inbox-triggered, output-stored. Execution is a deterministic sequence of typed steps. They survive restart. Created via `create_agent`.

- **Delegates** (`DelegateNode` + `BridgeProvider`): Ephemeral Claude SDK sessions. Each gets a fresh subprocess, works autonomously with file/bash/web tools, returns a text result. Cleared on restart. Created via `delegate()`.

These systems are wired together with duct tape: the orchestrator tools file (`orchestrator/tools.py`) has 1500+ lines juggling both, the Runtime tracks delegates via five separate dictionaries (`_delegate_providers`, `_delegate_tasks`, `_delegate_results`, `_delegate_done`, `_delegate_output_dir`), and there's no unified lifecycle, persistence, or supervision model.

**The goal:** One Agent entity. Two execution modes. Uniform lifecycle, hierarchy, communication, and persistence.

---

## 1.5. Non-Negotiable Requirements <a id="non-negotiables"></a>

Two requirements are **hard constraints** on the design. Everything else is negotiable.

### Requirement 1: Fractality

Every agent — root orchestrator, sub-agent, sub-sub-agent — has the **exact same interface**. Same definition schema, same tool surface (scoped by role, not by level), same ability to spawn children, same lifecycle.

**Reference implementation:** The Chevin framework (`c:\code\chevin`) achieves this cleanly. Every agent at every level gets the same MCP tool setup — including `delegate` to spawn further children. The hierarchy is just IDs (`root.1`, `root.1.1`, `root.1.1.1`). A sub-sub-agent at depth 4 works identically to the root orchestrator. The only thing that varies is the system prompt (which describes the agent's specific role).

**What this means concretely:**
- A cognitive sub-agent spawned by the orchestrator can itself spawn sub-sub-agents using the same `delegate`/`create_agent` tool
- Those sub-sub-agents can spawn further children, to arbitrary depth
- The tool surface at each level includes agent management tools — not just the "core 3" (delegate, post_message, get_snapshot), but the full set appropriate to the agent's role
- The `parent_id` chain is the only thing that distinguishes levels — not different agent types, registries, or execution models
- Registration, lifecycle, persistence, communication all work identically at every level

**Chevin's limitation we must surpass:** In Chevin, delegation is **synchronous** — the parent blocks waiting for the child to finish. This is the part that needed refactoring. Our design must support async delegation where the parent continues working while children execute in parallel.

### Requirement 2: Innate Wake-Up on Completion

When a child agent finishes its work, it **automatically wakes its parent**. No polling. No heartbeat needed for "is my child done?" The completion event flows up the hierarchy as an inbox message that triggers the parent's next turn.

**How it works:**
1. Child agent completes (cognitive session ends, or pipeline finishes)
2. Runtime automatically posts a HIGH-priority WORK message to the parent's inbox:
   ```python
   InboxMessage(
       source=child_id,
       target=parent_id,
       type=MessageType.WORK,
       priority=MessagePriority.HIGH,
       data={
           "type": "child_completed",
           "child_id": child_id,
           "status": "completed",  # or "failed"
           "result_preview": first_500_chars,
       }
   )
   ```
3. Parent's inbox watcher triggers a new turn — the parent wakes up and processes the result
4. If the parent is mid-session (running a cognitive turn), the message is **injected** into the active session via `send_user_message()` — so the parent sees it immediately without waiting for its current turn to end

**This replaces:**
- Manual `delegate_collect()` (blocking poll)
- Manual `delegate_status()` checks on heartbeat ticks
- The heartbeat pattern for monitoring child completion

**The heartbeat remains useful for:** periodic liveness signals to the parent while long-running work is in progress (not for completion notification — that's automatic).

**The cascade:** This composes fractally. When `orch.1.2.3` completes, it wakes `orch.1.2`. When `orch.1.2` finishes processing that result and completes its own work, it wakes `orch.1`. When `orch.1` completes, it wakes `orch` (the orchestrator). The orchestrator processes the result and may report to the user. Completion signals ripple up the tree automatically.

### UI Reference: Sidekick-Web Thread Navigation

The Sidekick-Web app (`c:\code\chevin`) implemented proven UX patterns for fractal agent navigation that should be carried into the unified design:

**Agent Tree Sidebar** (`chevin/widgets/agent_tree_sidebar.dart`):
- Hierarchical list with depth-based indentation (calculated from dot-notation IDs)
- Pulsing green dot for running agents, gray checkmark for completed, red X for errors
- Activity count per agent, task title display
- Pop-out button to open agent in separate browser window (position persisted in localStorage)

**Delegation Items in Chat Feed** (`chevin/chevin_chat.dart`):
- When an agent delegates, a 🚀 item appears inline in the activity feed
- Items carry `linkedAgentId` — clicking navigates into that sub-agent's thread
- Rendered as `InkWell` with ↗ icon, underlined in primary color

**Depth Navigation with Zoom Transitions**:
- Clicking a delegation item → `selectAgent(childId)` → feed switches to child's activity
- **Direction-aware animation** (400ms, easeInOut):
  - Zooming IN (parent→child): outgoing scales to 1.08 + fades, incoming scales from 0.92 + fades in
  - Zooming OUT (child→parent): reversed scaling
  - Creates a parallax "depth" effect that makes the hierarchy feel spatial

**Pop-Out Windows** (`screens/sub_agent_popup_screen.dart`):
- Sub-agents can open in separate browser windows via `window.open()`
- **BroadcastChannel sync** between windows — completion events propagate to main window
- Docking button returns agent to main window

**Layout modes**: Docked sidebar → Maximized (tree sidebar + chat area, max-width 860px) → Popped out window

These patterns should inform the Windows UI: delegation items in the orchestrator's conversation should be clickable to open/navigate to the child agent's window. The agent tree sidebar should show the full hierarchy with real-time status.

---

## 2. The Real Problems <a id="the-real-problems"></a>

Previous analyses identified the obvious structural duplication. Here are the **hard problems** they understate or miss:

### 2.1 The Bridge Subprocess Model vs Step Pipeline Execution

The step pipeline executes steps sequentially within the Runtime's event loop (`runtime._execute_pipeline`). The bridge provider runs a **separate OS process** (`bun run claude-bridge.ts`) that hosts the Claude Agent SDK — the SDK itself manages the multi-turn loop, spawns Claude Code subprocesses for file/bash/web tools, and communicates via NDJSON pipes.

This is not "just another provider." The bridge subprocess is the *reason* delegates have file I/O, bash, web search. These tools don't come from ATN — they come from the Claude Code subprocess that the SDK spawns internally. ATN's `_sub_tool_executor` only handles ATN-specific tools (delegate, post_message, get_snapshot) and connector tools. The SDK's built-in tools (Read, Write, Edit, Glob, Grep, Bash, WebSearch, WebFetch) are handled *inside* the bridge process, invisible to ATN.

**Implication:** You cannot give a persistent pipeline agent "delegate-style tools" by simply adding tool definitions. The tools require the Claude Agent SDK subprocess infrastructure. A cognitive step using `send_orchestrate()` via `BridgeProvider` already gets them — but only during that step's execution, not across the full agent lifecycle.

### 2.2 Session Continuity vs Step Boundaries

A delegate runs as a single continuous session — one `send_orchestrate()` call that may take 50 turns. The session has memory (the SDK manages conversation history). A pipeline agent runs discrete steps; each cognitive step is a fresh LLM call (or a fresh `send_orchestrate` invocation). There is no cross-step session state beyond what's explicitly piped via `previous_outputs`.

Unifying these means deciding: does a "cognitive mode" agent run as one long session (like a delegate) or as repeated wake-execute-sleep cycles (like a pipeline agent)? The answer must be **both**, depending on the task.

### 2.3 The Orchestrator is Special — And It Can't Not Be

The orchestrator (`orchestrator/__init__.py:444-508`) is created as an `AgentDefinition` with a single cognitive step, `tool_executors: "orchestrator"`, and concurrency 1. It *looks* like a normal agent. But:

- It cannot be unregistered (`runtime.py:296-299`)
- It has the full tool surface (40+ tools) — no other agent does
- It shares a single BridgeProvider instance with all cognitive steps using `claude_max`
- Its session persists across triggers (via `_session_id` on BridgeProvider)
- It receives user input via inbox messages, not step config

In the unified model, the orchestrator becomes the *root agent* in the hierarchy. It's special because of its *position* (root, human-facing), not because of its *type*. The design must make this explicit rather than implicit.

### 2.4 Heartbeat as Intrinsic vs Heartbeat as Pattern

Currently, the orchestrator's system prompt teaches an LLM *pattern*: "create a separate heartbeat agent that messages yourself." This is fragile — the LLM might forget, create it wrong, or create redundant ones. The user wants heartbeat to be intrinsic: every agent automatically pings its supervisor when work is done, and optionally sends periodic liveness signals.

This is a supervision protocol, not a scheduling feature. It requires the agent to be aware of its own completion state.

### 2.5 Token Budget for Autonomous Agents

Pipeline agents have bounded token usage: each cognitive step has a `max_tokens` cap, and the agent definition has per-provider `budgets`. Delegates have... nothing. They run until `max_turns: 50` is hit or the LLM decides to stop. A runaway delegate can burn through an entire context window with no guard.

The unified model needs budget enforcement that works for both bounded steps and autonomous sessions.

---

## 3. Unified AgentDefinition Schema <a id="unified-agentdefinition-schema"></a>

```python
@dataclass
class AgentDefinition:
    """Blueprint for any agent — pipeline or cognitive."""
    id: str
    name: str
    description: str = ""

    # --- Execution mode ---
    mode: AgentMode = AgentMode.PIPELINE
    # PIPELINE: execute steps[] sequentially (current behavior)
    # COGNITIVE: autonomous LLM session (replaces delegates)

    # --- Pipeline mode fields (ignored in COGNITIVE mode) ---
    steps: list[StepDefinition] = field(default_factory=list)

    # --- Cognitive mode fields (ignored in PIPELINE mode) ---
    provider: str | list[str] = ""          # provider or fallback chain
    model: str = ""                         # model override
    system_prompt: str = ""                 # inline or file reference
    agent_type: str = "general"             # explore, implement, research, etc.
    max_turns: int = 50                     # per-session turn limit
    tools: list[str] = field(default_factory=list)
    # Tool surface: "atn_core" (delegate/message/snapshot),
    #               "atn_full" (all orchestrator tools),
    #               "connectors" (MCP connectors),
    #               specific connector IDs

    # --- Common fields ---
    concurrency: int = 1
    schedule: str | None = None             # e.g. "5m", "1h"
    budgets: dict[str, int] = field(default_factory=dict)
    connector_ids: list[str] = field(default_factory=list)
    output_schema: dict | None = None

    # --- Hierarchy ---
    parent_id: str | None = None            # None = root (orchestrator)
    # Children cannot remove themselves — they signal completion to parent.

    # --- Heartbeat ---
    heartbeat: HeartbeatConfig | None = None
    # Intrinsic. If set, the runtime manages it automatically.

    # --- Metadata ---
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""                    # agent_id of creator


class AgentMode(Enum):
    PIPELINE = "pipeline"       # deterministic step sequence
    COGNITIVE = "cognitive"     # autonomous LLM session


@dataclass
class HeartbeatConfig:
    """Intrinsic heartbeat configuration."""
    interval: str = "5m"                    # ping frequency while active
    on_complete: str = "notify_parent"      # "notify_parent" | "self_deactivate"
    # The agent pings its parent_id on each heartbeat tick.
    # On completion, it sends a completion signal (not self-removal).
```

### Key design decisions:

1. **`mode` field, not a separate class.** A pipeline agent and a cognitive agent are the same entity with different execution strategies. This keeps the registry, loader, store, and all tooling working with one type.

2. **`steps` remains for pipeline mode.** No changes to how deterministic agents work. A pipeline agent with only script steps is still just a "glorified script" — and that's fine.

3. **`parent_id` is first-class.** Every agent knows who created it and who supervises it. The orchestrator has `parent_id=None` (or `parent_id="human"`). Currently this is implicit — delegates get hierarchical IDs (`orch.1.2`) but pipeline agents have no parent concept.

4. **Cognitive mode fields replace delegate config.** Instead of building a BridgeProvider ad-hoc in `_delegate()`, the AgentDefinition carries the provider, model, system prompt, and tool surface.

5. **`tools` is a named capability list, not raw tool definitions.** Instead of `_DELEGATE_TOOL_NAMES`, agents declare capability sets. The runtime resolves these to actual tool definitions at execution time.

---

## 4. Agent Modes: Pipeline vs Cognitive <a id="agent-modes"></a>

### Pipeline Mode (existing behavior, preserved)

```
trigger → drain inbox → step[0] → step[1] → ... → step[N] → output store
```

Each step is a typed executor (script, cognitive, message, pull, collect). Steps pass data forward via `previous_outputs`. The pipeline is a single execution, bounded and deterministic.

**No changes** to how pipeline agents execute. The `_execute_pipeline` method stays as-is.

### Cognitive Mode (replaces delegates)

```
trigger → drain inbox → spawn bridge session → autonomous work → result → output store
```

A cognitive-mode agent is essentially what a delegate is today, but:

- **Registered in the main agent registry** (`runtime._agents`), not `delegate_registry`
- **Has an inbox** — can receive messages mid-session
- **Has an output store** — result persists across restarts
- **Has execution history** — recorded in ExecutionLog like any pipeline agent
- **Has a lifecycle** — registered, active, running, stopped, error
- **Respects budgets** — token limits enforced during the session

#### Execution flow for cognitive mode:

```python
async def _execute_cognitive_agent(self, defn, record, cancel):
    """Run an autonomous LLM session for a cognitive-mode agent."""
    # 1. Resolve provider (with fallback chain)
    provider = self._resolve_provider(defn.provider)

    # 2. Build inbox context
    messages = self.inbox.drain(defn.id)
    user_message = self._build_cognitive_message(defn, messages)

    # 3. Resolve tool surface
    tools = self._resolve_tools(defn.tools, defn.connector_ids)

    # 4. Build system prompt (inline or from file)
    system = self._resolve_system_prompt(defn)

    # 5. Run session with budget enforcement
    response = await provider.send_orchestrate(
        message=user_message,
        system=system,
        model=defn.model,
        tools=tools,
        max_turns=defn.max_turns,
        tool_executor=self._make_tool_executor(defn),
        on_chunk=self._make_stream_callback(defn.id, record.execution_id),
    )

    # 6. Record result
    record.output = {"text": response.text, "usage": ..., ...}
    self.output_store.write(AgentOutput(
        agent_id=defn.id, data=record.output, ...
    ))
```

The key insight: this is almost exactly what `_run_delegate_session` does today, but integrated into the Runtime's execution pipeline infrastructure instead of being bolted on in orchestrator/tools.py.

### Hybrid: Pipeline with Cognitive Steps

A pipeline agent can still have cognitive steps (StepType.COGNITIVE). This doesn't change. A cognitive step within a pipeline is a single LLM invocation (or multi-turn orchestrate), bounded by that step's config. It's different from a cognitive-mode agent, which is an entire autonomous session.

Think of it as:
- **Pipeline + cognitive step** = "call an LLM as one step in a recipe"
- **Cognitive mode agent** = "give an LLM a task and let it work"

---

## 5. Lifecycle States and Transitions <a id="lifecycle-states"></a>

The current `AgentStatus` enum works for both modes, with one addition:

```python
class AgentStatus(Enum):
    REGISTERED = "registered"   # defined but not active
    ACTIVE = "active"           # scheduler + inbox triggers honored
    RUNNING = "running"         # executing (pipeline or cognitive session)
    STOPPED = "stopped"         # manually deactivated
    ERROR = "error"             # last execution failed
    COMPLETED = "completed"     # NEW: one-shot agent finished successfully
```

### State transitions:

```
                  ┌─────────────────────────────┐
                  │                               │
  create ──→ REGISTERED ──activate──→ ACTIVE ──trigger──→ RUNNING
                  ↑                      ↑          │
                  │                      │          ├──success──→ ACTIVE (if scheduled)
                  │                      │          ├──success──→ COMPLETED (if one-shot)
                  │                      │          ├──failure──→ ERROR
                  │                      │          └──killed───→ STOPPED
                  │                      │
                  │                      └──deactivate──→ STOPPED
                  │                                         │
                  └───────────activate─────────────────────┘
```

**COMPLETED** is new. One-shot cognitive agents (the equivalent of today's delegates) transition to COMPLETED instead of back to ACTIVE. The parent decides whether to remove them or keep them for inspection.

### For one-shot cognitive agents (replacing delegates):

```
create + activate + trigger → RUNNING → COMPLETED
```

The parent (orchestrator) creates, activates, and triggers in one operation — equivalent to today's `delegate()`. The agent runs autonomously and transitions to COMPLETED. The parent inspects the result via `get_output(agent_id)` and can `remove_agent` when done.

---

## 6. The Intrinsic Heartbeat <a id="intrinsic-heartbeat"></a>

### Problem with current approach

Today, the orchestrator's system prompt tells it to manually create a heartbeat agent — a separate pipeline agent with a MESSAGE step that pings the orchestrator on a schedule. This is:
- **Fragile:** LLM might forget or misconfigure it
- **Wasteful:** Creates a real agent just to send a timer message
- **Not universal:** Only the orchestrator knows this pattern

### How it relates to the innate wake-up

The completion callback (Section 8) handles the **event-driven** case — "child finished, wake parent." The heartbeat handles the **periodic** case — "I'm still alive, here's my status." Together they cover all supervision needs:

- **Child completes** → completion callback injects result into parent's turn (immediate)
- **Child is still running** → heartbeat pings parent with status (periodic)
- **Parent wants to check in** → pulls child's output store (on-demand)

The heartbeat is NOT the mechanism for collecting results. It's a liveness signal.

### Intrinsic heartbeat design

Every agent with `heartbeat` config gets automatic periodic pinging, managed by the Runtime:

```python
@dataclass
class HeartbeatConfig:
    interval: str = "5m"
    on_complete: str = "notify_parent"  # what to do when the agent signals done
```

#### How it works:

1. **While an agent is ACTIVE or RUNNING**, the Runtime's scheduler loop checks heartbeat intervals (same as it checks schedule intervals today).

2. **On each heartbeat tick**, the Runtime posts a message to the agent's **parent**:
   ```python
   InboxMessage(
       source=agent_id,
       target=parent_id,
       type=MessageType.INFO,
       priority=MessagePriority.NORMAL,
       data={"type": "heartbeat", "child_id": agent_id, "status": current_status}
   )
   ```

3. **When the agent transitions to COMPLETED**, the Runtime sends a completion signal to the parent:
   ```python
   InboxMessage(
       source=agent_id,
       target=parent_id,
       type=MessageType.WORK,
       priority=MessagePriority.HIGH,
       data={"type": "child_completed", "child_id": agent_id, "result_preview": ...}
   )
   ```

4. **The parent decides what to do.** It can inspect the child's output, remove it, or keep it. Agents never self-remove.

#### For the orchestrator specifically:

The orchestrator gets `heartbeat.interval = "5m"` by default when it has active work (the Runtime detects this from active child agents). This replaces the manual heartbeat pattern entirely.

#### For deterministic pipeline agents:

A pipeline agent with a heartbeat simply has the Runtime send periodic status pings to its parent. This is useful for long-running scheduled agents — the parent gets notified that the child is still alive and producing results.

#### Key invariant: agents cannot remove themselves

An agent can:
- Signal completion (transition to COMPLETED)
- Deactivate itself (transition to STOPPED)
- Write to its output store

An agent **cannot**:
- Remove itself from the registry
- Remove its parent or siblings
- Create agents above itself in the hierarchy

Only a parent can remove its children. This is enforced in `remove_agent`:

```python
async def remove_agent(self, agent_id: str, *, requester: str = "user") -> None:
    defn = self._agents.get(agent_id)
    if defn and defn.parent_id and defn.parent_id != requester:
        if requester != "user":  # human override always works
            raise PermissionError(f"Only parent '{defn.parent_id}' can remove '{agent_id}'")
    ...
```

---

## 7. Hierarchy and Supervision <a id="hierarchy-and-supervision"></a>

### The fractal hierarchy

```
human
  └── orchestrator (parent_id=None)
        ├── pipeline-agent-1 (parent_id="orchestrator")
        ├── cognitive-agent-2 (parent_id="orchestrator")
        │     ├── sub-agent-2.1 (parent_id="cognitive-agent-2")
        │     └── sub-agent-2.2 (parent_id="cognitive-agent-2")
        └── pipeline-agent-3 (parent_id="orchestrator")
```

### Rules:

1. **Human → Orchestrator** and **Orchestrator → Sub-agent** are the same relationship. The human is just another supervisor node.

2. **Agents can create children** (via `delegate` or `create_agent` with `parent_id` set).

3. **Kill cascades down.** Killing a parent kills all descendants (current `kill_agent` + `DelegateRegistry.get_descendants` — unified into the main registry).

4. **Completion signals go up.** When a child completes, it notifies its parent. The parent decides next steps.

5. **No orphans.** If a parent is removed, its children are either:
   - Re-parented to the grandparent (promotion)
   - Removed (cascade delete — current behavior for delegates)

   Decision: **cascade delete**, matching current delegate behavior. Pipeline agents created by the orchestrator are typically meant to persist, so `remove_agent` on the orchestrator requires `_force=True` (already the case).

### ID scheme

Current delegates use hierarchical IDs: `orch.1`, `orch.1.2`. Pipeline agents use user-provided IDs like `website-monitor`.

**Unified approach:** Keep both. Pipeline agents created by the orchestrator get human-readable IDs (the orchestrator picks them in `create_agent`). Cognitive agents spawned as one-shots get hierarchical IDs auto-generated from the parent.

```python
def generate_child_id(self, parent_id: str) -> str:
    count = self._child_counters.get(parent_id, 0) + 1
    self._child_counters[parent_id] = count
    return f"{parent_id}.{count}"
```

This already exists in `DelegateRegistry.generate_child_id` — it moves to Runtime.

---

## 8. Communication Patterns <a id="communication-patterns"></a>

### The Core Pattern: Result Bubbling as Tool Returns (Innate Wake-Up)

**This is the most important communication pattern in the unified model.**

In the current delegate system, the parent polls (`delegate_status`) or blocks (`delegate_collect`). Neither is natural. In the unified model, a child's completion **wakes the parent automatically** by returning into the parent's active tool-use turn.

This is the pattern from the user's Chevin project (c:\code\chevin), and it's how the system should work:

When a parent spawns a child via `create_agent` (the unified `delegate`), the call returns immediately with an `agent_id`. But under the hood, the Runtime registers a **completion callback** tied to the parent:

```python
async def _spawn_cognitive_child(self, parent_id: str, defn: AgentDefinition):
    """Spawn a child agent. The parent's tool call returns immediately.
    When the child completes, the result is injected into the parent's
    active session as a tool return or inbox message."""

    agent_id = defn.id
    self._completion_callbacks[agent_id] = parent_id

    # Register, activate, trigger — the child starts working
    await self.register_agent(defn)
    await self.activate_agent(agent_id)
    await self.trigger_run(agent_id, source=f"agent:{parent_id}")

    return agent_id  # Parent gets this immediately
```

When the child completes:

```python
async def _on_agent_completed(self, agent_id: str, result: str):
    parent_id = self._completion_callbacks.pop(agent_id, None)
    if not parent_id:
        return

    # Option A: If parent has an active bridge session, inject directly
    provider = self._active_providers.get(parent_id)
    if provider and hasattr(provider, 'send_user_message'):
        await provider.send_user_message(
            f"[Child agent '{agent_id}' completed]\n\n{result}"
        )
        return

    # Option B: Parent is not currently running — post to inbox
    # The next time the parent wakes (heartbeat, schedule, user message),
    # it will see this in its inbox
    self.inbox.post(InboxMessage(
        source=agent_id,
        target=parent_id,
        type=MessageType.WORK,
        priority=MessagePriority.HIGH,
        data={
            "type": "child_completed",
            "child_id": agent_id,
            "result_preview": result[:2000],
        }
    ))
```

**Two paths, one outcome:**
- **Parent is running** → result injected into the parent's active LLM turn. The parent sees it as a message in its conversation: "Child agent 'research-auth' completed" with the result. The parent can immediately act on it — no polling, no blocking.
- **Parent is not running** → result posted to parent's inbox as a HIGH priority WORK message. The heartbeat (or any other trigger) wakes the parent, which sees the completion in its inbox on the next turn.

This means: **agents wake each other up innately**. No separate heartbeat agent needed just for collection. The heartbeat's job is only to keep the parent alive for periodic check-ins — the actual work notifications come through completion callbacks.

### Fractal Context Propagation

Every cognitive agent gets the infrastructure to spawn its own children. This is fractality:

```python
def _resolve_tools_for_agent(self, defn: AgentDefinition) -> list[dict]:
    tools = []

    if "atn_core" in defn.tools:
        # The agent gets create_agent (scoped: children only),
        # post_message, get_snapshot, get_output
        tools.extend(self._get_scoped_tools(defn.id))

    if "connectors" in defn.tools:
        tools.extend(self.connectors.get_all_tools(defn.connector_ids))

    return tools

def _get_scoped_tools(self, agent_id: str) -> list[dict]:
    """Tools scoped to this agent's position in the hierarchy."""
    return [
        # create_agent — but children get parent_id=agent_id automatically
        {
            "name": "create_agent",
            "description": "Create a child agent to work on a subtask.",
            # Parent ID is injected by the Runtime — the agent can't
            # create siblings or agents above itself
        },
        # post_message — can message any agent (parent, siblings, children)
        {"name": "post_message", ...},
        # get_snapshot — see the full system state
        {"name": "get_snapshot", ...},
        # get_output — read any agent's last result
        {"name": "get_output", ...},
    ]
```

The key: when agent `orch.1` calls `create_agent`, the Runtime automatically sets `parent_id="orch.1"` on the new agent. The child gets `orch.1.1` as its ID. And that child, if it has `atn_core` tools, can spawn `orch.1.1.1`. Same API at every level. Same completion callbacks. Same heartbeat protocol.

### Unified inbox

Every agent has an inbox via `InboxManager`. Pipeline agents use it. The orchestrator uses it. The only agents that *don't* use it are delegates — because they're not in the inbox system at all.

In the unified model, cognitive-mode agents get inboxes. Messages delivered while a cognitive session is running are:

1. **Injected into the active session** via `BridgeProvider.send_user_message()` (real-time delivery)
2. **Queued in the inbox** (if the session has ended, picked up on next wake-up)

This is already implemented for delegates via `delegate_message` → `runtime.send_delegate_message`. In the unified model, `post_message` just works for any agent — the Runtime checks if there's an active bridge session and injects the message if so.

### Output store (no change needed)

Pipeline agents write to the output store on completion. Cognitive agents do the same. Pull steps read from the output store. This all works as-is.

### Mid-session messaging (delegate_message → post_message)

Currently `delegate_message` is a separate tool. In the unified model, `post_message` handles everything:

```python
async def post_message(target_id, content, priority="normal"):
    # 1. If target has an active cognitive session, inject directly
    provider = runtime._active_providers.get(target_id)
    if provider and hasattr(provider, 'send_user_message'):
        await provider.send_user_message(content)
        return  # Delivered in real-time

    # 2. Otherwise, post to inbox (picked up on next wake-up)
    inbox.post(InboxMessage(...))
```

### The working thread IS the record

The user wants: "no separate summary needed — the working thread is the record."

For cognitive agents, this means the bridge session transcript (all turns, tool calls, results) is the execution record. Currently, delegates only save a text stream to a `.log` file and a `result_preview` (500 chars). The full transcript lives in the Claude SDK session — accessible via `BridgeProvider.get_session_context()` but lost on restart.

**Design:** The cognitive agent's execution record stores:
- `output.text`: Final response text (already done)
- `output.transcript_path`: Path to the full session transcript file
- The Runtime persists the transcript to disk when the session completes

```python
# On cognitive session completion:
transcript = await provider.get_session_context()
transcript_path = agents_dir / defn.id / f"{execution_id}.transcript.json"
transcript_path.write_text(json.dumps(transcript))
record.output["transcript_path"] = str(transcript_path)
```

This gives us the "working thread as record" property while keeping the execution record lightweight.

---

## 9. Tool Surface Per Agent Mode <a id="tool-surface"></a>

### Pipeline mode agents

No change. Tool surfaces are defined per cognitive step via `step.config.tools` or `step.config.tool_executors`.

### Cognitive mode agents

Tools are declared on the AgentDefinition:

| `tools` value | What the agent gets |
|---|---|
| `["sdk_builtin"]` | Only Claude SDK tools (Read, Write, Bash, etc.) — the default |
| `["sdk_builtin", "atn_core"]` | SDK + delegate, post_message, get_snapshot |
| `["sdk_builtin", "atn_full"]` | SDK + all orchestrator tools (only for orchestrator) |
| `["sdk_builtin", "atn_core", "connectors"]` | SDK + ATN core + all assigned connectors |

The `atn_core` set replaces the current `_DELEGATE_TOOL_NAMES`. The `atn_full` set is what the orchestrator gets. The `sdk_builtin` tools (file I/O, bash, web) always come from the bridge subprocess — ATN doesn't define them, the SDK does.

**Resolution at execution time:**

```python
def _resolve_tools(self, tool_names: list[str], connector_ids: list[str]) -> list[dict]:
    tools = []
    for name in tool_names:
        if name == "atn_core":
            tools.extend(_get_delegate_tools())  # existing function
        elif name == "atn_full":
            tools.extend(get_tool_definitions_for_bridge())  # existing function
        elif name == "connectors":
            tools.extend(self.connectors.get_all_tools(connector_ids))
        # "sdk_builtin" is implicit — handled by the bridge subprocess
    return tools
```

---

## 10. Persistence Model <a id="persistence-model"></a>

### What survives restart

| Entity | Pipeline mode | Cognitive mode |
|---|---|---|
| AgentDefinition | Yes (YAML on disk) | Yes (YAML on disk) |
| Agent status | Yes (re-register from YAML) | Yes (re-register from YAML) |
| Execution history | Yes (JSONL) | Yes (JSONL) |
| Output store | Yes (last result in memory, hydrated from JSONL) | Yes (same) |
| Inbox messages | No (in-memory, lost on restart) | No (same) |
| Active session state | N/A | **No** — session is lost |
| Session transcript | N/A | Yes (persisted to .transcript.json on completion) |

### Cognitive agents after restart

A cognitive-mode agent that was RUNNING when the process died:
1. Its execution record is recovered from `.running.json` (existing crash recovery in `ExecutionLog.recover_running`)
2. It's marked as FAILED with error "Process crashed mid-execution"
3. If it has a schedule, it will re-trigger on the next interval
4. If it was a one-shot, the parent gets notified of the failure

**We do NOT attempt to resume Claude SDK sessions across restarts.** The SDK session is tied to the bridge subprocess. When the process dies, the session is gone. This is acceptable because:
- Most cognitive work completes in minutes
- The execution record captures what was accomplished before the crash
- The parent can retry by spawning a new agent

### Agent definition persistence

Cognitive-mode agents need to be saveable to YAML like pipeline agents:

```yaml
id: research-auth
name: "Research auth approaches"
mode: cognitive
provider: claude_max
model: sonnet
system_prompt: system.md
agent_type: research
max_turns: 30
parent_id: orchestrator
tools:
  - sdk_builtin
  - atn_core
heartbeat:
  interval: 5m
```

The loader (`loader.py`) needs to handle the `mode` field and the cognitive-specific config. This is straightforward — just additional fields in `_validate_agent`.

### One-shot vs persistent cognitive agents

One-shot agents (replacing delegates) are created, activated, triggered, and complete. They don't need a YAML file on disk — they live in memory during execution and in the execution log afterward. The parent inspects the output store and removes the agent.

Persistent cognitive agents (e.g., the orchestrator) have YAML files and survive restarts. They re-register, re-activate, and resume responding to inbox/schedule triggers.

**Decision:** One-shot cognitive agents are NOT persisted to YAML. They're created programmatically and removed after collection. This matches the current delegate model and avoids cluttering the agents directory with ephemeral tasks.

```python
# In create_agent:
if defn.mode == AgentMode.COGNITIVE and is_one_shot:
    # Register in memory only — don't save YAML
    await runtime.register_agent(defn)
    await runtime.activate_agent(defn.id)
    await runtime.trigger_run(defn.id, source=f"agent:{requester}")
else:
    save_agent(defn, config.agents_dir)
    await runtime.register_agent(defn)
```

---

## 11. The Orchestrator in the Unified Model <a id="orchestrator"></a>

The orchestrator becomes a cognitive-mode agent with special properties:

```python
def create_orchestrator_agent(config):
    return AgentDefinition(
        id="orchestrator",
        name="Orchestrator",
        mode=AgentMode.COGNITIVE,
        provider=["claude_max", "anthropic", "gemini"],
        model="claude-sonnet-4-6",
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        agent_type="orchestrator",
        max_turns=50,
        tools=["sdk_builtin", "atn_full", "connectors"],
        concurrency=1,
        parent_id=None,  # root of the hierarchy
        heartbeat=HeartbeatConfig(interval="5m"),
        description="Meta-agent that manages the fleet.",
    )
```

### What changes for the orchestrator:

1. **It's explicitly a cognitive-mode agent**, not a pipeline agent with a single cognitive step. This removes the need for `StepDefinition`, `tool_executors: "orchestrator"`, and the cognitive step executor orchestrate path as the entry point.

2. **Its execution uses the same code path as any cognitive agent.** `_execute_cognitive_agent` replaces the current pipeline→cognitive-step→orchestrate chain.

3. **`create_agent` and `delegate` converge.** When the orchestrator calls `create_agent` with `mode: "cognitive"` and no schedule, it's equivalent to today's `delegate()`. When it calls `create_agent` with `mode: "pipeline"` and a schedule, it's the existing behavior.

### What stays the same:

- It cannot be unregistered (enforced by its root position)
- It has the full tool surface (now expressed as `tools: ["atn_full"]`)
- It wakes on inbox messages from user, scheduler, and other agents
- Its session persists via BridgeProvider's `_session_id`

### Tool consolidation:

`delegate()` becomes sugar for `create_agent(mode="cognitive", ...)` + activate + trigger. The four delegate tools (`delegate`, `delegate_status`, `delegate_message`, `delegate_collect`) collapse:

| Old tool | New equivalent |
|---|---|
| `delegate(prompt, type, title)` | `create_agent(mode="cognitive", ...)` — returns agent_id |
| `delegate_status(id)` | `get_agent(id)` — shows status, output preview |
| `delegate_message(id, content)` | `post_message(target=id, content)` — auto-injects into active session |
| `delegate_collect(id)` | `await_agent(id)` — new tool, blocks until COMPLETED |

We keep `delegate()` as a convenience alias during migration (Phase 2 below).

---

## 12. Token Budgets and Autonomous Agents <a id="token-budgets"></a>

### The problem

Pipeline agents have per-execution budgets checked before each cognitive step (`runtime.py:469-483`). Delegates have no budget enforcement at all.

### Solution: Budget enforcement in the tool executor

For cognitive-mode agents, the Runtime tracks token usage across the session and enforces limits in the tool executor callback:

```python
def _make_tool_executor(self, defn, record):
    budget_limit = defn.budgets.get(provider_name, 0)

    async def executor(name, input):
        # Check budget before each tool call (which triggers a new LLM turn)
        if budget_limit > 0:
            used = record.token_usage.get(provider_name, TokenUsage(provider_name))
            if used.total >= budget_limit:
                # Return a budget-exceeded signal that causes the session to end
                return {"error": "TOKEN_BUDGET_EXCEEDED", "used": used.total, "limit": budget_limit}
        return await actual_executor(name, input)

    return executor
```

Additionally, the `send_orchestrate` response callback accumulates usage:

```python
async def _on_orchestrate_usage(usage):
    _accumulate_usage(record, provider_name, {"usage": usage_dict})
```

### Budget hierarchy

A parent's budget covers its children's usage. When a child completes, its token usage is aggregated into the parent's execution record (already done for collect steps via `_accumulate_child_usage`). In the unified model, this happens automatically when a child agent completes and the parent is notified.

---

## 13. Migration Strategy <a id="migration-strategy"></a>

### Phase 1: Schema Unification (low risk, no behavior change)

**Goal:** Add `mode`, `parent_id`, `heartbeat` to `AgentDefinition`. Add `AgentMode` enum. Add `COMPLETED` status.

**Changes:**
- `models.py`: Add fields with backward-compatible defaults (`mode=PIPELINE`, `parent_id=None`, `heartbeat=None`)
- `loader.py`: Accept new fields in YAML, ignore unknown fields for backward compat
- `agent_registry.py`: Merge `DelegateRegistry` child-counter logic into Runtime (it's 50 lines)
- All existing agents continue to work — new fields are optional

**Risk:** Zero. All additions are backward-compatible.

### Phase 2: Cognitive Mode Execution (medium risk)

**Goal:** Runtime can execute cognitive-mode agents using the same infrastructure as pipeline agents.

**Changes:**
- `runtime.py`: Add `_execute_cognitive_agent` method. `trigger_run` checks `defn.mode` and dispatches to either `_execute_pipeline` or `_execute_cognitive_agent`.
- `runtime.py`: Move delegate provider tracking (`_delegate_providers`, `_delegate_tasks`, etc.) into the unified execution tracking (`_executions`, `_tasks`, `_cancels`).
- `orchestrator/tools.py`: Add `create_agent` support for `mode: "cognitive"` — internally creates a cognitive-mode AgentDefinition, registers it, and triggers it.
- Keep `delegate()` as an alias for `create_agent(mode="cognitive", ...)` so existing orchestrator prompts work.

**Risk:** Medium. The cognitive execution path is new code, but it's largely extracted from `_run_delegate_session` (which is known-working).

### Phase 3: Orchestrator as Cognitive Agent (medium-high risk)

**Goal:** The orchestrator uses `_execute_cognitive_agent` instead of the pipeline→cognitive-step→orchestrate path.

**Changes:**
- `orchestrator/__init__.py`: `create_orchestrator_agent` returns a `mode=COGNITIVE` definition.
- Remove the orchestrator's step pipeline wrapper. It's now directly a cognitive agent.
- The BridgeProvider instance management simplifies — each cognitive agent gets its own (or shares one for the orchestrator's session continuity).

**Risk:** Medium-high. The orchestrator is the core interaction loop. Test thoroughly in parallel with the old code path before cutting over.

### Phase 4: Intrinsic Heartbeat + Hierarchy Enforcement (low risk)

**Goal:** Heartbeat and supervision rules enforced by the Runtime.

**Changes:**
- `runtime.py`: Scheduler loop checks `heartbeat.interval` for active agents and posts heartbeat messages to `parent_id`.
- `runtime.py`: On COMPLETED transition, post completion message to parent.
- `runtime.py`: `remove_agent` enforces parent-only deletion.
- Remove heartbeat pattern from orchestrator system prompt (or mark as deprecated).

**Risk:** Low. Additive feature, doesn't break existing behavior.

### Phase 5: Cleanup (low risk)

**Goal:** Remove dead code and the delegate subsystem.

**Changes:**
- Remove `DelegateRegistry` class (functionality absorbed into Runtime)
- Remove `DelegateNode`, `DelegateStatus` (use `AgentDefinition`, `AgentStatus`)
- Remove `delegate_prompts.py` (system prompts move into agent_type configs)
- Remove delegate-specific tools from orchestrator/tools.py (`_delegate`, `_delegate_status`, `_delegate_message`, `_delegate_collect`)
- Remove Runtime's five delegate dictionaries
- Clean up orchestrator system prompt to reflect unified model

**Risk:** Low, but touches many files. Do it in one PR with comprehensive tests.

---

## 14. What We Are NOT Changing <a id="not-changing"></a>

### Step pipeline execution

The `_execute_pipeline` method, step executors (script, cognitive, message, pull, collect), and the StepDefinition model are untouched. Pipeline-mode agents work exactly as they do today. This is the proven, stable core of ATN.

### The bridge subprocess architecture

We are NOT replacing the BridgeProvider + claude-bridge.ts subprocess model. The Claude Agent SDK requires a Node.js process to host the SDK. ATN is Python. The bridge is the correct architectural boundary. Both cognitive-mode agents and cognitive steps in pipelines use the same bridge infrastructure.

### Provider abstraction

The `Provider` base class, `send()`, `send_stream()`, `send_orchestrate()` — all stay. Cognitive-mode agents use `send_orchestrate()` just like the current orchestrator cognitive step does. Non-bridge providers (Anthropic direct, Ollama, OpenAI) use the base class's generic multi-turn loop.

### MCP connectors

The connector system (`ConnectorManager`, MCP protocol) is orthogonal to agent type. Both pipeline and cognitive agents can use connectors. No changes needed.

### Inbox and output store

`InboxManager` and `OutputStore` are already designed for N agents. They work as-is with cognitive-mode agents added to the registry.

### Event system

`EventBus` and event types are extensible. We'll add a few new event types (`AGENT_COMPLETED`, `HEARTBEAT_SENT`) but the infrastructure is unchanged.

### User-facing YAML format for pipeline agents

Existing `agents/<id>/agent.yaml` files continue to work. The `mode` field defaults to `pipeline` if absent.

---

## 15. Open Questions <a id="open-questions"></a>

### Q1: Should cognitive-mode agents be resumable across triggers?

Currently, the orchestrator maintains session continuity via `BridgeProvider._session_id` — each trigger resumes the same SDK session. Should other cognitive agents have this too?

**Tentative answer:** Yes, for *persistent* cognitive agents (those with a schedule). One-shot cognitive agents don't need it — they run once and complete. The BridgeProvider already handles session resumption; we just need to keep the provider instance alive between triggers for persistent cognitive agents (same as the orchestrator does today).

### Q2: How do we handle the orchestrator's tool surface in the unified model?

The orchestrator has ~40 tools. Sub-agents get ~3. The tool surface is currently controlled by the `tool_executors` config in the cognitive step. In the unified model, it's the `tools` field on the AgentDefinition.

But: `create_agent` and `remove_agent` are tools in the orchestrator's surface. If a cognitive sub-agent has `tools: ["atn_core"]`, it gets `delegate` (which is now `create_agent`). Does that mean it also gets `remove_agent`? It shouldn't — only the parent should remove children.

**Resolution:** The `atn_core` tool set includes: `create_agent` (creates children under self), `post_message`, `get_snapshot`, `get_output`. It does NOT include `remove_agent`, `activate_agent`, `deactivate_agent` — those require parent privilege and are in `atn_full`.

### Q3: What happens to the delegate output directory?

Currently, delegate text streams are persisted to `~/.atn/delegates/<agent_id>.log`. In the unified model, cognitive agent output goes into `agents/<agent_id>/` like everything else.

**Answer:** Remove the separate delegate output directory. Cognitive agent stream output goes to `agents/<agent_id>/<execution_id>.stream.log`. The execution record references it.

### Q4: Credit budget enforcement granularity

Should budget limits be per-execution or per-period (daily/monthly)? Currently, `AgentDefinition.budgets` is per-execution and `CreditBudget` is per-period at the provider level.

**Answer:** Keep both. Per-execution budgets prevent runaway single sessions. Per-period budgets prevent runaway total spend. The Runtime checks both.

### Q5: Can the human create agents directly?

Today, the human talks to the orchestrator, which creates agents. In the unified model, should the CLI/UI allow direct agent creation bypassing the orchestrator?

**Answer:** Yes. The CLI already has `atn create` (via YAML files). The UI can submit agent definitions directly to the Runtime. These agents get `parent_id="user"` or `parent_id=None` depending on whether they're meant to be orchestrator-supervised.

---

## Appendix A: Data Model Diff Summary

```
AgentDefinition:
  + mode: AgentMode = AgentMode.PIPELINE
  + parent_id: str | None = None
  + heartbeat: HeartbeatConfig | None = None
  + provider: str | list[str] = ""       # cognitive mode
  + model: str = ""                       # cognitive mode
  + system_prompt: str = ""               # cognitive mode
  + agent_type: str = "general"           # cognitive mode
  + max_turns: int = 50                   # cognitive mode
  + tools: list[str] = []                 # cognitive mode
  + created_at: datetime
  + created_by: str = ""

AgentStatus:
  + COMPLETED = "completed"

New:
  + AgentMode enum (PIPELINE, COGNITIVE)
  + HeartbeatConfig dataclass

Removed (Phase 5):
  - DelegateNode
  - DelegateStatus
  - DelegateRegistry
```

## Appendix B: File Change Map

| File | Phase | Change |
|---|---|---|
| `models.py` | 1 | Add AgentMode, HeartbeatConfig, extend AgentDefinition/AgentStatus |
| `loader.py` | 1 | Handle new YAML fields |
| `agent_registry.py` | 5 | Delete (child-counter logic moves to Runtime) |
| `runtime.py` | 2-4 | Add `_execute_cognitive_agent`, heartbeat loop, hierarchy enforcement |
| `orchestrator/tools.py` | 2-3 | Unify delegate tools into agent tools, keep aliases |
| `orchestrator/__init__.py` | 3 | Create orchestrator as cognitive-mode agent |
| `delegate_prompts.py` | 5 | Delete (absorbed into agent_type config) |
| `providers/bridge.py` | None | No changes needed |
| `providers/base.py` | None | No changes needed |
| `store.py` | 2 | No structural changes (cognitive agents use existing stores) |
| `steps/cognitive.py` | 3 | Simplify orchestrate path (orchestrator no longer uses it) |
| `steps/base.py` | None | No changes |
| `inbox.py` | None | No changes |

---

## Appendix C: Why Not Just Make Delegates Persistent?

An alternative approach: instead of unifying the models, just add persistence and inbox to delegates. This was considered and rejected because:

1. **Two registries remain.** You'd still have `DelegateRegistry` and the main agent registry, with parallel lookup paths everywhere.

2. **No unified lifecycle.** Delegates would need their own ACTIVE/STOPPED/ERROR states, duplicating `AgentStatus`.

3. **No unified execution tracking.** Delegates would need their own execution log, duplicating `ExecutionLog`.

4. **No unified tooling.** The orchestrator would still need separate tools for delegates vs agents.

5. **The bridge becomes a sidecar.** If delegates are persistent, their BridgeProvider needs to survive restarts and reconnect — much harder than the clean "session per execution" model.

The unified model is more work upfront but eliminates an entire category of "which system do I interact with?" questions.
