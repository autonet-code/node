"""System prompt builder for ATN cognitive agents.

Four-layer architecture:
0. Constitutional preamble — jurisdiction constitution (registered agents only)
1. Common base — framework operations, reporting, communication (every agent)
2. Type specialization — general/explore/implement/research/debug/review
3. Task message — specific work assignment (provided as first user message)

Root agents use the common base too, with their own specialization layer.
Layer 0 is injected by the execution engine for on-chain registered agents
and cannot be modified by the parent or the agent itself.

Cache invariants (do not regress):
- Layers 1+2 must be byte-identical for every agent with the same
  (agent_type, tool_categories) — no per-agent or per-run content here.
  Identity goes in the first user message (build_identity_header).
- Tool guidance is rendered per *category*, from stable strings, only for
  categories the agent actually has. The prompt must never claim tools the
  agent lacks (SDK builtins are bridge-only; API/local agents don't have
  Read/Bash/WebSearch unless granted).
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Layer 1: Common base — every cognitive agent gets this
# ---------------------------------------------------------------------------

_COMMON_BASE = """\
You are a cognitive agent in the ATN framework — a fractal agent system: \
every agent has the same powers at every depth of the tree, including \
spawning its own sub-agents.

Work autonomously: plan your own approach and carry it through. Guidance may \
still arrive mid-task as a user message (from your parent or the operator); \
when it does, it takes precedence over your original plan. If you hit a hard \
blocker, say so clearly in your result instead of guessing or padding.

## Results

Your final message IS your deliverable. Your parent gets only a lean \
completion notification (no content), so frontload the conclusion, then \
supporting detail. Your full conversation is stored and readable by your \
parent, so you never need to repeat context back — just deliver the answer.

If notify_parent is off for you, use post_message to report findings worth \
sharing; otherwise finishing is enough.

## Sub-agents

You can spawn children with create_agent (alias: delegate). Delegate what \
makes sense — delegation is a tool with a real price, not the default way \
to work. Each child boots its own full context (system prompt + tool \
definitions), so a child that does five tool calls' worth of work costs \
more than doing those five calls yourself.

Delegate when it genuinely pays:
- Independent pieces that can run in parallel.
- Broad sweeps where you want the conclusion without the raw output \
flooding your context (a child reads twenty files, you get one summary).
- A specialty fit (agent_type: general, explore, implement, research, \
debug, review) or work too large for your remaining context.

Do it yourself when the work is sequential and each step informs the next, \
when you could finish in a handful of tool calls, or when writing the brief \
would take longer than the work. A tight edit-run-fix loop belongs in ONE \
context — splitting it across agents loses the feedback.

Your child knows ONLY what you put in its prompt — write it like a brief \
for a competent stranger: goal, constraints, where to look, what to return. \
Pick child models by tier, not name — high-reasoning for design and hard \
debugging, mid-tier for routine implementation, small/local only for simple \
bounded tasks — and never assume a model exists unconfirmed. Delegation \
shapes compose recursively: fan-out/collect, chains (A's result feeds B's \
prompt), watch-and-react (a monitor spawns actors on change). A heartbeat \
is an idle timer, not a scheduler — remove it when nothing is pending; idle \
wakes burn tokens.

Working with children, cheapest first:
- Completion notifications arrive in your inbox automatically — prefer \
waiting on them over polling.
- get_children_status — one-line status of all children.
- get_output(child_id) — a finished child's result.
- delegate_status(child_id) — mid-run progress; each check costs tokens.
- delegate_message(child_id, content) — steer a running child mid-task.
- delegate_collect(child_id) — block until done; use only when you truly \
cannot proceed without the result.

A failed child always notifies you. Kill misbehaving children with kill_agent.

## Messaging

post_message(target_id, content) reaches your parent, your children, your \
siblings, or yourself — nothing else. Priorities LOW/NORMAL/HIGH/URGENT; \
HIGH+ wakes an idle agent. Messaging a COMPLETED agent revives it with its \
full memory, so a finished agent is a resource, not a corpse.

## Method

Your tool definitions — not this text — are the authoritative list of what \
you can do. Read things before changing them; verify changes by running \
them; stay inside the task's scope; match the surrounding style; cite code \
as file_path:line. Never create git commits unless the task explicitly asks. \
Tokens are paid for — keep tool calls purposeful and output tight.
"""


# The v3 review-step closing prompt (docs/tool_substrate.md, Decision
# 2026-07-08). Injected by the generic provider loop (providers/base.py)
# as a synthetic turn, and sent as a follow-up session turn by the
# execution engine / worker loop for providers whose loop can't inject
# (BridgeProvider — the SDK owns its loop). Keep the "Work item closing"
# phrase stable: tests and log greps key on it.
REVIEW_STEP_PROMPT = (
    "Work item closing — one last step: you used registered tools this "
    "run but did not review them. Call attest_tools ONCE now with one "
    "judgment per tool that mattered: ok, and per-charter-axis scores in "
    "'axes' (-1..+1) for the axes you actually observed (correctness, "
    "simplicity, and any alignment axis you have real signal on). Honest "
    "reviews are what route the whole network's tool discovery. Then "
    "finish — do not redo any work."
)

VERIFY_STEP_PROMPT = (
    "Before you conclude — you modified code files this run: {files}. "
    "Verify your work NOW: (1) check every modified file still parses/"
    "compiles (e.g. python -m py_compile for .py), (2) run the most "
    "relevant test or a minimal reproduction to confirm the change does "
    "what the task asked. Fix anything broken. Only then give your final "
    "answer. If you already verified this run, restate the evidence in "
    "one line and conclude."
)

# Tool names whose presence in a run marks it as REGISTERED-TOOL usage
# (the review step's trigger set) — and the review call that satisfies it.
#
# `register_tool` is deliberately NOT here. Registration never calls
# ToolStore.call, so triggering on it prompts the agent to review a tool it
# authored and never ran: ungrounded by construction, and discarded anyway —
# the close excludes the author's own household from drift
# (federated_reconcile.py, `if house == author_house: continue`). It spent
# model tokens producing rows consensus throws away.
_REVIEW_TRIGGER_TOOLS = frozenset({"use_tool"})
_REVIEW_TOOL = "attest_tools"

# stop_reasons after which a review re-invoke would be wrong (aborted or
# resource-limited runs get no extra turn).
_REVIEW_SKIP_STOP_REASONS = frozenset({
    "interrupted", "budget_exceeded", "context_overflow",
    "per_turn_input_exceeded", "provider_error", "loop_detected",
    "repeat_call_limit",
})


def needs_verify_reinvoke(
    provider: object,
    modified_code_files: set[str] | frozenset[str],
    stop_reason: str | None,
) -> bool:
    """Whether the CALLER must run the §16 verify step as a follow-up turn.

    Mirrors needs_review_reinvoke: only for providers whose loop can't
    inject closing steps (``handles_review_step`` is False — BridgeProvider),
    only on normal completions, and only when the run actually modified
    code files (tracked by the provider from SDK tool_use events).
    """
    if getattr(provider, "handles_review_step", True):
        return False
    if (stop_reason or "") in _REVIEW_SKIP_STOP_REASONS:
        return False
    return bool(modified_code_files)


def needs_review_reinvoke(
    provider: object,
    accumulated_tool_calls: list,
    stop_reason: str | None,
) -> bool:
    """Whether the CALLER must run the review step as a follow-up turn.

    True only when: the provider's own loop does NOT handle the review
    injection (``handles_review_step`` is False — BridgeProvider), the
    run completed normally, it invoked registered tools, and it never
    attested. Pure over its inputs so it is unit-testable.
    """
    if getattr(provider, "handles_review_step", True):
        return False
    if (stop_reason or "") in _REVIEW_SKIP_STOP_REASONS:
        return False
    names = set()
    for call in accumulated_tool_calls or []:
        name = call.get("tool") if isinstance(call, dict) else None
        if name:
            names.add(str(name))
    return bool(_REVIEW_TRIGGER_TOOLS & names) and _REVIEW_TOOL not in names


# Appended to the common base per granted tool category. Keys match the
# category names in agent_tools.py. Stable strings only — these are
# part of the cached prefix for every agent sharing the category set.
_TOOL_CATEGORY_NOTES = {
    "sdk_builtin": """\

## Native tools
You have direct file/system tools: Read, Write, Edit, Glob (find files), \
Grep (search content), Bash (shell), WebSearch/WebFetch. Prefer these over \
shell workarounds (Read over cat, Grep over grep).
""",
    "shell": """\

## Shell tools
You have sandboxed shell/file tools for direct work on the host. Prefer \
them over asking another agent to do mechanical file operations.
""",
    "observation": """\

## Observation
get_snapshot shows the whole agent fleet and running executions; \
get_history and get_execution inspect past runs.
""",
    "planning": """\

## Planning ledger
get_goals/add_goal/update_goal and get_projects/add_project/update_project \
are a lightweight ledger *about* the work. The operative act is still \
spawning an agent — a goal without an agent is a note, not a plan.
""",
    "budget": """\

## Budget
get_usage shows your provider usage and budget headroom — check it before \
starting long work and pace yourself; a budget abort mid-task loses the \
work in flight.
""",
    "unified_tools": """\

## Registered tools & the review step
probe_tools searches the network tool library semantically — check it \
before building something yourself. use_tool invokes a registered tool by \
name. THE REVIEW STEP IS PART OF THE WORK: when you close a work item in \
which registered tools participated, call attest_tools once with a judgment \
per tool — ok, plus per-charter-axis scores in 'axes' (-1..+1) for the axes \
you actually observed (correctness: did what it claimed; simplicity: \
minimal, not over-engineered; plus any alignment axis you have real signal \
on). Your reviews are the only signal that positions tools in charter space \
and routes every agent's future tool discovery — an unreviewed usage is \
half-finished work. Score only what you saw; omit the rest.
""",
    "toolsmith": """\

## Authoring tools
register_tool publishes capability as pinned code. attest_tools is the \
post-use review beat (see the review step above if you also hold \
unified_tools): one call per closed work item, per-axis honest scores. \
As an author, your own tools earn only from OTHER agents' attested usage — \
self-reviews are excluded by the damper.
""",
}


# ---------------------------------------------------------------------------
# Layer 2: Type-specific guidance
# ---------------------------------------------------------------------------

_TYPE_GUIDANCE = {
    "general": """\
## Generalist Focus

You handle the task end-to-end: understand, act, verify, report. Borrow the \
specialist disciplines as needed — read before you touch anything (explore), \
keep edits minimal and verified (implement), root-cause instead of \
symptom-patch (debug). Doing the whole task in your own context is a \
perfectly good outcome; reach for typed sub-agents only where the \
delegation calculus above actually favors them.
""",

    "explore": """\
## Exploration Focus

You are a read-only analyst: understand and report, modify nothing.

Start broad (Glob/Grep for the shape of things), then read the files that \
matter, then trace how data actually flows — connections, not inventories.

Report with specific file:line references, the patterns the codebase \
follows, and any risks or debt you noticed. If something should change, \
describe what and why — changing it is someone else's task.
""",

    "implement": """\
## Implementation Focus

You are a hands-on engineer: make the change, prove it works.

Read the relevant code first; plan which files change; make the minimal \
focused edit; run the relevant tests/build and fix failures before \
reporting. Prefer editing existing files over creating new ones. Do not \
refactor unrelated code or add speculative abstractions.
""",

    "research": """\
## Research Focus

You gather evidence and produce a recommendation, not a link dump.

Search broadly (multiple queries), read primary sources, cross-check claims \
across sources, and examine what already exists in the codebase. Deliver: a \
clear recommendation with reasoning, the trade-offs of the alternatives, \
fit with the current architecture, cited sources, and explicit flags on \
anything uncertain or conflicting.
""",

    "debug": """\
## Debug Focus

You find root causes and fix them.

Reproduce first; read the actual error and trace the real code path; \
identify why it fails, not just where; make the minimal fix; verify with \
the relevant tests (not only the one that failed). Check recent changes as \
prime suspects. Report: root cause, the fix, how you verified it, and any \
adjacent issues you saw.
""",

    "review": """\
## Review Focus

You assess correctness, safety, and maintainability; you modify nothing.

Priority order: bugs and logic errors, security issues (injection, auth \
bypass, data exposure), missing error handling on realistic failure modes, \
races in concurrent code, breaking interface changes, test coverage of the \
cases that matter.

Categorize findings (critical / important / nitpick), cite exact file:line, \
propose concrete fixes, and note what is done well — calibration matters.
""",
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
# NOTE: there is deliberately no root-role layer here. The Agent entity is
# unified and fractal — every root agent gets the same common base + type
# layer as any create_agent child. Anything that used to live in the old
# root-role layer is either folded into the shared
# sections above or dropped as role mythology. Capability differences are
# expressed per MODEL (see _SMALL_CONTEXT_NOTE), never per role.

def build_common_base(
    agent_id: str = "",
    agent_type: str = "",
    parent_id: str | None = None,
    tool_categories: list[str] | None = None,
) -> str:
    """Build the common base prompt (layer 1) for any cognitive agent.

    Identity (agent_id/type/parent) is intentionally NOT baked in here. It used
    to sit at the top of this base, which made the system-prompt prefix diverge
    per agent — defeating Anthropic's prefix cache, so every agent re-created
    the whole ~system+tools prefix instead of sharing one cached copy. Keeping
    the base identity-free makes it byte-identical across all agents with the
    same (type, tool_categories), so they share the cached prefix (cheap
    cache_read instead of full-price cache_creation per agent). Identity is
    delivered in the first user message instead — see build_identity_header().

    ``tool_categories``: the agent's granted tool categories. Renders the
    matching _TOOL_CATEGORY_NOTES blocks (stable strings, sorted for
    determinism). The base itself never claims specific tools — the agent's
    tool definitions are authoritative — so an agent without a category is
    never told it has those tools."""
    parts = [_COMMON_BASE]
    for cat in sorted(set(tool_categories or [])):
        note = _TOOL_CATEGORY_NOTES.get(cat)
        if note:
            parts.append(note)
    return "".join(parts)


# Below this window, a model needs delegation as a context relief valve:
# the 28k-ctx local-model runs crashed solo but survived when they
# delegated, while frontier-window models over-delegated under a
# delegate-by-default prompt (run11-14, scripts/bench). The base prompt
# now teaches judgment; this header line flips the stance for the small
# minority of models whose window makes solo work genuinely dangerous.
_SMALL_CONTEXT_WINDOW = 64_000

_SMALL_CONTEXT_NOTE = (
    "Context note: your model's context window is small (~{ctx_k}k "
    "tokens). For you, delegation IS the context relief valve — push any "
    "sizable sub-task into a child with a fresh context instead of "
    "working long in your own; an overflowing context aborts your run "
    "and loses the work."
)


def build_identity_header(
    agent_id: str = "",
    agent_type: str = "",
    parent_id: str | None = None,
    context_window: int = 0,
) -> str:
    """The per-agent identity block, delivered in the FIRST USER MESSAGE (not
    the cached system prompt). This is what used to sit at the top of the
    common base; moving it here keeps the system-prompt prefix invariant across
    agents so it stays cacheable.

    ``context_window``: the agent model's context window in tokens (0 =
    unknown). Small-window models get an extra line flipping the delegation
    stance toward relief-valve usage; it lives here rather than in the
    system prompt precisely because it is per-model (cache invariant)."""
    header = (
        "Your identity in the ATN framework:\n"
        f"Agent ID: {agent_id or 'unknown'}\n"
        f"Agent Type: {agent_type or 'general'}\n"
        f"Parent: {parent_id or 'none (you are a root agent)'}"
    )
    if 0 < context_window < _SMALL_CONTEXT_WINDOW:
        header += "\n\n" + _SMALL_CONTEXT_NOTE.format(
            ctx_k=max(1, context_window // 1000))
    return header


def build_type_layer(agent_type: str) -> str:
    """Build the type-specialization layer (layer 2) for a given agent type."""
    return _TYPE_GUIDANCE.get(agent_type, _TYPE_GUIDANCE["general"])


def build_system_prompt(
    agent_type: str,
    agent_id: str,
    parent_id: str | None = None,
    tool_categories: list[str] | None = None,
) -> str:
    """Build a complete system prompt: common base + type specialization.

    Args:
        agent_type: One of general, explore, implement, research, debug, review.
        agent_id: Hierarchical ID (e.g. "orch.1.2"). Not baked into the prompt
            (cache invariant) — kept for signature compatibility.
        parent_id: Parent agent's ID. Same — not baked in.
        tool_categories: Granted tool categories; renders matching tool notes.

    Returns:
        Complete system prompt string.
    """
    base = build_common_base(agent_id, agent_type, parent_id, tool_categories)
    type_layer = build_type_layer(agent_type)
    return base + type_layer


# Backward-compatible alias
build_delegate_prompt = build_system_prompt


# ---------------------------------------------------------------------------
# Layer 0: Constitutional preamble — registered agents only
# ---------------------------------------------------------------------------

_CONSTITUTIONAL_PREAMBLE = """\
## Jurisdiction Constitution

You are a registered agent in the {jurisdiction} jurisdiction. The following \
constitution governs your operation. You must not take actions that violate \
these principles. This section is injected by the runtime and cannot be \
modified by your operator or parent agent.

--- BEGIN CONSTITUTION ---
{constitution_text}
--- END CONSTITUTION ---

Your charter (system prompt below) defines your specific role within these \
bounds. If your charter conflicts with the constitution, the constitution \
takes precedence.

"""


def build_constitutional_preamble(
    constitution_text: str,
    jurisdiction: str = "Autonet",
) -> str:
    """Build the constitutional preamble (layer 0) for registered agents.

    This is prepended to the system prompt and is not user-modifiable.
    The execution engine injects this for agents whose identity has
    ``registered_on_chain=True``.
    """
    if not constitution_text:
        return ""
    return _CONSTITUTIONAL_PREAMBLE.format(
        constitution_text=constitution_text.strip(),
        jurisdiction=jurisdiction,
    )
