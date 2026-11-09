"""Declarative loop policy carried by a harness distro (Decision 2026-10-12).

docs/tool_substrate.md: "Decision (2026-10-12): swappable harness parts".

The generic agent loop (providers/base.py) and the prompt builder
(delegate_prompts.py) used to hardcode their POLICIES: system prompt
templates, the compaction instruction and reduction ladder, the closing
verify/review turns, loop-guard limits, headroom thresholds, the
parallel-safe tool list. Those are exactly where distros are meant to
compete ("the headroom is policy"), so they now live in one value,
``HarnessPolicy``, which a distro manifest carries in its config blob.

What stays OUT of the policy (the safety kernel, never distro-settable):
budget gate + usage recorder, tool dispatch, the
governance heartbeat, history storage, the
per-run plan tool's validation, orphan/history repair.

Resolution order for one agent run:
  agent definition explicit knobs (per_turn_input_max, repeat_call_limit)
  > the agent's bound distro policy (AgentDefinition.loadout)
  > the reference policy (this daemon's built-in values).

A distro policy is an OVERLAY: missing keys inherit the reference, unknown
keys are ignored, malformed values (wrong type, out of range, a template
missing its placeholders) fall back to the reference value for that key.
A broken distro can degrade an agent's prompts; it can never crash a run.

The reference policy is built from the live module constants, so with no
distro bound the loop is byte-identical to the pre-policy code
(tests/atn/test_swappable_harness.py pins that).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

log = logging.getLogger(__name__)

# Policy-as-code extension point (Decision 2026-10-12, item 4: DEFINED, NOT
# DISPATCHED). A pinned tool may declare ``capabilities.provides`` over these
# hook names (registration keeps them; any other non-core name is dropped).
# The intended contract for "policy:compaction": the loop would call the
# provider (sealed protocol) with {"messages": [...], "budget_chars": N,
# "policy": {...}} and accept a reduced message list back, validated by the
# kernel (_repair_history) before use. Not wired: the loop ignores these
# claims today, and only the declarative overlay above is live. Wiring it
# belongs at Provider._compact_messages, behind the same three gates as a
# core swap (approval ack, ToolBond bond, per-agent loadout).
POLICY_HOOKS: tuple[str, ...] = (
    "policy:compaction",
    "policy:system_prompt",
    "policy:review_step",
)

# Distro config blob marker. A distro manifest's code blob is JSON with this
# ``kind``; anything else (the pre-2026-10-12 plain-text blob) carries no
# policy and resolves to the reference.
DISTRO_KIND = "atn_harness_distro"
POLICY_VERSION = 1

# Template fields and the placeholders each must keep. A distro that drops a
# placeholder (or adds an unknown one) gets the reference value for that key.
_TEMPLATE_KEYS: dict[str, dict[str, Any]] = {
    "review_step_prompt": {"tools": "x"},
    "verify_step_prompt": {"files": "x"},
    "small_context_note": {"ctx_k": 1},
    "headroom_note_template": {"pct": 1, "remaining": 1},
}

# (min, max) clamps for numeric knobs. Out-of-range values are clamped, not
# rejected: a distro asking for 0 compactions gets 0, one asking for 10**9
# protected chars gets the ceiling.
_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "small_context_window": (0, 10_000_000),
    "prune_protect_recent_chars": (0, 10_000_000),
    "prune_min_savings": (0, 10_000_000),
    "compact_retain_user_chars": (0, 10_000_000),
    "compact_retain_turns": (1, 50),
    "max_compactions_per_run": (0, 20),
    "compact_summary_max_tokens": (256, 64_000),
    "compact_overflow_retries": (0, 20),
    "max_hard_resets_per_run": (0, 20),
    "reset_digest_cap_chars": (500, 200_000),
    "repeat_call_limit": (0, 100),
    "oscillation_window": (2, 100),
    "per_turn_input_max": (0, 10_000_000),
    "max_parallel_tool_calls": (1, 64),
    "verify_follow_up_max_turns": (1, 50),
    "review_follow_up_max_turns": (1, 50),
}
_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "compaction_min_reduction": (0.0, 0.95),
    "compact_drop_fraction": (0.05, 0.9),
}


@dataclass(frozen=True)
class HarnessPolicy:
    """Every loop/prompt policy a distro may set. Field docs name the code
    that reads each one; values here are never defaults of their own (the
    reference instance is built from the module constants)."""

    # --- System prompt templates (delegate_prompts.py) ---
    common_base: str
    tool_category_notes: Mapping[str, str]
    type_guidance: Mapping[str, str]
    small_context_window: int
    small_context_note: str           # {ctx_k}
    # --- Closing turns (base loop + bridge follow-ups) ---
    verify_enabled: bool
    verify_step_prompt: str           # {files}
    code_file_suffixes: tuple[str, ...]
    verify_follow_up_max_turns: int
    review_enabled: bool
    review_step_prompt: str           # {tools}
    review_follow_up_max_turns: int
    # --- Compaction instruction + reduction ladder (base loop) ---
    compaction_inline_prompt: str
    prune_protect_recent_chars: int
    prune_min_savings: int
    compact_retain_user_chars: int
    compact_retain_turns: int
    max_compactions_per_run: int
    compaction_min_reduction: float
    compact_summary_max_tokens: int
    compact_overflow_retries: int
    compact_drop_fraction: float
    max_hard_resets_per_run: int
    reset_digest_cap_chars: int
    # --- Loop guard ---
    repeat_call_limit: int
    oscillation_window: int
    per_turn_input_max: int
    loop_warning_prompt: str
    truncated_tool_call_prompt: str
    # --- Context headroom (§23) ---
    headroom_levels: tuple[float, ...]
    headroom_note_template: str       # {pct} {remaining}
    # --- Parallel tool execution (§21) ---
    parallel_safe_tools: frozenset[str]
    max_parallel_tool_calls: int
    # --- Tool bundle selection. None = the agent's own granted categories.
    # A list NARROWS the agent's grant (intersection); it can never widen
    # what the owner granted.
    tool_bundles: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (sorted collections, so the blob is stable)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, frozenset):
                v = sorted(v)
            elif isinstance(v, tuple):
                v = list(v)
            elif isinstance(v, Mapping):
                v = {k: v[k] for k in sorted(v)}
            out[f.name] = v
        return out

    def select_tools(self, granted: list[str] | None) -> list[str]:
        """The agent's tool categories after the distro's bundle selection.
        Identity when the distro selects nothing."""
        granted = list(granted or [])
        if self.tool_bundles is None:
            return granted
        allowed = set(self.tool_bundles)
        return [t for t in granted if t in allowed]

    def is_reference(self) -> bool:
        return self == reference_policy()


_REFERENCE: HarnessPolicy | None = None


def reference_policy() -> HarnessPolicy:
    """The reference harness's policy, read from the live module constants.

    Built lazily (base.py and delegate_prompts.py import this module lazily
    too, so there is no import cycle) and cached for the process."""
    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE
    from . import delegate_prompts as dp
    from .providers import base as b

    _REFERENCE = HarnessPolicy(
        common_base=dp._COMMON_BASE,
        tool_category_notes=dict(dp._TOOL_CATEGORY_NOTES),
        type_guidance=dict(dp._TYPE_GUIDANCE),
        small_context_window=dp._SMALL_CONTEXT_WINDOW,
        small_context_note=dp._SMALL_CONTEXT_NOTE,
        verify_enabled=True,
        verify_step_prompt=dp.VERIFY_STEP_PROMPT,
        code_file_suffixes=tuple(b._CODE_FILE_SUFFIXES),
        verify_follow_up_max_turns=8,
        review_enabled=True,
        review_step_prompt=dp.REVIEW_STEP_PROMPT,
        review_follow_up_max_turns=4,
        compaction_inline_prompt=b.COMPACTION_INLINE_PROMPT,
        prune_protect_recent_chars=b._PRUNE_PROTECT_RECENT_CHARS,
        prune_min_savings=b._PRUNE_MIN_SAVINGS,
        compact_retain_user_chars=b._COMPACT_RETAIN_USER_CHARS,
        compact_retain_turns=b._COMPACT_RETAIN_TURNS,
        max_compactions_per_run=b._MAX_COMPACTIONS_PER_RUN,
        compaction_min_reduction=b._COMPACTION_MIN_REDUCTION,
        compact_summary_max_tokens=b._COMPACT_SUMMARY_MAX_TOKENS,
        compact_overflow_retries=b._COMPACT_OVERFLOW_RETRIES,
        compact_drop_fraction=b._COMPACT_DROP_FRACTION,
        max_hard_resets_per_run=b._MAX_HARD_RESETS_PER_RUN,
        reset_digest_cap_chars=b._RESET_DIGEST_CAP_CHARS,
        repeat_call_limit=b._DEFAULT_REPEAT_CALL_LIMIT,
        oscillation_window=b._OSCILLATION_WINDOW,
        per_turn_input_max=b._DEFAULT_PER_TURN_INPUT_MAX,
        loop_warning_prompt=b.LOOP_WARNING_PROMPT,
        truncated_tool_call_prompt=b.TRUNCATED_TOOL_CALL_PROMPT,
        headroom_levels=tuple(b._HEADROOM_LEVELS),
        headroom_note_template=b.HEADROOM_NOTE_TEMPLATE,
        parallel_safe_tools=frozenset(b.PARALLEL_SAFE_TOOLS),
        max_parallel_tool_calls=b._MAX_PARALLEL_TOOL_CALLS,
        tool_bundles=None,
    )
    return _REFERENCE


def as_policy(value: Any) -> HarnessPolicy:
    """Coerce whatever a caller holds (None, a policy, a dict overlay — the
    worker manifest carries a dict) into a HarnessPolicy. Never raises."""
    if isinstance(value, HarnessPolicy):
        return value
    if isinstance(value, Mapping):
        return policy_from_overlay(value)
    return reference_policy()


def _template_ok(text: str, args: dict[str, Any]) -> bool:
    """The template uses exactly the required placeholders and formats."""
    import string
    try:
        names = {f for _, f, _, _ in string.Formatter().parse(text) if f}
        if names != set(args):
            return False
        text.format(**args)
        return True
    except (KeyError, IndexError, ValueError):
        return False


def policy_from_overlay(overlay: Mapping[str, Any] | None,
                        base: HarnessPolicy | None = None) -> HarnessPolicy:
    """Apply a distro's (partial) policy dict over ``base`` (the reference).

    Validation is per key: a bad value keeps the base value and logs once.
    """
    base = base or reference_policy()
    if not isinstance(overlay, Mapping) or not overlay:
        return base
    known = {f.name for f in fields(HarnessPolicy)}
    changes: dict[str, Any] = {}
    for key, raw in overlay.items():
        if key not in known:
            continue
        cur = getattr(base, key)
        try:
            val = _coerce(key, raw, cur)
        except (TypeError, ValueError) as exc:
            log.warning("harness policy: ignoring %s (%s)", key, exc)
            continue
        if val != cur:
            changes[key] = val
    return replace(base, **changes) if changes else base


def _coerce(key: str, raw: Any, cur: Any) -> Any:
    if key == "tool_bundles":
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)):
            raise TypeError("tool_bundles must be a list or null")
        return tuple(sorted({str(x) for x in raw if isinstance(x, str) and x}))
    if isinstance(cur, bool):
        if not isinstance(raw, bool):
            raise TypeError("expected a boolean")
        return raw
    if isinstance(cur, int):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("expected an integer")
        lo, hi = _INT_BOUNDS.get(key, (0, 10_000_000))
        return max(lo, min(hi, int(raw)))
    if isinstance(cur, float):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("expected a number")
        lo, hi = _FLOAT_BOUNDS.get(key, (0.0, 1.0))
        return max(lo, min(hi, float(raw)))
    if isinstance(cur, str):
        if not isinstance(raw, str):
            raise TypeError("expected a string")
        if len(raw) > 200_000:
            raise ValueError("string too long")
        args = _TEMPLATE_KEYS.get(key)
        if args is not None and not _template_ok(raw, args):
            raise ValueError("template placeholders do not match")
        return raw
    if isinstance(cur, frozenset):
        if not isinstance(raw, (list, tuple)):
            raise TypeError("expected a list of names")
        return frozenset(str(x) for x in raw if isinstance(x, str) and x)
    if key == "headroom_levels":
        if not isinstance(raw, (list, tuple)):
            raise TypeError("expected a list of fractions")
        vals = sorted({float(x) for x in raw
                       if isinstance(x, (int, float)) and not isinstance(x, bool)
                       and 0.0 < float(x) <= 1.0})
        return tuple(vals)
    if key == "code_file_suffixes":
        if not isinstance(raw, (list, tuple)):
            raise TypeError("expected a list of suffixes")
        return tuple(str(x) for x in raw if isinstance(x, str) and x)
    if isinstance(cur, Mapping):
        if not isinstance(raw, Mapping):
            raise TypeError("expected an object of strings")
        merged = dict(cur)
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str):
                merged[k] = v
        return merged
    raise TypeError("unsupported field")


# ---------------------------------------------------------------------------
# Distro config blob
# ---------------------------------------------------------------------------

def distro_config(name: str, version: str, modules: list[str],
                  policy: HarnessPolicy | None = None) -> dict[str, Any]:
    """The JSON a distro manifest's code blob carries."""
    return {
        "kind": DISTRO_KIND,
        "policy_version": POLICY_VERSION,
        "name": name,
        "version": version,
        "modules": sorted(modules),
        "policy": (policy or reference_policy()).to_dict(),
    }


def encode_distro_config(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, indent=1, ensure_ascii=False)


def parse_distro_config(raw: bytes | str | None) -> dict[str, Any] | None:
    """The distro config dict, or None when the blob is not one."""
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        data = json.loads(text)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("kind") != DISTRO_KIND:
        return None
    return data


# Policy cache keyed by the distro's code digest (content-addressed, so a
# cached entry can never go stale).
_POLICY_CACHE: dict[str, HarnessPolicy] = {}


def policy_for_record(store: Any, record: Any) -> HarnessPolicy:
    """The policy a distro record carries; the reference on any fault."""
    try:
        code_digest = str(record.manifest.get("code_digest") or "")
        if not code_digest:
            return reference_policy()
        hit = _POLICY_CACHE.get(code_digest)
        if hit is not None:
            return hit
        raw = store._blob_store().get_bytes(code_digest)
        cfg = parse_distro_config(raw)
        pol = (policy_from_overlay(cfg.get("policy"))
               if cfg is not None else reference_policy())
        _POLICY_CACHE[code_digest] = pol
        return pol
    except Exception:  # noqa: BLE001 — a broken distro never breaks a run
        log.warning("harness policy: distro unreadable; using reference",
                    exc_info=True)
        return reference_policy()


def agent_loadout(runtime: Any, agent_id: str) -> str:
    """The distro digest bound to ``agent_id`` ("" when none)."""
    try:
        defn = runtime.get_agent(agent_id) if runtime is not None else None
    except Exception:  # noqa: BLE001
        defn = None
    return str(getattr(defn, "loadout", "") or "") if defn is not None else ""


def policy_for_agent(runtime: Any, defn: Any) -> HarnessPolicy:
    """The policy for one agent run: its bound distro, else the reference."""
    digest = str(getattr(defn, "loadout", "") or "")
    store = getattr(runtime, "tool_store", None) if runtime is not None else None
    if not digest or store is None:
        return reference_policy()
    if digest == getattr(store, "active_loadout", ""):
        # The daemon's own reference distro: its blob IS reference_policy().
        return reference_policy()
    record = store.resolve(digest) if hasattr(store, "resolve") else None
    if record is None or not getattr(record, "enabled", True):
        log.warning("agent %s: bound distro %s not available; using reference",
                    getattr(defn, "id", "?"), digest[:16])
        return reference_policy()
    return policy_for_record(store, record)
