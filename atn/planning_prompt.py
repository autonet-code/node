"""Planning review context builder.

The planning loop periodically triggers the orchestrator with a work message
containing a context block built from this module.  The orchestrator then
reviews goals, proposes tasks, and allocates unused credit budget.
"""
from __future__ import annotations

from typing import Any

PLANNING_REVIEW_INSTRUCTION = """\
This is an automated planning review.  Review the user's goals, projects, and \
credit budget below.  Then:

1. **Assess goal progress** — Which goals have active agents or tasks?  Which are \
stalled?  Update statuses if needed (use update_goal).

2. **Check credit budget** — If utilization is low and auto_allocate is enabled, \
propose new tasks that would move the user toward their goals.  Don't waste budget \
on low-value work — only propose tasks that meaningfully advance a goal.

3. **Propose tasks** — For each proposed task, specify:
   - Which goal it serves
   - Whether it's an automation (create an agent), calendar event (block the user's \
time), or reminder
   - A clear description of what it will accomplish

4. **Review calendar** — If the Google Calendar connector is available, check the \
user's upcoming schedule.  Identify free time that could be allocated to goal work.  \
Don't over-schedule.

5. **Report** — Summarize what you found and what you propose.  Keep it concise.

Use the planning tools: get_goals, get_projects, get_credit_budget, propose_task, \
list_tasks.  If a connector is available, use use_connector to check the calendar.\
"""


def build_planning_context(
    *,
    goals: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    strengths: list[str],
    weaknesses: list[str],
    budget_summary: dict[str, Any],
    active_agents: list[dict[str, Any]],
    pending_tasks: list[dict[str, Any]],
    calendar_available: bool,
) -> str:
    """Build the context block injected into the planning review message.

    This is a deterministic text block — no LLM call.  The orchestrator's
    cognitive step receives it as the inbox message content.
    """
    lines: list[str] = [PLANNING_REVIEW_INSTRUCTION, ""]

    # Goals
    lines.append("## Current Goals")
    if goals:
        for g in goals:
            status = g.get("status", "active")
            tf = g.get("timeframe", "")
            lines.append(f"- [{status}] {g.get('title', '?')} ({tf}): {g.get('description', '')}")
    else:
        lines.append("- No goals defined yet.")
    lines.append("")

    # Projects
    lines.append("## Current Projects")
    if projects:
        for p in projects:
            status = p.get("status", "active")
            lines.append(f"- [{status}] {p.get('title', '?')}: {p.get('description', '')}")
            ns = p.get("next_steps")
            if ns:
                lines.append(f"  Next steps: {ns}")
    else:
        lines.append("- No projects defined yet.")
    lines.append("")

    # Strengths/weaknesses
    if strengths or weaknesses:
        lines.append("## User Profile")
        if strengths:
            lines.append("Strengths: " + ", ".join(strengths))
        if weaknesses:
            lines.append("Weaknesses/growth areas: " + ", ".join(weaknesses))
        lines.append("")

    # Budget
    lines.append("## Credit Budget")
    if budget_summary:
        for pid, info in budget_summary.items():
            util_pct = info.get("utilization", 0) * 100
            limit = info.get("limit", 0)
            used = info.get("used", 0)
            auto = "auto-allocate ON" if info.get("auto_allocate") else "auto-allocate OFF"
            lines.append(f"- {pid}: {used:,}/{limit:,} tokens ({util_pct:.1f}% used, {auto})")
    else:
        lines.append("- No budgets configured.")
    lines.append("")

    # Active agents
    lines.append("## Active Agents")
    if active_agents:
        for a in active_agents:
            lines.append(f"- {a.get('name', a.get('id', '?'))} [{a.get('status', '?')}]")
    else:
        lines.append("- No active agents.")
    lines.append("")

    # Pending tasks
    lines.append("## Pending Planning Tasks")
    if pending_tasks:
        for t in pending_tasks:
            lines.append(f"- [{t.get('status', '?')}] {t.get('title', '?')} (goal: {t.get('goal_id', '?')})")
    else:
        lines.append("- No pending tasks.")
    lines.append("")

    # Calendar
    if calendar_available:
        lines.append("## Calendar")
        lines.append("Google Calendar is connected.  Use use_connector(\"google_calendar\", ...) to check schedule.")
    else:
        lines.append("## Calendar")
        lines.append("Google Calendar not connected.  Calendar-based planning unavailable.")

    return "\n".join(lines)
