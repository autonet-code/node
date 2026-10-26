"""Cursor paging of an agent's conversation (survey 2026-10-26: paging by
offset from the end repeated turns when new ones arrived between pages)."""
from __future__ import annotations

from atn.conversation import ConversationStore


def _store(tmp_path, n):
    s = ConversationStore(tmp_path)
    for i in range(n):
        s.add_user_turn(f"u{i}")
    return s


def test_pages_do_not_shift_when_turns_arrive(tmp_path):
    s = _store(tmp_path, 10)
    page1, start1, total = s.get_turns_before(None, 4)
    assert [t.content for t in page1] == ["u6", "u7", "u8", "u9"] and start1 == 6
    # New turns arrive between page loads.
    for i in range(10, 13):
        s.add_user_turn(f"u{i}")
    page2, start2, _ = s.get_turns_before(start1, 4)
    assert [t.content for t in page2] == ["u2", "u3", "u4", "u5"] and start2 == 2
    page3, start3, _ = s.get_turns_before(start2, 4)
    assert [t.content for t in page3] == ["u0", "u1"] and start3 == 0


def test_the_ws_tool_returns_indices_and_the_next_cursor(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from atn.agent_tools import _get_agent_conversation
    from atn.models import AgentMode

    s = _store(tmp_path, 5)
    rt = SimpleNamespace(
        get_agent=lambda a: SimpleNamespace(mode=AgentMode.COGNITIVE),
        get_agent_conversation_store=lambda a: s)
    out = asyncio.run(_get_agent_conversation(
        rt, {"agent_id": "x", "limit": 2, "cursor": True}))
    assert [t["index"] for t in out["turns"]] == [3, 4]
    assert out["before"] == 3 and out["has_more"] is True
    out = asyncio.run(_get_agent_conversation(
        rt, {"agent_id": "x", "limit": 5, "before": 3}))
    assert [t["content"] for t in out["turns"]] == ["u0", "u1", "u2"]
    assert out["has_more"] is False
