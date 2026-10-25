"""Importing a Claude Code session as an agent (atn/claude_sessions.py)."""
from __future__ import annotations

import json
import os
import time

import pytest

from atn import claude_sessions as cs

SID = "11111111-2222-3333-4444-555555555555"
SID2 = "66666666-7777-8888-9999-000000000000"


def _transcript(path, sid, *, model="claude-sonnet-5-5", title="Fix the parser"):
    rows = [
        {"type": "user", "sessionId": sid, "cwd": "x", "entrypoint": "cli",
         "message": {"role": "user", "content": "please fix the parser"}},
        {"type": "assistant", "sessionId": sid, "message": {"model": model}},
        {"type": "ai-title", "aiTitle": "Parser work", "sessionId": sid},
        {"type": "ai-title", "aiTitle": title, "sessionId": sid},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def project(tmp_path, monkeypatch):
    home = tmp_path / "claude"
    folder = tmp_path / "work" / "my repo"
    folder.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    pdir = cs.project_dir(folder)
    pdir.mkdir(parents=True)
    for sid in (SID, SID2):
        p = pdir / f"{sid}.jsonl"
        _transcript(p, sid)
        old = time.time() - 3600
        os.utime(p, (old, old))
    return folder


def test_slug_matches_claude_code():
    assert cs.folder_slug(r"C:\code\autonet").endswith("C--code-autonet")


def test_lists_sessions_with_title_and_model(project):
    out = cs.list_sessions(project, processes=[])
    assert out["exists"] and len(out["sessions"]) == 2
    s = out["sessions"][0]
    assert s["title"] == "Fix the parser"          # the latest generated title
    assert s["model"] == "claude-sonnet-5-5"
    assert s["first_prompt"] == "please fix the parser"
    assert not s["in_use"]


def test_a_claude_process_in_the_folder_blocks_import(project):
    procs = [{"pid": 42, "cwd": str(project), "session_id": ""}]
    info, why = cs.import_check(str(project), SID, processes=procs)
    assert "in use" in why and "pid 42" in why


def test_a_named_session_blocks_only_itself(project):
    procs = [{"pid": 7, "cwd": str(project), "session_id": SID2}]
    _, why = cs.import_check(str(project), SID, processes=procs)
    assert why == ""
    _, why2 = cs.import_check(str(project), SID2, processes=procs)
    assert "has this session open" in why2


def test_recently_written_session_is_in_use(project):
    p = cs.project_dir(project) / f"{SID}.jsonl"
    os.utime(p, None)
    _, why = cs.import_check(str(project), SID, processes=[])
    assert "last minute" in why


def test_unknown_session(project):
    _, why = cs.import_check(str(project), "99999999-2222-3333-4444-555555555555",
                             processes=[])
    assert "no session" in why


@pytest.mark.asyncio
async def test_import_makes_a_claude_max_agent(project, tmp_path, monkeypatch):
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime
    from atn.ws_server import WebSocketBridge

    monkeypatch.setattr(cs, "claude_processes", lambda: [])
    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir()
    agents_dir.mkdir()
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    rt = Runtime(EventBus(), data_dir=data_dir, config=config)
    bridge = WebSocketBridge(rt)

    out = await bridge._import_claude_session({"folder": str(project), "session_id": SID})
    assert out["ok"], out
    defn = rt.get_agent(out["result"]["agent_id"])
    assert defn.provider == "claude_max" and defn.cognitive_model == "claude-sonnet-5-5"
    assert defn.claude_session["session_id"] == SID
    assert defn.name == "Fix the parser"

    again = await bridge._import_claude_session({"folder": str(project), "session_id": SID})
    assert not again["ok"] and "already imported" in again["error"]

    from atn.loader import load_agents_dir
    loaded = {d.id: d for d in load_agents_dir(agents_dir)[0]}
    assert loaded[defn.id].claude_session == defn.claude_session


def test_transcript_turns_keep_the_conversation_only(tmp_path):
    rows = [
        {"type": "user", "message": {"content": "<command-name>/model</command-name>"}},
        {"type": "user", "message": {"content": [
            {"type": "text", "text": "check the logs<system-reminder>noise</system-reminder>"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Looking."}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "x"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Found it."}]}},
        {"type": "user", "isSidechain": True, "message": {"content": "subagent brief"}},
        {"type": "user", "isMeta": True, "message": {"content": "meta"}},
        {"type": "user", "message": {"content": "thanks"}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    assert cs.transcript_turns(p) == [
        ("user", "check the logs"), ("assistant", "Looking.\n\nFound it."), ("user", "thanks")]
    assert cs.transcript_turns(p, limit=1) == [("user", "thanks")]


@pytest.mark.asyncio
async def test_import_shows_the_sessions_history(project, tmp_path, monkeypatch):
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime
    from atn.ws_server import WebSocketBridge

    p = cs.project_dir(project) / f"{SID}.jsonl"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "assistant", "message": {
            "content": [{"type": "text", "text": "Fixed."}]}}) + "\n")
    old = time.time() - 3600
    os.utime(p, (old, old))
    monkeypatch.setattr(cs, "claude_processes", lambda: [])
    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir()
    agents_dir.mkdir()
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    rt = Runtime(EventBus(), data_dir=data_dir, config=config)
    out = await WebSocketBridge(rt)._import_claude_session(
        {"folder": str(project), "session_id": SID})
    assert out["ok"], out
    turns = rt.get_agent_conversation_store(out["result"]["agent_id"]).get_turns()
    assert [(t.role, t.content) for t in turns] == [
        ("user", "please fix the parser"), ("assistant", "Fixed.")]
