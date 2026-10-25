"""The one system-prompt resolver (atn/system_prompt.py)."""
from __future__ import annotations

from types import SimpleNamespace

from atn.delegate_prompts import build_system_prompt
from atn.system_prompt import system_prompt_parts


def _agent(**kw):
    base = dict(id="a1", parent_id=None, tools=[], system_prompt="", claude_session=None,
                identity=None, loadout="")
    return SimpleNamespace(**{**base, **kw})


def _runtime(constitution=""):
    return SimpleNamespace(tool_store=None, autonet=SimpleNamespace(constitution_text=constitution))


def test_default_is_atns_prompt_and_types_are_gone():
    parts = system_prompt_parts(_runtime(), _agent())
    assert not parts["custom"] and parts["prompt"] == parts["default_body"]
    assert "Generalist Focus" in parts["prompt"]
    # Every agent gets the same prompt, whatever type an old definition named.
    assert build_system_prompt("explore", "x") == build_system_prompt("general", "y")
    assert "agent_type" not in parts["prompt"] and "typed sub-agents" not in parts["prompt"]


def test_custom_prompt_replaces_the_body_and_keeps_the_default_for_reset():
    parts = system_prompt_parts(_runtime(), _agent(system_prompt="Be terse."))
    assert parts["custom"] and parts["body"] == "Be terse."
    assert "Generalist Focus" in parts["default_body"]


def test_constitution_is_prepended_only_for_registered_agents():
    rt = _runtime("Life is precious.")
    plain = system_prompt_parts(rt, _agent())
    assert plain["constitution"] and not plain["constitution_applies"]
    assert "Life is precious." not in plain["prompt"]
    reg = system_prompt_parts(rt, _agent(identity=SimpleNamespace(registered_on_chain=True),
                                         system_prompt="Be terse."))
    assert reg["constitution_applies"]
    assert reg["prompt"].startswith(reg["constitution"]) and reg["prompt"].endswith("Be terse.")


def test_imported_claude_code_session_gets_nothing_from_atn():
    parts = system_prompt_parts(_runtime("x"), _agent(claude_session={"folder": "c:/x"}))
    assert parts["imported"] and parts["prompt"] == ""


def test_new_agent_prompt_follows_the_granted_tools():
    without = system_prompt_parts(_runtime(), None, tools=[])["prompt"]
    with_web = system_prompt_parts(_runtime(), None, tools=["web"])["prompt"]
    assert without != with_web or with_web.startswith(without)
