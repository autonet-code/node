"""Tests for atn.run_summary — deterministic post-run summary extractor."""
from __future__ import annotations

import pytest

from atn.run_summary import extract_run_summary


# ---------------------------------------------------------------------------
# Turn count
# ---------------------------------------------------------------------------

class TestTurnCount:
    def test_basic_turn_count(self):
        summary = extract_run_summary([], max_turns=15, actual_turns=8)
        assert "8/15 turns" in summary

    def test_hit_turn_limit(self):
        summary = extract_run_summary([], max_turns=15, actual_turns=15)
        assert "Hit turn limit" in summary
        assert "may be incomplete" in summary

    def test_no_max_turns(self):
        summary = extract_run_summary([], actual_turns=5)
        assert "5 turns" in summary
        assert "/" not in summary.split("\n")[0]

    def test_no_turn_info(self):
        summary = extract_run_summary([], status="completed")
        assert "completed" in summary.lower()


# ---------------------------------------------------------------------------
# File tracking
# ---------------------------------------------------------------------------

class TestFileTracking:
    def test_files_created(self):
        calls = [
            {"tool": "Write", "args": {"file_path": "/src/new_file.py"}, "result": "ok", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Created:" in summary
        assert "new_file.py" in summary

    def test_files_modified(self):
        calls = [
            {"tool": "Edit", "args": {"file_path": "/src/existing.py"}, "result": "ok", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Modified:" in summary
        assert "existing.py" in summary

    def test_files_read_count(self):
        calls = [
            {"tool": "Read", "args": {"file_path": "/src/a.py"}, "result": "...", "success": True},
            {"tool": "Read", "args": {"file_path": "/src/b.py"}, "result": "...", "success": True},
            {"tool": "Read", "args": {"file_path": "/src/c.py"}, "result": "...", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Files read: 3" in summary
        # Individual paths should NOT appear for reads
        assert "a.py" not in summary
        assert "b.py" not in summary

    def test_dedup_write_then_edit(self):
        """If a file is both written and edited, it should appear only once."""
        calls = [
            {"tool": "Write", "args": {"file_path": "/src/file.py"}, "result": "ok", "success": True},
            {"tool": "Edit", "args": {"file_path": "/src/file.py"}, "result": "ok", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Created:" in summary
        # Should not also appear as modified
        assert "Modified:" not in summary

    def test_read_not_counted_if_also_written(self):
        calls = [
            {"tool": "Read", "args": {"file_path": "/src/file.py"}, "result": "...", "success": True},
            {"tool": "Write", "args": {"file_path": "/src/file.py"}, "result": "ok", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Created:" in summary
        assert "Files read:" not in summary  # Read of the same file shouldn't count


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

class TestCommands:
    def test_notable_command(self):
        calls = [
            {"tool": "Bash", "args": {"command": "python -m pytest tests/"}, "result": "exit code: 0", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Commands:" in summary
        assert "pytest" in summary

    def test_trivial_commands_skipped(self):
        calls = [
            {"tool": "Bash", "args": {"command": "cat /etc/passwd"}, "result": "...", "success": True},
            {"tool": "Bash", "args": {"command": "ls -la"}, "result": "...", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Commands:" not in summary

    def test_failed_command_shows_exit_code(self):
        calls = [
            {"tool": "Bash", "args": {"command": "flutter analyze"}, "result": "exit code: 1", "success": False},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "exit 1" in summary


# ---------------------------------------------------------------------------
# Git commits
# ---------------------------------------------------------------------------

class TestGitCommits:
    def test_git_commit_message_extracted(self):
        calls = [
            {"tool": "Bash", "args": {"command": 'git commit -m "feat: add login page"'},
             "result": "[main abc1234] feat: add login page", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Committed:" in summary
        assert "feat: add login page" in summary

    def test_heredoc_commit_message(self):
        cmd = """git commit -m "$(cat <<'EOF'
fix: resolve null pointer in parser

Co-Authored-By: Claude
EOF
)"
"""
        calls = [
            {"tool": "Bash", "args": {"command": cmd},
             "result": "[main def5678]", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Committed:" in summary
        assert "fix: resolve null pointer" in summary


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_tool_errors_listed(self):
        calls = [
            {"tool": "Edit", "args": {"file_path": "/foo.py"}, "result": "old_string not found in file", "success": False},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Errors:" in summary
        assert "Edit" in summary

    def test_execution_error(self):
        summary = extract_run_summary([], actual_turns=3, error="ProviderError: rate limited")
        assert "Error:" in summary
        assert "rate limited" in summary


# ---------------------------------------------------------------------------
# Web searches
# ---------------------------------------------------------------------------

class TestWebSearches:
    def test_search_topics(self):
        calls = [
            {"tool": "WebSearch", "args": {"query": "python asyncio best practices"}, "result": "...", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Searched:" in summary
        assert "asyncio" in summary


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

class TestAgents:
    def test_agents_spawned(self):
        calls = [
            {"tool": "create_agent", "args": {"name": "test-runner", "agent_type": "implement"},
             "result": "agent-123", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Agents spawned:" in summary
        assert "test-runner" in summary


# ---------------------------------------------------------------------------
# Last action
# ---------------------------------------------------------------------------

class TestLastAction:
    def test_last_action_edit(self):
        calls = [
            {"tool": "Read", "args": {"file_path": "/src/a.py"}, "result": "...", "success": True},
            {"tool": "Edit", "args": {"file_path": "/src/widget.dart"}, "result": "ok", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Last action:" in summary
        assert "editing widget.dart" in summary

    def test_last_action_bash(self):
        calls = [
            {"tool": "Bash", "args": {"command": "npm run build"}, "result": "ok", "success": True},
        ]
        summary = extract_run_summary(calls, actual_turns=1)
        assert "Last action:" in summary
        assert "npm run build" in summary

    def test_no_calls_no_last_action(self):
        summary = extract_run_summary([], actual_turns=0)
        assert "Last action:" not in summary


# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------

class TestSizeCap:
    def test_summary_under_2000_chars(self):
        # Generate a lot of tool calls
        calls = []
        for i in range(100):
            calls.append({
                "tool": "Write",
                "args": {"file_path": f"/src/module_{i}/very_long_component_name_{i}.py"},
                "result": "ok",
                "success": True,
            })
            calls.append({
                "tool": "Bash",
                "args": {"command": f"python -m pytest tests/test_module_{i}.py --verbose --coverage"},
                "result": f"exit code: {i % 2}",
                "success": i % 2 == 0,
            })
        summary = extract_run_summary(calls, max_turns=200, actual_turns=100)
        assert len(summary) <= 2000


# ---------------------------------------------------------------------------
# Integration-style
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_realistic_implementation_run(self):
        """Simulate a realistic agent run."""
        calls = [
            {"tool": "Read", "args": {"file_path": "/app/lib/widget.dart"}, "result": "class Widget...", "success": True},
            {"tool": "Read", "args": {"file_path": "/app/lib/state.dart"}, "result": "class State...", "success": True},
            {"tool": "Grep", "args": {"pattern": "setState"}, "result": "3 matches", "success": True},
            {"tool": "Write", "args": {"file_path": "/app/lib/chat_state.dart"}, "result": "ok", "success": True},
            {"tool": "Edit", "args": {"file_path": "/app/lib/widget.dart"}, "result": "ok", "success": True},
            {"tool": "Edit", "args": {"file_path": "/app/lib/state.dart"}, "result": "ok", "success": True},
            {"tool": "Bash", "args": {"command": "flutter analyze"}, "result": "No issues found! exit code: 0", "success": True},
            {"tool": "Bash", "args": {"command": 'git commit -m "feat: unified chat state management"'},
             "result": "[main abc1234]", "success": True},
        ]
        summary = extract_run_summary(calls, max_turns=15, actual_turns=8)

        assert "8/15 turns" in summary
        assert "Created:" in summary
        assert "chat_state.dart" in summary
        assert "Modified:" in summary
        assert "widget.dart" in summary
        assert "flutter analyze" in summary
        assert 'Committed: "feat: unified chat state management"' in summary
        assert "Last action:" in summary
        assert len(summary) <= 2000

    def test_empty_run(self):
        """Agent that did nothing (e.g. no tools used)."""
        summary = extract_run_summary([], max_turns=20, actual_turns=1, status="completed")
        assert "1/20 turns" in summary
        assert len(summary) < 100

    def test_killed_mid_work(self):
        """Agent killed while still working."""
        calls = [
            {"tool": "Read", "args": {"file_path": "/src/main.py"}, "result": "...", "success": True},
            {"tool": "Edit", "args": {"file_path": "/src/main.py"}, "result": "ok", "success": True},
        ]
        summary = extract_run_summary(
            calls, max_turns=20, actual_turns=20,
            status="killed", error="Interrupted",
        )
        assert "Hit turn limit" in summary
        assert "Error: Interrupted" in summary
