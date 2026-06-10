"""Tests for the DelegateRegistry — agent CRUD, hierarchy, and persistence."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atn.agent_registry import DelegateNode, DelegateRegistry, DelegateStatus


class TestDelegateNode:
    """Tests for the DelegateNode dataclass."""

    def test_default_status_is_pending(self):
        node = DelegateNode(
            agent_id="a.1", parent_id="a", agent_type="implement",
            prompt="Do stuff", title="Stuff",
        )
        assert node.status == DelegateStatus.PENDING
        assert node.completed_at is None
        assert node.error is None

    def test_to_dict_contains_all_fields(self):
        node = DelegateNode(
            agent_id="a.1", parent_id="a", agent_type="explore",
            prompt="Find things", title="Find",
            status=DelegateStatus.RUNNING, tokens_used=100, tool_calls=3,
        )
        d = node.to_dict()
        assert d["agent_id"] == "a.1"
        assert d["parent_id"] == "a"
        assert d["agent_type"] == "explore"
        assert d["status"] == "running"
        assert d["tokens_used"] == 100
        assert d["tool_calls"] == 3
        assert d["completed_at"] is None

    def test_to_dict_truncates_prompt(self):
        long_prompt = "x" * 1000
        node = DelegateNode(
            agent_id="a.1", parent_id="a", agent_type="implement",
            prompt=long_prompt, title="Long",
        )
        d = node.to_dict()
        assert len(d["prompt"]) == 500

    def test_from_dict_roundtrip(self):
        node = DelegateNode(
            agent_id="a.1.2", parent_id="a.1", agent_type="debug",
            prompt="Fix bug", title="Fix",
            status=DelegateStatus.COMPLETED,
            result_preview="Fixed!", tokens_used=5000, tool_calls=12,
        )
        d = node.to_dict()
        restored = DelegateNode.from_dict(d)
        assert restored.agent_id == node.agent_id
        assert restored.parent_id == node.parent_id
        assert restored.agent_type == node.agent_type
        assert restored.status == node.status
        assert restored.result_preview == node.result_preview
        assert restored.tokens_used == node.tokens_used
        assert restored.tool_calls == node.tool_calls

    def test_from_dict_with_minimal_fields(self):
        d = {"agent_id": "a.1", "created_at": datetime.now(timezone.utc).isoformat()}
        node = DelegateNode.from_dict(d)
        assert node.agent_id == "a.1"
        assert node.parent_id is None
        assert node.agent_type == "implement"
        assert node.status == DelegateStatus.PENDING


class TestDelegateRegistryRegistration:
    """Tests for register, unregister, and ID generation."""

    # NOTE: child-id generation moved off DelegateRegistry to the
    # runtime AgentRegistry (rt.generate_child_id). Sequential/nested
    # coverage lives in test_cognitive_mode.TestGenerateChildId, and
    # restart-rebuild coverage in
    # TestGenerateChildId.test_counters_rebuild_from_registered_agents.

    def test_register_creates_node(self):
        reg = DelegateRegistry()
        node = reg.register("x.1", "x", "explore", "Search", "Search task")
        assert node.agent_id == "x.1"
        assert node.parent_id == "x"
        assert node.agent_type == "explore"
        assert node.prompt == "Search"
        assert node.title == "Search task"
        assert node.status == DelegateStatus.PENDING

    def test_register_default_title_uses_agent_id(self):
        reg = DelegateRegistry()
        node = reg.register("x.1", "x", "implement", "Do stuff")
        assert node.title == "x.1"

    def test_get_node_returns_none_for_missing(self):
        reg = DelegateRegistry()
        assert reg.get_node("nonexistent") is None

    def test_remove_single_node(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        assert reg.remove("a.1") is True
        assert reg.get_node("a.1") is None

    def test_remove_nonexistent_returns_false(self):
        reg = DelegateRegistry()
        assert reg.remove("nope") is False

    def test_remove_cascades_to_descendants(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        reg.register("a.1.1", "a.1", "debug", "Sub", "S")
        reg.register("a.1.1.1", "a.1.1", "review", "Deep", "D")
        assert reg.remove("a.1") is True
        assert reg.get_node("a.1") is None
        assert reg.get_node("a.1.1") is None
        assert reg.get_node("a.1.1.1") is None


class TestDelegateRegistryStatusUpdates:
    """Tests for status transitions and metadata updates."""

    def test_update_status_basic(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        node = reg.update_status("a.1", DelegateStatus.RUNNING)
        assert node.status == DelegateStatus.RUNNING
        assert node.completed_at is None  # RUNNING doesn't set completed_at

    def test_update_status_completed_sets_timestamp(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        node = reg.update_status("a.1", DelegateStatus.COMPLETED, result_preview="Done")
        assert node.status == DelegateStatus.COMPLETED
        assert node.completed_at is not None
        assert node.result_preview == "Done"

    def test_update_status_failed_sets_error(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        node = reg.update_status("a.1", DelegateStatus.FAILED, error="crash")
        assert node.status == DelegateStatus.FAILED
        assert node.error == "crash"
        assert node.completed_at is not None

    def test_update_status_killed_sets_timestamp(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        node = reg.update_status("a.1", DelegateStatus.KILLED)
        assert node.completed_at is not None

    def test_update_status_unknown_id_returns_none(self):
        reg = DelegateRegistry()
        assert reg.update_status("nope", DelegateStatus.RUNNING) is None

    def test_result_preview_truncated(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        long_preview = "x" * 3000
        node = reg.update_status("a.1", DelegateStatus.COMPLETED, result_preview=long_preview)
        assert len(node.result_preview) == 2000


class TestDelegateRegistryTreeQueries:
    """Tests for tree navigation: children, descendants, active."""

    def setup_method(self):
        self.reg = DelegateRegistry()
        self.reg.register("a.1", "a", "implement", "Task 1", "T1")
        self.reg.register("a.2", "a", "explore", "Task 2", "T2")
        self.reg.register("a.1.1", "a.1", "debug", "Subtask 1", "S1")
        self.reg.register("a.1.2", "a.1", "review", "Subtask 2", "S2")
        self.reg.register("a.1.1.1", "a.1.1", "implement", "Deep", "D")

    def test_get_children(self):
        children = self.reg.get_children("a")
        assert len(children) == 2
        assert {c.agent_id for c in children} == {"a.1", "a.2"}

    def test_get_children_of_leaf(self):
        assert self.reg.get_children("a.2") == []

    def test_get_descendants(self):
        desc = self.reg.get_descendants("a")
        assert len(desc) == 5

    def test_get_descendants_partial(self):
        desc = self.reg.get_descendants("a.1")
        assert len(desc) == 3
        ids = {d.agent_id for d in desc}
        assert ids == {"a.1.1", "a.1.2", "a.1.1.1"}

    def test_get_active(self):
        # All start as PENDING (active)
        active = self.reg.get_active()
        assert len(active) == 5

        self.reg.update_status("a.1", DelegateStatus.COMPLETED)
        self.reg.update_status("a.2", DelegateStatus.FAILED)
        active = self.reg.get_active()
        assert len(active) == 3

    def test_get_tree_structure(self):
        tree = self.reg.get_tree()
        assert tree["total_count"] == 5
        assert tree["active_count"] == 5
        assert len(tree["nodes"]) == 5


class TestDelegateRegistryCleanup:
    """Tests for cleanup_orphans and clear."""

    def test_cleanup_orphans_kills_running(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        reg.update_status("a.1", DelegateStatus.RUNNING)
        reg.register("a.2", "a", "explore", "Task 2", "T2")
        # a.2 is PENDING (also "active")

        count = reg.cleanup_orphans()
        assert count == 2
        assert reg.get_node("a.1").status == DelegateStatus.KILLED
        assert reg.get_node("a.2").status == DelegateStatus.KILLED

    def test_cleanup_orphans_ignores_completed(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        reg.update_status("a.1", DelegateStatus.COMPLETED)
        count = reg.cleanup_orphans()
        assert count == 0

    def test_clear_removes_everything(self):
        reg = DelegateRegistry()
        reg.register("a.1", "a", "implement", "Task", "T")
        reg.register("a.2", "a", "explore", "Task 2", "T2")
        reg.clear()
        assert reg.get_node("a.1") is None
        assert reg.get_node("a.2") is None


class TestDelegateRegistryPersistence:
    """Tests for save/load with JSON file."""

    def test_save_and_load(self, tmp_path: Path):
        store = tmp_path / "delegates.json"
        reg = DelegateRegistry(store_path=store)
        reg.register("a.1", "a", "implement", "Write code", "Code")
        reg.update_status("a.1", DelegateStatus.COMPLETED, result_preview="Done")
        reg.register("a.2", "a", "explore", "Search", "Search")
        reg.save()

        assert store.exists()
        data = json.loads(store.read_text())
        assert data["version"] == "1.0"
        assert len(data["nodes"]) == 2

        # Load into fresh registry
        reg2 = DelegateRegistry(store_path=store)
        reg2.load()
        assert reg2.get_node("a.1").status == DelegateStatus.COMPLETED
        assert reg2.get_node("a.2") is not None

    # NOTE: counter-rebuild-on-load moved with child-id generation to
    # the runtime AgentRegistry (register_agent bumps counters past
    # hierarchical ids); see test_counters_rebuild_from_registered_agents
    # in test_cognitive_mode.py.

    def test_save_without_store_path_is_noop(self):
        reg = DelegateRegistry(store_path=None)
        reg.register("a.1", "a", "implement", "T", "T")
        reg.save()  # Should not raise

    def test_load_without_store_path_is_noop(self):
        reg = DelegateRegistry(store_path=None)
        reg.load()  # Should not raise

    def test_load_nonexistent_file_is_noop(self, tmp_path: Path):
        store = tmp_path / "missing.json"
        reg = DelegateRegistry(store_path=store)
        reg.load()  # Should not raise
        assert reg.get_tree()["total_count"] == 0

    def test_load_corrupt_file_is_handled(self, tmp_path: Path):
        store = tmp_path / "delegates.json"
        store.write_text("not valid json!!!")
        reg = DelegateRegistry(store_path=store)
        reg.load()  # Should not raise
        assert reg.get_tree()["total_count"] == 0
