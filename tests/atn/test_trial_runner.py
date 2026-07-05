"""Service verifier trials (docs/tool_substrate.md — Verifier trials).

Covers the daemon-facing run_trial flow with FABRICATED prospectuses and
a FAKE service transport: fetch → execute battery → score per the
prospectus's own criteria → blob-store report → return verdict + report
digest + attestTrial calldata. Live service plumbing is thin, so the
transport is injected (a callable seam), exactly as the runner is meant
to be exercised until the WS service-request client lands.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.orchestrator.tools import execute_tool
from atn.trial_runner import PROSPECTUS_KIND, TrialRunner


def _make_runtime(tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


async def _register_agent(rt, agent_id, parent_id=None):
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5", parent_id=parent_id,
    )
    await rt.register_agent(defn)
    return defn


def _result_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _store_prospectus(rt, prospectus: dict) -> str:
    """Blob-store a prospectus locally so fetch_payload resolves it."""
    prospectus = {"kind": PROSPECTUS_KIND, **prospectus}
    return rt.tool_store._blob_store().add_json(prospectus)


class TestRunTrial:
    @pytest.mark.asyncio
    async def test_all_cases_pass_verdict_pass(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")

        good_digest = _result_digest({"answer": 42})
        prospectus = {
            "service_digest": "ab" * 32,
            "endpoint": "wss://venture.example/mcp",
            "battery": [
                {"op": "compute", "args": {"x": 1}, "expect_digest": good_digest},
                {"op": "compute", "args": {"bad": True}, "expect_error": True},
                {"op": "compute", "args": {"x": 2}, "expect_contains": "answer"},
            ],
        }
        pdigest = _store_prospectus(rt, prospectus)

        async def fake_transport(endpoint, op, args):
            assert endpoint == "wss://venture.example/mcp"
            if args.get("bad"):
                return {"error": "boom"}
            return {"result": {"answer": 42}}

        rt.trial_runner.transport = fake_transport
        out = await rt.trial_runner.run_trial("validator", pdigest)

        assert out["verdict"] == "pass"
        assert out["passed"] == 3 and out["total"] == 3
        assert out["report_digest"]
        # attestTrial calldata: selector + 3 packed 32-byte args.
        assert out["attest_calldata"].startswith("0x")
        assert len(out["attest_calldata"]) == 2 + (4 + 32 * 3) * 2
        # Report is blob-stored and reviewable.
        report = rt.tool_store._blob_store().get_json(out["report_digest"])
        assert report["kind"] == "venture_trial_report"
        assert report["verdict"] == "pass"
        assert len(report["cases"]) == 3

    @pytest.mark.asyncio
    async def test_wrong_answer_fails_the_trial(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        prospectus = {
            "battery": [
                {"args": {}, "expect_digest": _result_digest({"answer": 42})},
            ],
        }
        pdigest = _store_prospectus(rt, prospectus)

        async def fake_transport(endpoint, op, args):
            return {"result": {"answer": 0}}   # wrong

        rt.trial_runner.transport = fake_transport
        out = await rt.trial_runner.run_trial("validator", pdigest)
        assert out["verdict"] == "fail"
        assert out["passed"] == 0
        # calldata's last arg byte encodes passed=False.
        assert out["attest_calldata"].endswith("00")

    @pytest.mark.asyncio
    async def test_pass_threshold_partial_credit(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        prospectus = {
            "pass_threshold": 0.5,
            "battery": [
                {"args": {"n": 1}, "expect_contains": "ok"},
                {"args": {"n": 2}, "expect_contains": "ok"},
            ],
        }
        pdigest = _store_prospectus(rt, prospectus)

        async def fake_transport(endpoint, op, args):
            # First passes (contains ok), second doesn't.
            return {"result": ("ok" if args.get("n") == 1 else "no")}

        rt.trial_runner.transport = fake_transport
        out = await rt.trial_runner.run_trial("validator", pdigest)
        assert out["passed"] == 1 and out["total"] == 2
        assert out["pass_rate"] == 0.5
        assert out["verdict"] == "pass"          # meets the 0.5 bar

    @pytest.mark.asyncio
    async def test_sync_transport_supported(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        prospectus = {"battery": [{"args": {}, "expect_contains": "hi"}]}
        pdigest = _store_prospectus(rt, prospectus)

        def sync_transport(endpoint, op, args):   # not a coroutine
            return {"result": "hi there"}

        rt.trial_runner.transport = sync_transport
        out = await rt.trial_runner.run_trial("validator", pdigest)
        assert out["verdict"] == "pass"

    @pytest.mark.asyncio
    async def test_no_transport_fails_loud(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        prospectus = {"battery": [{"args": {}, "expect_contains": "x"}]}
        pdigest = _store_prospectus(rt, prospectus)
        # transport stays None
        out = await rt.trial_runner.run_trial("validator", pdigest)
        assert out["verdict"] == "fail"
        assert "transport" in out["cases"][0]["error"]

    @pytest.mark.asyncio
    async def test_unresolvable_prospectus_errors(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        out = await rt.trial_runner.run_trial("validator", "cd" * 32)
        assert "error" in out and "prospectus" in out["error"]

    @pytest.mark.asyncio
    async def test_wrong_kind_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        # A non-prospectus blob at a valid digest.
        digest = rt.tool_store._blob_store().add_json(
            {"kind": "tool_manifest", "name": "not a prospectus"})
        out = await rt.trial_runner.run_trial("validator", digest)
        assert "error" in out

    @pytest.mark.asyncio
    async def test_empty_battery_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        pdigest = _store_prospectus(rt, {"battery": []})
        out = await rt.trial_runner.run_trial("validator", pdigest)
        assert "error" in out and "battery" in out["error"]

    @pytest.mark.asyncio
    async def test_run_trial_tool_dispatch_and_bundle(self, tmp_path):
        """The agent-facing run_trial threads caller and lands in the
        vetting bundle."""
        from atn.orchestrator.tools import _TOOL_CATEGORIES

        assert "run_trial" in _TOOL_CATEGORIES["vetting"]
        assert "check_evidence" in _TOOL_CATEGORIES["vetting"]

        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        prospectus = {"battery": [{"args": {}, "expect_contains": "ok"}]}
        pdigest = _store_prospectus(rt, prospectus)
        rt.trial_runner.transport = lambda e, o, a: {"result": "ok"}

        out = await execute_tool(
            "run_trial", {"prospectus_digest": pdigest},
            rt, caller_id="validator")
        assert out["verdict"] == "pass"


class TestCredentialsThreaded:
    @pytest.mark.asyncio
    async def test_free_inference_credentials_reach_the_service(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        prospectus = {
            "credentials": {"token": "venture-funded"},
            "battery": [{"args": {"q": "probe"}, "expect_contains": "seen"}],
        }
        pdigest = _store_prospectus(rt, prospectus)

        seen = {}

        async def fake_transport(endpoint, op, args):
            seen.update(args)
            return {"result": "seen:" + json.dumps(args.get("_credentials"))}

        rt.trial_runner.transport = fake_transport
        out = await rt.trial_runner.run_trial("validator", pdigest)
        assert out["verdict"] == "pass"
        # The venture-funded credentials were threaded into the call.
        assert seen["_credentials"] == {"token": "venture-funded"}
