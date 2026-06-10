"""Shared fixtures for the atn test suite."""

import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio


def _refuse_real_provider(name):
    def _ctor(*args, **kwargs):
        raise RuntimeError(
            f"test attempted to construct a real {name} — it would spawn "
            "an actual CLI subprocess. Patch the provider (with "
            "patch('atn.runtime.provider_manager." + name + "', ...)) or "
            "make sure the agent run finishes inside your patch block."
        )
    return _ctor


@pytest.fixture(autouse=True)
def _no_real_cli_providers():
    """Fail fast if any test path constructs a real subprocess-backed
    provider. This happened in practice: tests started cognitive runs
    under a ``with patch(BridgeProvider)`` and returned with the run
    still in flight; the run then constructed a REAL bridge after the
    patch exited, spawned the real CLI, and its Windows proactor pipe
    read resisted task cancellation — hanging pytest-asyncio teardown
    indefinitely (and pointing a unit test at the live LLM).

    Tests' own ``with patch(...)`` blocks layer over these stubs and
    keep working unchanged.
    """
    with patch(
        "atn.runtime.provider_manager.BridgeProvider",
        side_effect=_refuse_real_provider("BridgeProvider"),
    ), patch(
        "atn.runtime.provider_manager.CodexBridgeProvider",
        side_effect=_refuse_real_provider("CodexBridgeProvider"),
    ):
        yield


@pytest_asyncio.fixture(autouse=True)
async def _reap_leftover_tasks():
    """Cancel any asyncio task a test leaves running, immediately after
    the test body — while every ``with patch(...)`` the test used is
    still the world the tasks saw.

    Why: several tests start cognitive agent runs (create_agent →
    "running") and return without awaiting completion. Without this,
    those runs keep executing during fixture teardown — AFTER the
    test's BridgeProvider patch has exited — so they construct a REAL
    bridge provider, spawn a real CLI subprocess, and pytest-asyncio's
    unbounded ``_cancel_all_tasks`` teardown then hangs on Windows
    proactor pipe reads (observed: suite wedged indefinitely at
    teardown). Reaping here kills the run while it still holds the
    mock, which is also what keeps a unit test from ever invoking the
    real LLM CLI.
    """
    yield
    me = asyncio.current_task()
    leftovers = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
    if not leftovers:
        return
    for t in leftovers:
        t.cancel()
    done, pending = await asyncio.wait(leftovers, timeout=10)
    if pending:
        details = []
        for t in pending:
            where = "; ".join(
                f"{f.f_code.co_filename}:{f.f_lineno} {f.f_code.co_name}"
                for f in t.get_stack()[-3:]
            )
            details.append(f"{t.get_name()} [{where}]")
        raise AssertionError(
            "tasks survived cancellation after the test body: "
            + " | ".join(details)
        )
