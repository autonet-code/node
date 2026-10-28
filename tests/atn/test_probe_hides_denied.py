"""Discovery hides a tool only once the network's harm verdict is final;
containment acts on harm still under appeal (2026-10-28)."""
from types import SimpleNamespace

from atn import tool_security
from nodes.common.federated_close_driver import FederatedCloseDriver

DIGEST = "ab" * 32


def _driver(final_harm, cooling_harm):
    drv = object.__new__(FederatedCloseDriver)
    drv.rep_supply = 10_000
    drv.review_snapshot = {
        "tools": {DIGEST: {"harm": {"sum": final_harm * 100, "mass": 100}}} if final_harm else {},
        "cooling_harm": {DIGEST: {"sum": cooling_harm * 100, "mass": 100}} if cooling_harm else {},
    }
    return drv


def _runtime(drv):
    provider = lambda d, include_cooling=True: drv.security_standing(d, include_cooling=include_cooling)
    return SimpleNamespace(tool_store=SimpleNamespace(security_standing_provider=provider),
                           autonet=None)


def test_harm_under_appeal_contains_but_does_not_hide():
    drv = _driver(final_harm=0, cooling_harm=90)
    assert drv.security_standing(DIGEST)["harm_mass"] == 100
    assert tool_security.finally_denied(_runtime(drv), DIGEST) is False


def test_final_harm_hides_the_tool():
    assert tool_security.finally_denied(_runtime(_driver(final_harm=90, cooling_harm=0)), DIGEST)


def test_unknown_standing_is_shown():
    rt = SimpleNamespace(tool_store=SimpleNamespace(security_standing_provider=None))
    assert tool_security.finally_denied(rt, DIGEST) is False
